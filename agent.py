"""
agent.py - CarSense AI: a goal-based, utility-based, learning agent for car buying in India.

Pipeline (Perceive -> Estimate -> Filter/Replan -> Score -> Decide -> Explain -> Learn):
  1. PERCEIVE   : read the buyer profile (form or natural-language query)
  2. ESTIMATE   : 5-year Total Cost of Ownership (TCO) per car, with a Random Forest resale model
  3. FILTER     : hard constraints (budget, seats, safety, fuel, body); auto-replans if too few cars remain
  4. SCORE      : TOPSIS (multi-criteria) + content-based cosine similarity + SVD collaborative score
  5. DECIDE     : weighted hybrid score -> ranked list, with Monte-Carlo stability check
  6. EXPLAIN    : plain-English reason for every recommendation
  7. LEARN      : likes / dislikes update the buyer's priority weights (online learning)
"""
import re
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import CARS_DF, BODY_TYPES, FUEL_TYPES, BRAND_SERVICE

# ----------------------------------------------------------------------------
# Constants (Pune-like prices, Rs)
# ----------------------------------------------------------------------------
FUEL_PRICE = {"Petrol": 104.0, "Diesel": 91.0, "CNG": 90.0, "Hybrid": 104.0}   # Rs per litre / kg
HOME_KWH, PUBLIC_KWH, HOME_SHARE = 8.0, 18.0, 0.8                              # Rs per kWh
REAL_WORLD_FACTOR = 0.85                                                       # claimed -> real mileage
USAGE_ICE = {"City": 0.90, "Mixed": 1.00, "Highway": 1.08}
USAGE_EV = {"City": 1.05, "Mixed": 1.00, "Highway": 0.92}
ONROAD_FACTOR = {"Petrol": 1.15, "Diesel": 1.18, "CNG": 1.15, "Hybrid": 1.17, "Electric": 1.03}
MAINT_FUEL = {"Petrol": 1.0, "Diesel": 1.2, "CNG": 1.1, "Hybrid": 1.05, "Electric": 0.55}
FUEL_DECAY = {"Petrol": 0.0, "Diesel": -0.005, "CNG": 0.01, "Hybrid": -0.01, "Electric": 0.04}

PRIORITIES = ["Economy", "Safety", "Performance", "Space", "Road Presence", "Resale"]
CRIT = ["tco", "cost_km", "ncap", "bhp", "space", "clearance", "resale_pct"]
CRIT_LABELS = ["Ownership-period TCO", "Running cost/km", "Safety (NCAP)", "Power", "Space", "Ground clearance", "Resale value"]
BENEFIT = np.array([False, False, True, True, True, True, True])


def default_profile():
    return {"budget_lakh": 12.0, "seats": 5, "km_per_year": 12000, "usage": "Mixed", "years": 5,
            "fuels": list(FUEL_TYPES), "bodies": [], "min_ncap": 0, "home_charging": True,
            "priorities": {p: 5.0 for p in PRIORITIES}}


def validate_profile(profile):
    """Return a complete, type-safe buyer profile or raise a readable ValueError."""
    p = copy.deepcopy(default_profile())
    if profile:
        p.update(copy.deepcopy(profile))
    try:
        p["budget_lakh"] = float(p["budget_lakh"])
        p["seats"] = int(p["seats"])
        p["km_per_year"] = int(p["km_per_year"])
        p["years"] = int(p["years"])
        p["min_ncap"] = int(p.get("min_ncap", 0))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Budget, seats, annual kilometres, ownership years and safety rating must be numbers.") from exc
    if not 1 <= p["budget_lakh"] <= 200:
        raise ValueError("Budget must be between Rs 1 lakh and Rs 200 lakh.")
    if not 2 <= p["seats"] <= 9:
        raise ValueError("Required seating must be between 2 and 9.")
    if not 1000 <= p["km_per_year"] <= 100000:
        raise ValueError("Annual driving distance must be between 1,000 and 100,000 km.")
    if not 1 <= p["years"] <= 15:
        raise ValueError("Ownership period must be between 1 and 15 years.")
    if p["usage"] not in USAGE_ICE:
        raise ValueError("Usage must be City, Mixed or Highway.")
    if not 0 <= p["min_ncap"] <= 5:
        raise ValueError("Minimum safety rating must be between 0 and 5.")
    p["fuels"] = [f for f in (p.get("fuels") or FUEL_TYPES) if f in FUEL_TYPES]
    if not p["fuels"]:
        p["fuels"] = list(FUEL_TYPES)
    p["bodies"] = [b for b in (p.get("bodies") or []) if b in BODY_TYPES]
    raw_priorities = p.get("priorities") or {}
    p["priorities"] = {k: float(np.clip(float(raw_priorities.get(k, 5.0)), 1, 10)) for k in PRIORITIES}
    p["home_charging"] = bool(p.get("home_charging", True))
    return p


# ----------------------------------------------------------------------------
# 1. Resale-value model (Random Forest)
# ----------------------------------------------------------------------------
def resale_features(df, age, km_year):
    """Feature table for the resale model. age and km_year can be scalars or arrays."""
    X = pd.DataFrame({"ex_showroom": df["ex_showroom"].values, "brand_score": df["brand_score"].values,
                      "ncap": df["ncap"].values, "age": age, "km_k": np.asarray(km_year) / 1000.0})
    for f in FUEL_TYPES:
        X["fuel_" + f] = (df["fuel"].values == f).astype(int)
    for b in BODY_TYPES:
        X["body_" + b] = (df["body"].values == b).astype(int)
    return X


def train_resale_model(seed=7):
    """
    Trains a Random Forest to predict the resale value (% of ex-showroom price) of a used car.
    Training data is SYNTHETIC but calibrated to Indian market patterns (brand trust, fuel type,
    depreciation curve, odometer penalty, hidden per-model effect + noise). Swap in scraped
    used-car listings (Cars24 / Spinny / OLX) to make it fully real.
    """
    rng = np.random.default_rng(seed)
    car_effect = rng.normal(0, 2.0, len(CARS_DF))          # hidden model-specific effect
    recs = []
    for i, c in CARS_DF.iterrows():
        for age in range(1, 9):
            for _ in range(3):
                km = rng.uniform(5000, 25000)
                k = 0.14 - 0.06 * (c.brand_score - 0.5) + FUEL_DECAY[c.fuel]
                pct = 100 * np.exp(-k * age) - 0.0003 * (km - 12000) * np.sqrt(age) + car_effect[i] + rng.normal(0, 1.8)
                recs.append((i, age, km, float(np.clip(pct, 8, 95))))
    long = pd.DataFrame(recs, columns=["idx", "age", "km", "y"])
    base = CARS_DF.loc[long["idx"]].reset_index(drop=True)
    X = resale_features(base, long["age"].values, long["km"].values)
    y = long["y"].values

    # split by CAR MODEL so the test set contains cars the model has never seen
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
    tr, te = next(gss.split(X, y, groups=long["idx"]))
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=0)
    rf.fit(X.iloc[tr], y[tr])
    pred = rf.predict(X.iloc[te])
    metrics = {"MAE (pct points)": mean_absolute_error(y[te], pred),
               "RMSE (pct points)": float(np.sqrt(mean_squared_error(y[te], pred))),
               "R2": r2_score(y[te], pred), "train rows": len(tr), "test rows (unseen models)": len(te)}
    rf.fit(X, y)   # final model uses all data
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return rf, metrics, importances, (y[te], pred)


# ----------------------------------------------------------------------------
# 2. Total Cost of Ownership
# ----------------------------------------------------------------------------
def compute_costs(profile, model):
    """Adds running cost, on-road price, maintenance, insurance, resale and ownership-period TCO for every car."""
    df = CARS_DF.copy()
    years, km, usage = profile["years"], profile["km_per_year"], profile["usage"]

    real_kmpl = (df["mileage"] * REAL_WORLD_FACTOR * USAGE_ICE[usage]).replace(0, np.nan)
    ice_cost_km = df["fuel"].map(FUEL_PRICE) / real_kmpl
    kwh_km = df["battery_kwh"] / (df["ev_range"].replace(0, np.nan) * REAL_WORLD_FACTOR)
    kwh_price = HOME_SHARE * HOME_KWH + (1 - HOME_SHARE) * PUBLIC_KWH if profile["home_charging"] else PUBLIC_KWH
    ev_cost_km = kwh_price * kwh_km / USAGE_EV[usage]
    df["cost_km"] = np.where(df["fuel"] == "Electric", ev_cost_km, ice_cost_km)

    df["onroad"] = df["ex_showroom"] * 1e5 * df["fuel"].map(ONROAD_FACTOR)
    df["onroad_lakh"] = df["onroad"] / 1e5
    df["fuel_cost"] = df["cost_km"] * km * years

    base_service = (3000 + 900 * df["ex_showroom"]) * df["brand"].map(BRAND_SERVICE) * df["fuel"].map(MAINT_FUEL)
    base_service = base_service * (0.7 + 0.3 * km / 10000)
    df["maint_cost"] = base_service * sum(1.06 ** t for t in range(years))
    df["ins_cost"] = df["ex_showroom"] * 1e5 * 0.032 * sum(0.9 ** t for t in range(years))

    df["resale_pct"] = model.predict(resale_features(df, years, km))
    df["resale_value"] = df["ex_showroom"] * 1e5 * df["resale_pct"] / 100
    df["depreciation"] = df["onroad"] - df["resale_value"]

    df["tco"] = df["depreciation"] + df["fuel_cost"] + df["maint_cost"] + df["ins_cost"]
    df["tco_per_km"] = df["tco"] / (km * years)
    df["space"] = df["boot_l"] / 100 + (df["seats"] - 4)
    return df


def priority_scores(df):
    """Normalised 0-1 score of every car on the six buyer priorities."""
    def mm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0 + 0.5
    return pd.DataFrame({"Economy": 1 - mm(df["tco"]), "Safety": df["ncap"] / 5, "Performance": mm(df["bhp"]),
                         "Space": mm(df["space"]), "Road Presence": mm(df["clearance"]),
                         "Resale": mm(df["resale_pct"])}, index=df.index)


# ----------------------------------------------------------------------------
# 3. Scoring algorithms
# ----------------------------------------------------------------------------
def topsis(X, weights, is_benefit):
    """TOPSIS: closeness to the ideal solution (1 = ideal, 0 = worst)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.array([], dtype=float)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    norm = X / denom
    V = norm * weights
    best = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    worst = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
    d_best = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst + 1e-12)


def entropy_weights(X):
    """Objective weights: criteria that vary more across the shortlist carry more information."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return np.ones(X.shape[1], dtype=float) / max(X.shape[1], 1)
    denom = X.sum(axis=0)
    denom[denom == 0] = 1.0
    P = np.clip(X / denom, 1e-12, None)
    e = -(P * np.log(P)).sum(axis=0) / np.log(len(X))
    d = 1 - e
    return d / d.sum() if d.sum() > 0 else np.ones(X.shape[1]) / X.shape[1]


def criteria_weights(priorities):
    p = {k: max(v, 0.01) for k, v in priorities.items()}
    w = np.array([0.6 * p["Economy"], 0.4 * p["Economy"], p["Safety"], p["Performance"],
                  p["Space"], p["Road Presence"], p["Resale"]])
    return w / w.sum()


def minmax(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    r = a.max() - a.min()
    return (a - a.min()) / r if r > 0 else np.full(len(a), 0.5)


# content-based feature space (built once over the whole catalogue)
NUM_COLS = ["ex_showroom", "bhp", "boot_l", "seats", "clearance", "ncap"]
_lo, _hi = CARS_DF[NUM_COLS].min(), CARS_DF[NUM_COLS].max()
CONTENT = pd.concat([(CARS_DF[NUM_COLS] - _lo) / (_hi - _lo),
                     pd.get_dummies(CARS_DF["body"]).astype(float).reindex(columns=BODY_TYPES, fill_value=0),
                     pd.get_dummies(CARS_DF["fuel"]).astype(float).reindex(columns=FUEL_TYPES, fill_value=0)], axis=1)


def ideal_vector(profile, priorities, liked):
    """The buyer's 'dream car' in the same feature space as the catalogue."""
    def sc(col, v):
        return float(np.clip((v - _lo[col]) / (_hi[col] - _lo[col]), 0, 1))
    num = [sc("ex_showroom", profile["budget_lakh"] / 1.15 * 0.85), priorities["Performance"] / 10,
           priorities["Space"] / 10, sc("seats", profile["seats"]), priorities["Road Presence"] / 10,
           priorities["Safety"] / 10]
    body = [1.0 if b in profile["bodies"] else 0.0 for b in BODY_TYPES] if profile["bodies"] else [0.3] * len(BODY_TYPES)
    fuel = [1.0 if f in profile["fuels"] else 0.0 for f in FUEL_TYPES] if len(profile["fuels"]) < len(FUEL_TYPES) else [0.3] * len(FUEL_TYPES)
    v = np.array(num + body + fuel)
    liked = [i for i in liked if i in CONTENT.index]
    if liked:
        v = 0.5 * v + 0.5 * CONTENT.loc[liked].values.mean(axis=0)   # blend dream car with liked cars
    return v


def cosine(a, M):
    return (M @ a) / (np.linalg.norm(M, axis=1) * np.linalg.norm(a) + 1e-12)


# collaborative filtering on SIMULATED users (SVD matrix completion)
ARCHETYPES = {"Value family": [6, 7, 2, 6, 2, 5], "Safety-first family": [4, 10, 2, 7, 3, 4],
              "City commuter": [10, 4, 2, 2, 2, 6], "Enthusiast": [2, 6, 10, 3, 5, 3],
              "Adventurer": [3, 6, 6, 5, 10, 4], "Resale-focused": [3, 6, 5, 6, 5, 10],
              "Large family": [4, 7, 3, 10, 4, 4]}


def build_peer_model(base_df, n_users=350, rank=8, seed=11):
    """
    Simulates a user x car rating matrix (25% observed), completes it with truncated SVD and keeps
    the user weight vectors. A new buyer is matched to similar simulated users (k-NN) and receives
    their predicted ratings ('people like you liked ...').
    """
    rng = np.random.default_rng(seed)
    ps = priority_scores(base_df).values
    arch = list(ARCHETYPES.values())
    W = np.array([rng.dirichlet(np.array(arch[rng.integers(len(arch))]) * 4 + 0.5) for _ in range(n_users)])
    budget = rng.lognormal(np.log(10), 0.45, n_users)
    util = W @ ps.T
    util = (util - util.min(1, keepdims=True)) / (util.max(1, keepdims=True) - util.min(1, keepdims=True))
    over = np.clip(base_df["onroad_lakh"].values[None, :] / budget[:, None] - 1, 0, None)
    R = np.clip(1 + 4 * util - 2.5 * over + rng.normal(0, 0.4, util.shape), 1, 5)
    mask = rng.random(R.shape) < 0.25
    item_mean = np.where(mask.sum(0) > 0, (R * mask).sum(0) / np.maximum(mask.sum(0), 1), R[mask].mean())
    user_mean = np.where(mask.sum(1) > 0, (R * mask).sum(1) / np.maximum(mask.sum(1), 1), R[mask].mean())
    filled = np.where(mask, R, item_mean[None, :]) - user_mean[:, None]
    U, S, Vt = np.linalg.svd(filled, full_matrices=False)
    R_hat = (U[:, :rank] * S[:rank]) @ Vt[:rank] + user_mean[:, None]
    return {"W": W, "R_hat": R_hat, "index": base_df.index}


def peer_scores(priorities, car_index, peer, k=25):
    w = np.array([max(priorities[p], 0.01) for p in PRIORITIES])
    w = w / w.sum()
    sim = cosine(w, peer["W"])
    top = np.argsort(-sim)[:k]
    pos = [list(peer["index"]).index(i) for i in car_index]
    return peer["R_hat"][top][:, pos].mean(axis=0)


def build_assets():
    model, metrics, importances, test_pairs = train_resale_model()
    base = compute_costs(default_profile(), model)
    peer = build_peer_model(base)
    return {"model": model, "metrics": metrics, "importances": importances, "test_pairs": test_pairs, "peer": peer}


# ----------------------------------------------------------------------------
# 4. Natural-language understanding (rule based, works offline)
# ----------------------------------------------------------------------------
def parse_query(text, base_profile=None):
    """Parse common buyer phrases offline; base_profile lets follow-up turns retain context."""
    p = validate_profile(base_profile) if base_profile else default_profile()
    t = (text or "").lower().replace("₹", " rs ")

    # Budget in lakh/lac or crore. Supports: 12 lakh, Rs 10L, 1.1 crore.
    money = re.search(r"(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lakhs|lac|lacs|l)\b", t)
    if money:
        amount = float(money.group(1))
        p["budget_lakh"] = amount * (100 if money.group(2) in {"crore", "cr"} else 1)

    # Seating: family size, explicit seat count or a 7-seater request.
    seat_match = (re.search(r"family\s+of\s+(\d+)", t)
                  or re.search(r"(?:need|want|minimum|min)\s+(\d+)\s+seats?", t)
                  or re.search(r"(\d+)\s*[- ]?seater", t))
    if seat_match:
        requested = int(seat_match.group(1))
        p["seats"] = 4 if requested <= 4 else (5 if requested <= 5 else 7)

    # Annual distance; '15k km' and '15,000 km per year' are both accepted.
    km_match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:km|kilomet(?:re|er)s?)\b", t)
    if km_match:
        amount = float(km_match.group(1).replace(",", ""))
        p["km_per_year"] = int(amount * (1000 if km_match.group(2) else 1))

    if any(term in t for term in ["mostly city", "city driving", "in city", "urban", "traffic"]):
        p["usage"] = "City"
    elif any(term in t for term in ["highway", "long distance", "long trips"]):
        p["usage"] = "Highway"
    elif "mixed" in t or "city and highway" in t:
        p["usage"] = "Mixed"

    fuel_patterns = {
        "Petrol": r"\bpetrol\b|\bgasoline\b",
        "Diesel": r"\bdiesel\b",
        "CNG": r"\bcng\b",
        "Hybrid": r"\bhybrid\b",
        "Electric": r"\belectric\b|\bev\b|\belectric vehicle\b",
    }
    requested_fuels = [fuel for fuel, pattern in fuel_patterns.items() if re.search(pattern, t)]
    if requested_fuels:
        p["fuels"] = requested_fuels

    bodies = []
    if re.search(r"\bsuv\b", t):
        bodies += ["Compact SUV", "Midsize SUV", "Large SUV"]
    if re.search(r"\bsedan\b", t): bodies.append("Sedan")
    if re.search(r"\bhatch(?:back)?\b", t): bodies.append("Hatchback")
    if re.search(r"\bmpv\b|\bpeople mover\b", t): bodies.append("MPV")
    if bodies:
        p["bodies"] = list(dict.fromkeys(bodies))

    if any(k in t for k in ["no home charg", "no charging", "cannot charge", "can't charge", "apartment"]):
        p["home_charging"] = False
    elif any(k in t for k in ["home charging", "charge at home", "home charger"]):
        p["home_charging"] = True

    priority_terms = {
        "Safety": ["safe", "safety", "ncap", "crash"],
        "Economy": ["mileage", "economical", "economy", "cheap", "running cost", "fuel efficient", "save money"],
        "Performance": ["power", "sporty", "performance", "fast"],
        "Space": ["space", "boot", "luggage", "family"],
        "Road Presence": ["rough", "ghat", "clearance", "off-road", "pothole"],
        "Resale": ["resale", "resell", "second hand value"],
    }
    for priority, terms in priority_terms.items():
        if any(term in t for term in terms):
            p["priorities"][priority] = max(p["priorities"].get(priority, 5), 9.0)

    return validate_profile(p)


# ----------------------------------------------------------------------------
# 5. The agent
# ----------------------------------------------------------------------------
def explain(row, ps_row, pri, shortlist):
    texts = {"Economy": lambda: f"ownership-period TCO Rs {row.tco/1e5:.1f}L ({(row.tco - shortlist.tco.median())/1e5:+.1f}L vs shortlist median)",
             "Safety": lambda: f"{int(row.ncap)}-star safety rating",
             "Performance": lambda: f"{int(row.bhp)} bhp power",
             "Space": lambda: f"{int(row.boot_l)} L boot, {int(row.seats)} seats",
             "Road Presence": lambda: f"{int(row.clearance)} mm ground clearance",
             "Resale": lambda: f"about {row.resale_pct:.0f}% resale value retained"}
    contrib = {k: pri[k] * ps_row[k] for k in PRIORITIES if ps_row[k] >= 0.6}
    strengths = sorted(contrib, key=contrib.get, reverse=True)[:2]
    out = "Strong on " + " and ".join(texts[s]() for s in strengths) if strengths else "A balanced all-rounder for your profile"
    median_p = np.median(list(pri.values()))
    weak = [k for k in PRIORITIES if pri[k] >= median_p and ps_row[k] < 0.4]
    if weak:
        w = min(weak, key=lambda k: ps_row[k])
        out += f". Trade-off: below average on {w.lower()} ({texts[w]()})"
    return out + "."


def run_agent(profile, assets, liked=(), disliked=(), method_weights=(0.6, 0.2, 0.2),
              entropy_share=0.25, top_n=10, seed=0):
    profile = validate_profile(profile)
    model, peer = assets["model"], assets["peer"]
    trace = []
    pri = dict(profile["priorities"])
    liked, disliked = list(liked), list(disliked)

    trace.append(f"PERCEIVE: budget Rs {profile['budget_lakh']:.1f}L on-road, {profile['seats']}+ seats, "
                 f"{profile['km_per_year']:,} km/yr ({profile['usage']}), {profile['years']}-yr horizon, "
                 f"priorities {', '.join(f'{k} {v:.0f}' for k, v in pri.items())}")

    costs = compute_costs(profile, model)
    trace.append(f"ESTIMATE: computed {profile['years']}-year estimated TCO (depreciation + fuel + service + insurance) "
                 f"for {len(costs)} cars; "
                 f"resale predicted by Random Forest")

    # ---- hard-constraint filter with automatic re-planning
    budget, min_ncap, bodies, fuels = profile["budget_lakh"], profile["min_ncap"], list(profile["bodies"]), list(profile["fuels"])

    def make_mask():
        m = (costs["onroad_lakh"] <= budget) & (costs["seats"] >= profile["seats"]) & (costs["ncap"] >= min_ncap) & costs["fuel"].isin(fuels)
        return m & costs["body"].isin(bodies) if bodies else m

    mask = make_mask()
    trace.append(f"FILTER: {int(mask.sum())} of {len(costs)} cars satisfy all hard constraints")
    relaxed, step = [], 0
    while mask.sum() < 5 and step < 5:
        if step < 2:
            budget *= 1.10
            relaxed.append(f"budget raised to Rs {budget:.1f}L")
        elif step == 2 and min_ncap > 0:
            min_ncap -= 1
            relaxed.append(f"minimum safety lowered to {min_ncap} stars")
        elif step == 3 and bodies:
            bodies = []
            relaxed.append("body-type restriction removed")
        elif step == 4 and len(fuels) < len(FUEL_TYPES):
            fuels = list(FUEL_TYPES)
            relaxed.append("fuel restriction removed")
        step += 1
        mask = make_mask()
        if relaxed and relaxed[-1] and step <= 5:
            trace.append(f"REPLAN: fewer than 5 candidates -> {relaxed[-1]} -> {int(mask.sum())} candidates")
    short = costs[mask].copy()
    if len(short) < 2:
        trace.append("STOP: no feasible cars found even after relaxing constraints")
        return {"ranked": short, "trace": trace, "priorities": pri, "relaxed": relaxed, "costs": costs,
                "weights": None, "n_candidates": len(short)}

    # ---- online learning from feedback
    if liked or disliked:
        ps = priority_scores(short)
        for sign, ids in [(1, liked), (-1, disliked)]:
            for i in ids:
                if i in ps.index:
                    for k in PRIORITIES:
                        pri[k] = float(np.clip(pri[k] + sign * 3.0 * (ps.loc[i, k] - ps[k].mean()), 0.5, 10))
        trace.append("LEARN: feedback updated priorities -> " + ", ".join(f"{k} {v:.1f}" for k, v in pri.items()))

    # ---- scoring
    X = short[CRIT].values.astype(float)
    w_user = criteria_weights(pri)
    w_ent = entropy_weights(X) if len(short) >= 5 else w_user
    w = (1 - entropy_share) * w_user + entropy_share * w_ent
    w = w / w.sum()
    t_score = minmax(topsis(X, w, BENEFIT))
    c_score = minmax(cosine(ideal_vector(profile, pri, liked), CONTENT.loc[short.index].values))
    p_score = minmax(peer_scores(pri, short.index, peer))
    trace.append("SCORE: TOPSIS weights = " + ", ".join(f"{l} {x:.2f}" for l, x in zip(CRIT_LABELS, w)))
    mw = np.asarray(method_weights, dtype=float)
    if mw.shape != (3,) or not np.isfinite(mw).all() or (mw < 0).any() or mw.sum() <= 0:
        raise ValueError("method_weights must contain three non-negative values with a positive sum.")
    mw = mw / mw.sum()
    final = mw[0] * t_score + mw[1] * c_score + mw[2] * p_score
    trace.append(f"DECIDE: hybrid = {mw[0]:.2f} x TOPSIS + {mw[1]:.2f} x content similarity + {mw[2]:.2f} x peer (SVD) score")

    short["topsis"], short["content"], short["peer"], short["score"] = t_score, c_score, p_score, final

    # ---- robustness: perturb the priority weights +-20% and count how often a car stays in the top 3
    rng = np.random.default_rng(seed)
    top3 = np.zeros(len(short))
    for _ in range(40):
        pw = criteria_weights({k: v * rng.uniform(0.8, 1.2) for k, v in pri.items()})
        wp = (1 - entropy_share) * pw + entropy_share * w_ent
        f = mw[0] * minmax(topsis(X, wp / wp.sum(), BENEFIT)) + mw[1] * c_score + mw[2] * p_score
        top3[np.argsort(-f)[:3]] += 1
    short["stability"] = top3 / 40

    short = short.sort_values("score", ascending=False)
    ps = priority_scores(short)
    short["why"] = ""
    for idx in short.index[:top_n]:
        short.at[idx, "why"] = explain(short.loc[idx], ps.loc[idx], pri, short)
    short["rank"] = range(1, len(short) + 1)
    trace.append(f"EXPLAIN: top pick is {short.iloc[0]['name']} (relative shortlist score "
                 f"{short.iloc[0]['score']*100:.0f}/100; stays in top-3 in "
                 f"{short.iloc[0]['stability']*100:.0f}% of weight perturbations)")
    return {"ranked": short, "trace": trace, "priorities": pri, "relaxed": relaxed, "costs": costs,
            "weights": w, "n_candidates": len(short), "score_type": "relative shortlist score"}


if __name__ == "__main__":
    assets = build_assets()
    print("Resale model:", {k: round(v, 3) for k, v in assets["metrics"].items()})
    q = "family of 5, budget 12 lakh, 15000 km per year, mostly city, want a safe SUV with good resale"
    prof = parse_query(q)
    res = run_agent(prof, assets)
    print("\n".join(res["trace"]))
    print(res["ranked"][["name", "fuel", "onroad_lakh", "tco", "score", "stability", "why"]].head(5).to_string())
