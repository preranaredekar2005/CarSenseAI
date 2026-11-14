"""
evaluate.py - Evaluation of CarSense AI.

Protocol
--------
* 8 buyer personas. Each persona has MUST-HAVE conditions written by a domain expert
  (e.g. "5-star safety", "boot >= 350 L", "within budget"). These conditions are the ground truth.
* A recommended car's graded relevance = fraction of must-haves it satisfies;
  it is "fully relevant" if it satisfies all of them.
* The agent only sees the persona's profile and priority sliders, never the must-have rules.
* Metrics @5: Precision, graded relevance, NDCG, Hit rate, MRR.
* Compared against 4 baselines and 3 ablations (TOPSIS only, content only, peer only),
  plus the full hybrid and the hybrid after simulated feedback.
"""
import numpy as np
import pandas as pd
from agent import build_assets, run_agent, default_profile, PRIORITIES


def persona(name, budget, seats, km, usage, pri, must, fuels=None, home=True):
    p = default_profile()
    p.update({"budget_lakh": budget, "seats": seats, "km_per_year": km, "usage": usage, "home_charging": home})
    p["priorities"] = dict(zip(PRIORITIES, pri))      # Economy, Safety, Performance, Space, Road Presence, Resale
    if fuels:
        p["fuels"] = fuels
    return {"name": name, "profile": p, "must": must + [("within budget", lambda d, b=budget: d.onroad_lakh <= b)]}


PERSONAS = [
    persona("Safety-first family", 12, 5, 12000, "Mixed", [5, 10, 2, 8, 4, 5],
            [("5-star safety", lambda d: d.ncap >= 5), ("boot >= 350 L", lambda d: d.boot_l >= 350),
             ("running cost <= Rs 8/km", lambda d: d.cost_km <= 8)]),
    persona("High-km city commuter", 9, 5, 22000, "City", [10, 4, 2, 3, 2, 6],
            [("running cost <= Rs 4.5/km", lambda d: d.cost_km <= 4.5), ("safety >= 3 stars", lambda d: d.ncap >= 3),
             ("cheapest 35% by TCO", lambda d: d.tco <= d.tco.quantile(0.35))]),
    persona("Large family (7 seats)", 22, 7, 15000, "Mixed", [5, 8, 3, 10, 5, 4],
            [("7 seats", lambda d: d.seats >= 7), ("safety >= 4 stars", lambda d: d.ncap >= 4),
             ("running cost <= Rs 7.5/km", lambda d: d.cost_km <= 7.5)]),
    persona("Performance enthusiast", 15, 5, 10000, "Highway", [2, 6, 10, 3, 6, 3],
            [("power >= 110 bhp", lambda d: d.bhp >= 110), ("safety >= 4 stars", lambda d: d.ncap >= 4)]),
    persona("Rough-road / ghat driver", 14, 5, 12000, "Mixed", [4, 7, 5, 4, 10, 4],
            [("clearance >= 190 mm", lambda d: d.clearance >= 190), ("safety >= 4 stars", lambda d: d.ncap >= 4),
             ("power >= 100 bhp", lambda d: d.bhp >= 100)]),
    persona("EV-first urban buyer", 18, 5, 15000, "City", [8, 7, 3, 4, 3, 3],
            [("5-star or 4-star safety", lambda d: d.ncap >= 4), ("running cost <= Rs 2/km", lambda d: d.cost_km <= 2),
             ("boot >= 300 L", lambda d: d.boot_l >= 300)], fuels=["Electric", "Hybrid"]),
    persona("Budget first-time buyer", 8, 5, 8000, "Mixed", [8, 6, 2, 4, 3, 5],
            [("safety >= 3 stars", lambda d: d.ncap >= 3), ("running cost <= Rs 6.5/km", lambda d: d.cost_km <= 6.5)]),
    persona("Highway executive (sedan)", 15, 5, 25000, "Highway", [5, 8, 7, 6, 3, 5],
            [("safety >= 5 stars", lambda d: d.ncap >= 5), ("power >= 100 bhp", lambda d: d.bhp >= 100),
             ("boot >= 450 L", lambda d: d.boot_l >= 450)]),
]


def relevance(costs, pers):
    return pd.concat([f(costs).astype(float) for _, f in pers["must"]], axis=1).mean(axis=1)


def metrics_at_k(order, rel, k=5):
    top = list(order)[:k]
    gains = rel.loc[top].values
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = np.sort(rel.values)[::-1][:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    full = [i for i in list(order)[:10] if rel.loc[i] >= 0.999]
    first = next((r + 1 for r, i in enumerate(list(order)[:10]) if rel.loc[i] >= 0.999), None)
    return {"Precision@5": float(np.mean(gains >= 0.999)), "GradedRel@5": float(gains.mean()),
            "NDCG@5": dcg / idcg if idcg > 0 else 0.0, "HitRate@5": float(any(rel.loc[i] >= 0.999 for i in top)),
            "MRR": 1 / first if first else 0.0}


def baseline_orders(costs, prof, seed):
    pool = costs[(costs.seats >= prof["seats"]) & (costs.onroad_lakh <= prof["budget_lakh"]) & costs.fuel.isin(prof["fuels"])]
    rng = np.random.default_rng(seed)
    return {"Random (in budget)": list(rng.permutation(pool.index)),
            "Cheapest first": list(pool.sort_values("ex_showroom").index),
            "Best mileage first": list(pool.sort_values("cost_km").index),
            "Brand popularity": list(pool.sort_values(["brand_score", "ncap"], ascending=False).index)}


def run_evaluation(assets=None):
    assets = assets or build_assets()
    rows, n_relevant = [], {}
    for pers in PERSONAS:
        prof = pers["profile"]
        base = run_agent(prof, assets)
        costs = base["costs"]
        rel = relevance(costs, pers)
        assert (rel >= 0.999).sum() >= 3, f"persona {pers['name']} is infeasible - fix its must-haves"
        n_relevant[pers['name']] = int((rel >= 0.999).sum())

        def add(method, order):
            rows.append({"persona": pers["name"], "method": method, **metrics_at_k(order, rel)})

        # baselines (random averaged over 30 draws)
        rnd = [metrics_at_k(baseline_orders(costs, prof, s)["Random (in budget)"], rel) for s in range(30)]
        rows.append({"persona": pers["name"], "method": "Random (in budget)", **pd.DataFrame(rnd).mean().to_dict()})
        for name, order in baseline_orders(costs, prof, 0).items():
            if name != "Random (in budget)":
                add(name, order)

        # ablations + full model
        for name, mw in [("TOPSIS only", (1, 0, 0)), ("Content-based only", (0, 1, 0)),
                         ("Peer (SVD) only", (0, 0, 1)), ("CarSense hybrid", (0.6, 0.2, 0.2))]:
            res = run_agent(prof, assets, method_weights=mw)
            add(name, res["ranked"].index)

        # simulated feedback: like 2 fully-relevant cars, dislike 2 non-relevant cars from the first result
        first = list(base["ranked"].index[:10])
        liked = [i for i in first if rel.loc[i] >= 0.999][:2]
        disliked = [i for i in first if rel.loc[i] < 0.999][:2]
        fb = run_agent(prof, assets, liked=liked, disliked=disliked)
        add("CarSense hybrid + feedback", fb["ranked"].index)

        rows.append({"persona": pers["name"], "method": "_stability", "Precision@5": base["ranked"]["stability"].iloc[0]})

    df = pd.DataFrame(rows)
    stability = df[df.method == "_stability"][["persona", "Precision@5"]].rename(columns={"Precision@5": "top1_stability"})
    df = df[df.method != "_stability"]
    order = ["Random (in budget)", "Cheapest first", "Best mileage first", "Brand popularity", "TOPSIS only",
             "Content-based only", "Peer (SVD) only", "CarSense hybrid", "CarSense hybrid + feedback"]
    summary = df.groupby("method").mean(numeric_only=True).loc[order].round(3)
    stability["fully_relevant_cars"] = stability["persona"].map(n_relevant)
    return {"per_persona": df, "summary": summary, "stability": stability,
            "resale_metrics": assets["metrics"], "importances": assets["importances"], "test_pairs": assets["test_pairs"]}


def save_charts(ev, folder="results"):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(folder, exist_ok=True)
    ev["summary"].to_csv(f"{folder}/summary_metrics.csv")
    ev["per_persona"].to_csv(f"{folder}/per_persona_metrics.csv", index=False)

    s = ev["summary"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#9aa5b1"] * 4 + ["#7aa7d9"] * 3 + ["#1f6feb", "#0b8a5f"]
    for a, col in zip(ax, ["Precision@5", "NDCG@5"]):
        a.barh(s.index[::-1], s[col][::-1], color=colors[::-1])
        a.set_xlim(0, 1.1)
        a.set_title(col + " (mean of 8 personas)")
        for y, v in enumerate(s[col][::-1]):
            a.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{folder}/method_comparison.png", dpi=160)
    plt.close()

    y, p = ev["test_pairs"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(y, p, s=10, alpha=0.5, color="#1f6feb")
    ax[0].plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=1)
    ax[0].set_xlabel("Actual resale %")
    ax[0].set_ylabel("Predicted resale %")
    ax[0].set_title(f"Resale model on unseen car models (R2 = {ev['resale_metrics']['R2']:.2f})")
    imp = ev["importances"].head(8)[::-1]
    ax[1].barh(imp.index, imp.values, color="#0b8a5f")
    ax[1].set_title("Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(f"{folder}/resale_model.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    ev = run_evaluation()
    pd.set_option("display.width", 200)
    print("\n=== Resale model (held-out car models) ===")
    print({k: round(v, 3) for k, v in ev["resale_metrics"].items()})
    print("\n=== Mean metrics over 8 personas ===")
    print(ev["summary"])
    print("\n=== Precision@5 per persona ===")
    print(ev["per_persona"].pivot(index="persona", columns="method", values="Precision@5").round(2).T)
    print("\n=== Top-1 stability (share of +-20% weight perturbations where #1 stays in top 3) ===")
    print(ev["stability"])
    save_charts(ev)
    print("\nSaved results/summary_metrics.csv, per_persona_metrics.csv, method_comparison.png, resale_model.png")
