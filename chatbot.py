"""
chatbot.py - Terminal conversational interface for the CarSense AI decision agent.
Run with: python chatbot.py

This is an offline, rule-based conversational interface. It does not call an LLM API.
It reuses the same recommendation engine, TCO model and feedback-learning logic as app.py.
"""
from agent import build_assets, default_profile, parse_query, run_agent


def print_recommendations(result, ownership_years=5, limit=5):
    ranked = result["ranked"].head(limit)
    if ranked.empty:
        print("\nI could not find a feasible car. Try increasing your budget or relaxing a constraint.")
        return
    print("\nTop recommendations")
    print("-" * 78)
    for _, car in ranked.iterrows():
        print(f"#{int(car['rank'])} {car['name']} | {car['fuel']} | {car['body']} | {int(car['seats'])} seats")
        print(f"   On-road estimate: Rs {car['onroad_lakh']:.2f} lakh | "
              f"{ownership_years}-year TCO: Rs {car['tco']/100000:.2f} lakh")
        print(f"   Running cost: Rs {car['cost_km']:.2f}/km | "
              f"Resale estimate: {car['resale_pct']:.1f}% | "
              f"Relative score: {car['score']*100:.1f}/100")
        print(f"   Why: {car['why']}\n")
    if result.get("relaxed"):
        print("Constraint adjustments made by the agent:")
        for item in result["relaxed"]:
            print(f" - {item}")
    print("Decision trace:")
    for step in result["trace"]:
        print(f" - {step}")
    print("\nNote: prices, specifications, resale and TCO are academic estimates, not live quotations. ")


def main():
    print("=" * 78)
    print("CAR SENSE AI | Conversational Car-Buying Decision Agent")
    print("=" * 78)
    print("I can understand budget, family size, annual kilometres, usage, fuel, body type and priorities.")
    print("Try: family of 5, budget 12 lakh, 15000 km per year, mostly city, safe SUV, good resale")
    print("Follow-ups retain your previous profile, e.g. 'make my budget 15 lakh' or 'prefer diesel'.")
    print("Commands: help | profile | reset | like 1 | dislike 2 | quit")

    print("Preparing the recommendation models. This may take a few seconds...")
    assets = build_assets()
    profile = default_profile()
    liked, disliked, last_ranked = [], [], None

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCarSense AI: Goodbye!")
            break
        if not query:
            continue
        lower = query.lower()
        if lower in {"quit", "exit", "bye"}:
            print("CarSense AI: Goodbye. Drive safe!")
            break
        if lower == "help":
            print("CarSense AI: Describe your needs in normal language. You can also use 'like 1', "
                  "'dislike 2', 'profile', 'reset' or 'quit'.")
            continue
        if lower == "profile":
            print("Current buyer profile:")
            for key, value in profile.items():
                print(f" - {key}: {value}")
            continue
        if lower == "reset":
            profile, liked, disliked = default_profile(), [], []
            print("CarSense AI: Buyer profile and feedback reset.")
            continue
        if lower.startswith(("like ", "dislike ")):
            if last_ranked is None:
                print("CarSense AI: Ask for recommendations before rating a car.")
                continue
            try:
                rank_no = int(lower.split()[1])
                car_index = int(last_ranked.iloc[rank_no - 1].name)
                target = liked if lower.startswith("like ") else disliked
                other = disliked if lower.startswith("like ") else liked
                if car_index not in target: target.append(car_index)
                if car_index in other: other.remove(car_index)
                print(f"CarSense AI: Feedback saved for {last_ranked.iloc[rank_no - 1]['name']}. Re-ranking...")
            except (ValueError, IndexError):
                print("CarSense AI: Use a valid visible rank, such as 'like 1'.")
                continue
        else:
            try:
                profile = parse_query(query, base_profile=profile)
            except ValueError as exc:
                print(f"CarSense AI: {exc}")
                continue

        try:
            result = run_agent(profile, assets, liked=liked, disliked=disliked)
            last_ranked = result["ranked"].head(10)
            print_recommendations(result, profile["years"])
        except Exception as exc:
            print(f"CarSense AI: I could not complete that recommendation: {exc}")
            print("Please try a simpler request or reset the profile.")


if __name__ == "__main__":
    main()
