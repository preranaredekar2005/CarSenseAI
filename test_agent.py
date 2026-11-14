"""Lightweight smoke tests for the CarSense AI core engine. Run: python -m unittest test_agent.py"""
import unittest
from agent import default_profile, parse_query, validate_profile, build_assets, run_agent


class CarSenseTests(unittest.TestCase):
    def test_parse_initial_profile(self):
        p = parse_query("family of 5, budget 12 lakh, 15000 km per year, mostly city, safe SUV")
        self.assertEqual(p["budget_lakh"], 12.0)
        self.assertEqual(p["seats"], 5)
        self.assertEqual(p["km_per_year"], 15000)
        self.assertEqual(p["usage"], "City")
        self.assertIn("Compact SUV", p["bodies"])

    def test_follow_up_preserves_profile(self):
        first = parse_query("family of 5, budget 12 lakh, 15000 km per year, mostly city")
        second = parse_query("prefer diesel", base_profile=first)
        self.assertEqual(second["budget_lakh"], 12.0)
        self.assertEqual(second["km_per_year"], 15000)
        self.assertEqual(second["fuels"], ["Diesel"])

    def test_k_notation_and_crore(self):
        p = parse_query("budget 1.1 crore, 15k km, electric")
        self.assertAlmostEqual(p["budget_lakh"], 110.0)
        self.assertEqual(p["km_per_year"], 15000)
        self.assertEqual(p["fuels"], ["Electric"])

    def test_invalid_profile_rejected(self):
        p = default_profile(); p["budget_lakh"] = -5
        with self.assertRaises(ValueError): validate_profile(p)

    def test_agent_smoke(self):
        assets = build_assets()
        p = parse_query("family of 5, budget 12 lakh, 15000 km per year, mostly city, safe SUV")
        result = run_agent(p, assets)
        self.assertGreaterEqual(len(result["ranked"]), 2)
        self.assertIn("score", result["ranked"].columns)
        self.assertTrue(result["ranked"]["score"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
