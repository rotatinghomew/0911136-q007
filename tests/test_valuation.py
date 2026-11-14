"""验证检修组合信息价值评估的业务边界与反作弊不变量。"""
import copy
import json
import unittest
from pathlib import Path

from station_health.datasource import SourceClient
from station_health.service import serve
from station_health.service import load_repository
from station_health.valuation import (
    IDW_POWERS,
    additivity_overcount,
    aggregate,
    assess,
    build_domain,
    build_scenarios,
    evaluate_plan,
    feasible_plans,
    greedy_plans,
    marginal_contributions,
)

ROOT = Path(__file__).resolve().parents[1]


class ValuationTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads((ROOT / "fixtures/dataset.json").read_text())["tables"]
        self.domain = build_domain(self.tables)
        self.scenarios = build_scenarios(self.domain)

    def test_scenarios_separate_historical_and_counterfactual(self):
        kinds = [s["kind"] for s in self.scenarios]
        self.assertEqual(kinds.count("historical"), len(self.domain.historical_outages))
        # 每个无停电实测过程 × 每个故障组各一个压力情景
        self.assertEqual(kinds.count("stress"),
                         len(self.domain.stress_eligible) * len(self.domain.groups))
        for scenario in self.scenarios:
            self.assertEqual(set(scenario["unavailable"]),
                             set(self.domain.groups[scenario["failed_group"]]))

    def test_estimator_cannot_see_masked_truth(self):
        g0 = next(s for s in self.scenarios if s["failed_group"] == "g0")
        baseline = evaluate_plan(self.domain, [g0], ())
        v0 = next(r for r in baseline["village_records"] if r["village_id"] == "v0")
        self.assertEqual(v0["in_village_available"], 0)
        self.assertEqual(v0["estimate_source"], "interpolated")
        repaired = evaluate_plan(self.domain, [g0], ("s0",))
        v0_repaired = next(r for r in repaired["village_records"] if r["village_id"] == "v0")
        self.assertEqual(v0_repaired["in_village_available"], 1)

    def test_unrepairable_station_is_rejected_even_in_diagnostic(self):
        with self.assertRaises(ValueError):
            evaluate_plan(self.domain, self.scenarios, ("s7",))

    def test_unknown_station_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_plan(self.domain, self.scenarios, ("nope",))

    def test_feasible_plans_respect_budget_and_include_baseline(self):
        plans = feasible_plans(self.domain)
        self.assertIn((), plans)
        for plan in plans:
            self.assertLessEqual(
                sum(self.domain.costs[s] for s in plan), self.domain.budget)
        # s2+s3 合计 9，超过预算 8
        self.assertNotIn(("s2", "s3"), plans)

    def test_joint_gain_is_not_sum_of_single_gains(self):
        base = aggregate(self.domain, self.scenarios, ())["overall_mae"]
        s0 = aggregate(self.domain, self.scenarios, ("s0",))["overall_mae"]
        s1 = aggregate(self.domain, self.scenarios, ("s1",))["overall_mae"]
        both = aggregate(self.domain, self.scenarios, ("s0", "s1"))["overall_mae"]
        additive = (base - s0) + (base - s1)
        joint = base - both
        self.assertLess(joint, additive)
        report = additivity_overcount(self.domain, self.scenarios, 2)["g0"]
        self.assertIsNotNone(report["additive_vs_joint_ratio"])
        self.assertGreater(report["additive_vs_joint_ratio"], 0)

    def test_overcount_flags_group_without_repairable_member(self):
        tables = copy.deepcopy(self.tables)
        for row in tables["assets"]:
            if row["station_id"] == "s6":
                row["repairable"] = False
        domain = build_domain(tables)
        report = additivity_overcount(domain, build_scenarios(domain), 2)["g3"]
        self.assertIsNone(report["additive_vs_joint_ratio"])
        self.assertIn("没有可检修站点", report["note"])

    def test_ranked_plans_are_budget_feasible_and_deterministic(self):
        first = assess(self.tables)
        second = assess(copy.deepcopy(self.tables))
        self.assertEqual(
            [(r["stations"], r["cost"]) for r in first["ranked_plans"]],
            [(r["stations"], r["cost"]) for r in second["ranked_plans"]])
        for row in first["ranked_plans"]:
            self.assertLessEqual(row["cost"], first["budget"])

    def test_best_plan_is_stable_across_idw_powers(self):
        report = assess(self.tables)
        self.assertTrue(report["stability"]["top1_identical"])
        self.assertEqual(set(report["stability"]["idw_powers"]), set(IDW_POWERS))
        self.assertEqual(report["ranked_plans"][0]["confidence"], "stable")

    def test_unmeasured_station_is_marked_unverifiable_not_scored(self):
        report = assess(self.tables)
        unverifiable = {item["station_id"] for item in report["unverifiable_regions"]}
        self.assertIn("s7", unverifiable)
        # 基线村级 MAE 只建立在有有效参照的村记录上，s7 不产生真值误差
        for row in report["ranked_plans"]:
            self.assertIsInstance(row["overall_mae"], float)

    def test_marginal_contribution_is_leave_one_out_joint_change(self):
        plan = ("s0", "s4", "s6")
        full = aggregate(self.domain, self.scenarios, plan)["overall_mae"]
        margins = marginal_contributions(self.domain, self.scenarios, plan, 2)
        for station in plan:
            reduced = tuple(s for s in plan if s != station)
            expected = round(
                aggregate(self.domain, self.scenarios, reduced)["overall_mae"] - full, 4)
            self.assertEqual(margins[station], expected)

    def test_outage_unknown_failure_group_is_rejected(self):
        tables = copy.deepcopy(self.tables)
        tables["outages"][0]["failure_group"] = "ghost"
        with self.assertRaises(ValueError):
            build_domain(tables)

    def test_outage_station_mismatch_is_rejected(self):
        tables = copy.deepcopy(self.tables)
        tables["outages"][0]["unavailable_stations"] = ["s0"]
        with self.assertRaises(ValueError):
            build_domain(tables)

    def test_unknown_plan_id_is_rejected(self):
        with self.assertRaises(ValueError):
            assess(self.tables, plan_id="missing")

    def test_duplicate_truth_key_is_rejected(self):
        tables = copy.deepcopy(self.tables)
        tables["truth"].append(copy.deepcopy(tables["truth"][0]))
        with self.assertRaises(ValueError):
            build_domain(tables)

    def test_report_scope_explicitly_forbids_auto_work_orders(self):
        report = assess(self.tables)
        self.assertIn("不生成自动工单", report["scope"])
        self.assertIn("讨论", report["scope"])

    def test_valuation_over_http_matches_direct_assessment(self):
        repository = load_repository(ROOT / "fixtures/dataset.json",
                                     ROOT / "station_health/schema.json")
        with serve(repository) as address:
            snapshot = SourceClient(address).read()
        remote = assess(snapshot.tables)
        local = assess(self.tables)
        self.assertEqual(
            [row["stations"] for row in remote["ranked_plans"]],
            [row["stations"] for row in local["ranked_plans"]])

    def test_greedy_fallback_stays_feasible_and_no_better_than_optimum(self):
        chain = greedy_plans(self.domain, self.scenarios, 2)
        for plan in chain:
            self.assertLessEqual(
                sum(self.domain.costs[s] for s in plan), self.domain.budget)
        exhaustive_best = aggregate(
            self.domain, self.scenarios,
            assess(self.tables)["ranked_plans"][0]["stations"], 2)["overall_mae"]
        greedy_best = aggregate(self.domain, self.scenarios, chain[-1], 2)["overall_mae"]
        self.assertGreaterEqual(greedy_best, exhaustive_best - 1e-12)

    def test_fully_uncovered_domain_degrades_without_scoring_none(self):
        syn = copy.deepcopy(self.tables)
        all_stations = [row["station_id"] for row in syn["assets"]]
        for row in syn["assets"]:
            row["failure_group"] = "g0"
        syn["outages"] = [dict(syn["outages"][0], failure_group="g0",
                               unavailable_stations=list(all_stations))]
        report = assess(syn)
        self.assertIsNone(report["baseline"]["overall_mae"])
        # 无法验证时不能给出稳定结论，更不能假装检修有效
        self.assertTrue(all(row["confidence"] == "fragile"
                            for row in report["ranked_plans"] if row["stations"]))


if __name__ == "__main__":
    unittest.main()
