"""基于历史观测的检修组合信息价值评估。

设计原则（与 docs/data-contract.md 的边界一致）：

* 只在 ``quality=valid`` 且 ``rain_mm`` 非空的观测上评价；模拟可用性把被
  遮蔽站点的真值从估计器输入中剔除，估计器永远看不到留出真值。
* 检修收益通过同一批有实测的降雨过程、对“不检修基线 vs 检修方案”做配对
  重放得到，方案收益是整体重算的联合量，从不把各站单独收益相加。
* 共用供电故障组的相邻站同进同退：同组两站的单站收益在组级情景里天然
  重叠，联合评估自动避免重复计收，并显式报告单站相加相对联合收益的偏差
  （正值为高估重叠、负值为联合协同）。
* 没有有效参照的区域（如无实测且不可检修的 s7/v3 部分）不参与误差证明，
  只给不依赖降雨假设的覆盖距离指标，并标注为不可验证。
* 输出是供预算会讨论的决策支持材料，不是自动工单。
"""
from dataclasses import dataclass
from itertools import combinations

# 逆距离加权幂参数；主估计取 2，其余用于稳健性敏感性分析。
IDW_POWERS = (1, 2, 3)
PRIMARY_POWER = 2
# 可检修站数不超过该阈值时穷举所有预算内组合，否则退回贪心搜索。
EXHAUSTIVE_LIMIT = 16
# 报告中保留的方案数量与稳健性判定的一致性门槛。
TOP_N = 10
SIGN_CONSISTENCY_FLOOR = 0.6
STABLE_RANK_WINDOW = 3


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _round(value, digits=4):
    if value is None:
        return None
    return round(value, digits)


@dataclass(frozen=True)
class Domain:
    assets: tuple
    stations: tuple
    repairable: frozenset
    costs: dict
    groups: dict
    villages: tuple
    station_village: dict
    village_stations: dict
    coords: dict
    rainfall: dict          # (process_id, station_id) -> rain_mm，仅含有效观测
    processes: tuple
    valid_stations: frozenset
    plan: dict
    historical_outages: tuple
    stress_eligible: tuple  # 没有任何历史停电记录的降雨过程

    @property
    def budget(self):
        return self.plan["budget"]

    @property
    def protected_villages(self):
        return tuple(self.plan["protected_villages"])


def build_domain(tables, plan_id=None):
    """校验资料引用并整理成评估所需的只读领域结构。"""
    assets = tuple(tables["assets"])
    stations = tuple(row["station_id"] for row in assets)
    station_set = frozenset(stations)
    if len(station_set) != len(stations):
        raise ValueError("设备清单存在重复站点")
    for row in assets:
        if row["cost"] < 0:
            raise ValueError("检修成本不能为负")
    repairable = frozenset(row["station_id"] for row in assets if row["repairable"])
    costs = {row["station_id"]: row["cost"] for row in assets}
    groups = {}
    for row in assets:
        groups.setdefault(row["failure_group"], []).append(row["station_id"])
    groups = {name: tuple(sorted(members)) for name, members in sorted(groups.items())}
    villages = tuple(sorted({row["village_id"] for row in assets}))
    station_village = {row["station_id"]: row["village_id"] for row in assets}
    village_stations = {village: [] for village in villages}
    for row in assets:
        village_stations[row["village_id"]].append(row["station_id"])
    village_stations = {village: tuple(sorted(members)) for village, members in village_stations.items()}
    coords = {row["station_id"]: (float(row["x"]), float(row["y"])) for row in assets}
    rainfall = {}
    valid_stations = set()
    seen = set()
    for row in tables["truth"]:
        if row["station_id"] not in station_set:
            raise ValueError("参照引用未知设备")
        key = (row["process_id"], row["station_id"])
        if key in seen:
            raise ValueError("同过程同站点存在重复参照")
        seen.add(key)
        if row["quality"] == "valid" and row["rain_mm"] is not None:
            rainfall[key] = float(row["rain_mm"])
            valid_stations.add(row["station_id"])
    processes = tuple(sorted({key[0] for key in rainfall}, key=lambda p: int(p[1:])))
    plans = tuple(tables["plans"])
    if plan_id is None:
        if len(plans) != 1:
            raise ValueError("存在多个预算方案时必须显式指定 plan_id")
        plan = plans[0]
    else:
        plan = next((row for row in plans if row["plan_id"] == plan_id), None)
        if plan is None:
            raise ValueError("未知预算方案：" + str(plan_id))
    unknown_protected = set(plan["protected_villages"]) - set(villages)
    if unknown_protected:
        raise ValueError("重点村引用未知村落：" + ",".join(sorted(unknown_protected)))
    historical = []
    historical_processes = set()
    for row in tables["outages"]:
        members = groups.get(row["failure_group"])
        if members is None:
            raise ValueError("停电记录引用未知故障组：" + row["failure_group"])
        if sorted(row["unavailable_stations"]) != sorted(members):
            raise ValueError("停电站点与故障组成员不一致：" + row["outage_id"])
        if row["process_id"] not in processes:
            raise ValueError("停电记录落在没有有效实测的过程：" + row["outage_id"])
        historical.append((row["process_id"], row["failure_group"]))
        historical_processes.add(row["process_id"])
    historical = tuple(sorted(historical, key=lambda item: (int(item[0][1:]), item[1])))
    stress_eligible = tuple(p for p in processes if p not in historical_processes)
    return Domain(
        assets=assets, stations=stations, repairable=repairable, costs=costs,
        groups=groups, villages=villages,
        station_village=station_village, village_stations=village_stations, coords=coords,
        rainfall=rainfall, processes=processes, valid_stations=frozenset(valid_stations),
        plan=plan, historical_outages=historical, stress_eligible=stress_eligible)


def build_scenarios(domain):
    """构造代表性情景。

    historical：资料里真实发生过的组级停电，只在当时的降雨过程上重放。
    stress：把每个故障组的组级停电套到没有停电记录、但有完整实测的过程
    上，属于反事实压力情景；其可用性模式来自真实故障组，真值来自真实观测，
    但“该过程会发生该组停电”是反事实假设，需与历史情景分开汇报。
    不构造多组同时停电情景：资料没有联合频率，凭空设定会伪造概率结构。
    """
    scenarios = []
    for process_id, group in domain.historical_outages:
        scenarios.append({"scenario_id": f"hist-{process_id}-{group}", "kind": "historical",
                          "process_id": process_id, "failed_group": group,
                          "unavailable": frozenset(domain.groups[group])})
    for process_id in domain.stress_eligible:
        for group in sorted(domain.groups):
            scenarios.append({"scenario_id": f"stress-{process_id}-{group}", "kind": "stress",
                              "process_id": process_id, "failed_group": group,
                              "unavailable": frozenset(domain.groups[group])})
    return scenarios


def _distance_matrix(coords):
    points = tuple(sorted(coords))
    matrix = {}
    for a in points:
        xa, ya = coords[a]
        for b in points:
            xb, yb = coords[b]
            matrix[(a, b)] = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
    return matrix


def evaluate_plan(domain, scenarios, plan_stations, power=PRIMARY_POWER):
    """在全部情景上重放一个检修方案，返回逐情景逐村与站点级评价记录。

    模拟可用性 = 情景停电集合，减去方案中已检修且可检修的站点；没有有效
    观测的站点（如 s7）任何方案都无法恢复，始终不可用。估计器只接收可用
    站点的有效观测，被遮蔽真值不进入估计。
    """
    plan_stations = frozenset(plan_stations)
    unknown = plan_stations - set(domain.stations)
    if unknown:
        raise ValueError("方案包含未知站点：" + ",".join(sorted(unknown)))
    not_repairable = plan_stations - domain.repairable
    if not_repairable:
        raise ValueError("方案包含不可检修站点：" + ",".join(sorted(not_repairable)))
    # 预算约束属于方案搜索边界（feasible_plans / assess 入口），此处只做模拟
    # 重放；可加性诊断等内部分析需要评估超预算的假设组合。

    matrix = _distance_matrix(domain.coords)
    village_centroids = {
        village: (
            _mean(domain.coords[s][0] for s in members),
            _mean(domain.coords[s][1] for s in members),
        )
        for village, members in domain.village_stations.items()
    }
    village_records = []
    point_records = []
    for scenario in scenarios:
        process_id = scenario["process_id"]
        restored = frozenset(
            station for station in scenario["unavailable"]
            if station in plan_stations and station in domain.valid_stations)
        available = frozenset(
            station for station in domain.stations
            if station not in scenario["unavailable"] or station in restored)
        values = {station: domain.rainfall[(process_id, station)]
                  for station in available if (process_id, station) in domain.rainfall}

        def idw_at(target, xy):
            if target in values:
                return "observed", values[target]
            weighted = []
            for station, value in values.items():
                distance = ((xy[0] - domain.coords[station][0]) ** 2
                            + (xy[1] - domain.coords[station][1]) ** 2) ** 0.5
                if distance == 0:
                    return "observed", value
                weighted.append((value, distance ** (-power)))
            if not weighted:
                return "uncovered", None
            weight_sum = sum(weight for _, weight in weighted)
            return "interpolated", sum(value * weight for value, weight in weighted) / weight_sum

        for village in domain.villages:
            members = domain.village_stations[village]
            truth_values = [domain.rainfall[(process_id, s)] for s in members
                            if (process_id, s) in domain.rainfall]
            source, estimate = idw_at("centroid@" + village, village_centroids[village])
            in_village_available = [s for s in members if s in available and s in domain.valid_stations]
            nearest_all = None
            if values:
                nearest_all = min(
                    min(matrix[(s, a)] for a in values) for s in members)
            record = {
                "scenario_id": scenario["scenario_id"], "kind": scenario["kind"],
                "process_id": process_id, "failed_group": scenario["failed_group"],
                "village_id": village, "estimate_source": source,
                "abs_error": abs(estimate - _mean(truth_values)) if source != "uncovered" and truth_values else None,
                "in_village_available": len(in_village_available),
                "zero_in_village_gauge": len(in_village_available) == 0,
                "nearest_available_distance": nearest_all,
                "verifiable": bool(truth_values),
            }
            village_records.append(record)

        # 站点级留一误差：被遮蔽且有真值的站点，用其余可用站点插值后对照真值。
        for station in sorted(scenario["unavailable"]):
            truth = domain.rainfall.get((process_id, station))
            if truth is None or station in available:
                continue
            others = {s: value for s, value in values.items() if s != station}
            weighted = []
            for other, value in others.items():
                distance = matrix[(station, other)]
                if distance == 0:
                    weighted = [(value, 1.0)]
                    break
                weighted.append((value, distance ** (-power)))
            if not weighted:
                point_records.append({"scenario_id": scenario["scenario_id"], "kind": scenario["kind"],
                                      "station_id": station, "restored": station in restored,
                                      "abs_error": None, "bias": None})
                continue
            weight_sum = sum(weight for _, weight in weighted)
            estimate = sum(value * weight for value, weight in weighted) / weight_sum
            point_records.append({"scenario_id": scenario["scenario_id"], "kind": scenario["kind"],
                                  "station_id": station, "restored": station in restored,
                                  "abs_error": abs(estimate - truth), "bias": estimate - truth})
    return {"village_records": village_records, "point_records": point_records}


def _pooled_mae(records):
    errors = [r["abs_error"] for r in records if r["abs_error"] is not None]
    return _mean(errors)


def aggregate(domain, scenarios, plan_stations, power=PRIMARY_POWER):
    """把逐情景记录汇总成方案级指标。"""
    result = evaluate_plan(domain, scenarios, plan_stations, power)
    records = result["village_records"]
    point_records = result["point_records"]
    by_kind = {}
    for kind in ("historical", "stress"):
        subset = [r for r in records if r["kind"] == kind]
        by_kind[kind] = {
            "mae": _pooled_mae(subset),
            "uncovered_scenarios": sum(1 for r in subset if r["estimate_source"] == "uncovered"),
            "scenario_count": len({r["scenario_id"] for r in subset}),
        }
    by_village = {}
    for village in domain.villages:
        subset = [r for r in records if r["village_id"] == village]
        verifiable = [r for r in subset if r["verifiable"]]
        by_village[village] = {
            "mae": _pooled_mae(verifiable),
            "verifiable": bool(verifiable),
            "zero_gauge_scenarios": sum(1 for r in subset if r["zero_in_village_gauge"]),
            "mean_nearest_available_distance": _mean(
                r["nearest_available_distance"] for r in subset
                if r["nearest_available_distance"] is not None),
        }
    protected = [r for r in records if r["village_id"] in domain.protected_villages and r["verifiable"]]
    worst_village = max(
        (village for village in domain.villages if by_village[village]["mae"] is not None),
        key=lambda village: by_village[village]["mae"], default=None)
    point_errors = [r for r in point_records if r["abs_error"] is not None]
    return {
        "stations": tuple(sorted(plan_stations)),
        "cost": sum(domain.costs[s] for s in plan_stations),
        "overall_mae": _pooled_mae([r for r in records if r["verifiable"]]),
        "by_class": by_kind,
        "by_village": by_village,
        "protected_mae": _pooled_mae(protected),
        "worst_village": (worst_village, by_village[worst_village]["mae"]) if worst_village else (None, None),
        "point_loo_mae": _mean(r["abs_error"] for r in point_errors),
        "point_loo_bias": _mean(r["bias"] for r in point_errors),
        "point_samples": len(point_errors),
        "records": records,
        "point_records": point_records,
    }


def feasible_plans(domain):
    """枚举预算内的可检修站组合（含空方案基线）。"""
    ordered = tuple(sorted(domain.repairable))
    plans = [()]
    for size in range(1, len(ordered) + 1):
        for combo in combinations(ordered, size):
            if sum(domain.costs[s] for s in combo) <= domain.budget:
                plans.append(combo)
    return plans


def greedy_plans(domain, scenarios, power):
    """穷举不可行时的备选：按联合 MAE 改善逐步贪心加站。"""
    chosen = []
    remaining = set(domain.repairable)
    while remaining:
        best = None
        for station in sorted(remaining):
            candidate = tuple(sorted(chosen + [station]))
            if sum(domain.costs[s] for s in candidate) > domain.budget:
                continue
            metrics = aggregate(domain, scenarios, candidate, power)
            key = (metrics["overall_mae"] if metrics["overall_mae"] is not None else float("inf"),
                   metrics["cost"], station)
            if best is None or key < best[0]:
                best = (key, station)
        if best is None:
            break
        chosen.append(best[1])
        remaining.discard(best[1])
    return [tuple(sorted(chosen[:size])) for size in range(len(chosen) + 1)]


def _improvement(base, value):
    if base is None or value is None or base == 0:
        return None
    return base - value


def additivity_overcount(domain, scenarios, power):
    """同组两站单站收益之和相对联合收益的偏差比例。

    对每个故障组分别在“该组停电”的情景上计算；组内站点在完全相同的情景
    里失效，单站收益天然重叠，这正是不能简单相加的部分。比值为正表示相加
    高估（重叠），为负表示联合恢复存在协同、相加反而低估。
    """
    report = {}
    for group, members in sorted(domain.groups.items()):
        repairable_members = tuple(sorted(set(members) & domain.repairable))
        group_scenarios = [s for s in scenarios if s["failed_group"] == group]
        base = aggregate(domain, group_scenarios, (), power)["overall_mae"]
        singles = []
        for station in repairable_members:
            mae = aggregate(domain, group_scenarios, (station,), power)["overall_mae"]
            singles.append(_improvement(base, mae) or 0.0)
        joint_mae = aggregate(domain, group_scenarios, repairable_members, power)["overall_mae"]
        joint_gain = _improvement(base, joint_mae)
        additive = sum(singles)
        report[group] = {
            "repairable_members": repairable_members,
            "single_station_gains": {s: _round(g) for s, g in zip(repairable_members, singles)},
            "joint_gain": _round(joint_gain),
            "additive_sum": _round(additive),
            "additive_vs_joint_ratio": _round(_safe_ratio(additive - (joint_gain or 0.0), joint_gain))
            if joint_gain and joint_gain > 0 else None,
            "ratio_interpretation": "正值=单站相加高估联合收益；负值=联合恢复有协同、相加低估",
            "note": None if repairable_members else "该组没有可检修站点，无法评估检修收益",
        }
    return report


def marginal_contributions(domain, scenarios, plan_stations, power):
    """留一边际贡献：方案内每个站被拿掉后联合 MAE 的变化。"""
    full = aggregate(domain, scenarios, plan_stations, power)["overall_mae"]
    out = {}
    for station in sorted(plan_stations):
        reduced = frozenset(plan_stations) - {station}
        mae = aggregate(domain, scenarios, reduced, power)["overall_mae"]
        out[station] = _round(mae - full)
    return out


def rank_plans(domain, scenarios, power, use_exhaustive):
    if use_exhaustive:
        candidates = feasible_plans(domain)
        search = "exhaustive"
    else:
        candidates = greedy_plans(domain, scenarios, power)
        search = "greedy"
    ranked = []
    for combo in candidates:
        metrics = aggregate(domain, scenarios, combo, power)
        ranked.append(metrics)
    ranked.sort(key=lambda m: (
        m["overall_mae"] if m["overall_mae"] is not None else float("inf"),
        m["worst_village"][1] if m["worst_village"][1] is not None else float("inf"),
        m["cost"], m["stations"]))
    return ranked, search


def confidence_label(domain, baseline, metrics, rank_lookup_by_power):
    if not metrics["stations"]:
        return "baseline", ["不检修基线，无收益可判定"]
    reasons = []
    hist_gain = _improvement(baseline["by_class"]["historical"]["mae"],
                             metrics["by_class"]["historical"]["mae"])
    active = [r for r in metrics["records"]
              if any(s in domain.groups[r["failed_group"]] for s in metrics["stations"])]
    base_by_scenario = {}
    for record in baseline["records"]:
        base_by_scenario.setdefault(record["scenario_id"], []).append(record)
    improved = 0
    for scenario_id in {r["scenario_id"] for r in active}:
        current = _mean(r["abs_error"] for r in active if r["scenario_id"] == scenario_id
                        and r["abs_error"] is not None)
        base = _mean(r["abs_error"] for r in base_by_scenario[scenario_id]
                     if r["abs_error"] is not None)
        if current is None or base is None:
            continue
        if current < base - 1e-12:
            improved += 1
    consistency = _safe_ratio(improved, len({r["scenario_id"] for r in active}))
    station_key = tuple(sorted(metrics["stations"]))
    top_window = all(
        rank_lookup_by_power[p].get(station_key, 10 ** 9) <= STABLE_RANK_WINDOW
        for p in IDW_POWERS)
    fragile = []
    if hist_gain is None or hist_gain <= 1e-12:
        fragile.append("历史真实停电情景下没有可验证的误差改善")
    if consistency is not None and consistency < SIGN_CONSISTENCY_FLOOR:
        fragile.append(f"仅 {consistency:.0%} 的相关情景改善，符号一致性低于门槛")
    if not top_window:
        fragile.append("插值幂参数变化时排名跌出前三，结论依赖估计器设定")
    if fragile:
        return "fragile", fragile
    if (hist_gain and hist_gain > 0 and consistency is not None
            and consistency >= SIGN_CONSISTENCY_FLOOR and top_window):
        reasons.append(f"历史情景 MAE 改善 {_round(hist_gain)} mm")
        reasons.append(f"相关情景改善占比 {consistency:.0%}")
        reasons.append("在三种插值幂参数下均位列前三")
        return "stable", reasons
    return "mixed", ["部分稳健性条件成立，建议会上补充审查"]


def assess(tables, plan_id=None):
    """完整评估入口：构造情景、穷举搜索、敏感性分析与结论分级。"""
    domain = build_domain(tables, plan_id)
    scenarios = build_scenarios(domain)
    use_exhaustive = len(domain.repairable) <= EXHAUSTIVE_LIMIT

    ranked_by_power = {}
    rank_lookup_by_power = {}
    search = None
    for power in IDW_POWERS:
        ranked, search = rank_plans(domain, scenarios, power, use_exhaustive)
        ranked_by_power[power] = ranked
        rank_lookup_by_power[power] = {m["stations"]: index + 1 for index, m in enumerate(ranked)}
    primary = ranked_by_power[PRIMARY_POWER]
    baseline = next(m for m in primary if not m["stations"])
    baseline_worst = baseline["worst_village"][1]

    stability = {
        "idw_powers": list(IDW_POWERS),
        "rank_by_power": {
            str(power): {"|".join(stations): rank for stations, rank in lookup.items()}
            for power, lookup in rank_lookup_by_power.items()},
        "top1_by_power": {str(power): list(ranked_by_power[power][0]["stations"]) for power in IDW_POWERS},
        "top1_identical": len({ranked_by_power[power][0]["stations"] for power in IDW_POWERS}) == 1,
    }
    overcount = additivity_overcount(domain, scenarios, PRIMARY_POWER)

    plan_rows = []
    for index, metrics in enumerate(primary[:TOP_N], start=1):
        overall_gain = _improvement(baseline["overall_mae"], metrics["overall_mae"])
        hist_gain = _improvement(baseline["by_class"]["historical"]["mae"],
                                 metrics["by_class"]["historical"]["mae"])
        stress_gain = _improvement(baseline["by_class"]["stress"]["mae"],
                                   metrics["by_class"]["stress"]["mae"])
        worst_gain = _improvement(baseline_worst, metrics["worst_village"][1])
        label, reasons = confidence_label(domain, baseline, metrics, rank_lookup_by_power)
        plan_rows.append({
            "rank": index,
            "stations": list(metrics["stations"]),
            "cost": metrics["cost"],
            "overall_mae": _round(metrics["overall_mae"]),
            "overall_mae_gain_mm": _round(overall_gain),
            "historical_mae": _round(metrics["by_class"]["historical"]["mae"]),
            "historical_mae_gain_mm": _round(hist_gain),
            "stress_mae": _round(metrics["by_class"]["stress"]["mae"]),
            "stress_mae_gain_mm": _round(stress_gain),
            "protected_villages_mae": _round(metrics["protected_mae"]),
            "worst_village": metrics["worst_village"][0],
            "worst_village_mae": _round(metrics["worst_village"][1]),
            "worst_village_gain_mm": _round(worst_gain),
            "point_loo_mae": _round(metrics["point_loo_mae"]),
            "point_loo_bias_mm": _round(metrics["point_loo_bias"]),
            "ranks_under_idw_powers": {str(p): rank_lookup_by_power[p].get(metrics["stations"])
                                       for p in IDW_POWERS},
            "marginal_contributions_mm": marginal_contributions(
                domain, scenarios, metrics["stations"], PRIMARY_POWER),
            "confidence": label,
            "confidence_reasons": reasons,
        })

    greedy = greedy_plans(domain, scenarios, PRIMARY_POWER) if use_exhaustive else None
    greedy_gap = None
    if greedy is not None:
        greedy_best = aggregate(domain, scenarios, greedy[-1], PRIMARY_POWER)["overall_mae"]
        exhaustive_best = primary[0]["overall_mae"]
        greedy_gap = _round(greedy_best - exhaustive_best)

    unverifiable = []
    for station in domain.stations:
        if station not in domain.valid_stations:
            village = domain.station_village[station]
            unverifiable.append({
                "station_id": station, "village_id": village,
                "repairable": station in domain.repairable,
                "reason": "没有任何有效降雨参照，插值结果无法被证实或证伪",
            })
    for village in domain.villages:
        members = domain.village_stations[village]
        if not any(s in domain.valid_stations for s in members):
            unverifiable.append({"station_id": None, "village_id": village, "repairable": None,
                                 "reason": "全村无有效参照，该村误差指标不成立"})

    return {
        "scope": ("基于历史有效观测的检修组合信息价值评估，仅用于预算会讨论；"
                  "不生成自动工单，不代表对未来停电概率的预测"),
        "plan_id": domain.plan["plan_id"],
        "budget": domain.budget,
        "protected_villages": list(domain.protected_villages),
        "estimator": {
            "method": "inverse_distance_weighting",
            "primary_power": PRIMARY_POWER,
            "sensitivity_powers": list(IDW_POWERS),
            "village_estimate": "村质心处对全部可用站点有效观测做逆距离加权",
            "truth_rule": "仅 quality=valid 且 rain_mm 非空的记录作为参照；被遮蔽真值不进入估计器",
        },
        "scenario_design": {
            "historical_count": sum(1 for s in scenarios if s["kind"] == "historical"),
            "stress_count": sum(1 for s in scenarios if s["kind"] == "stress"),
            "total_count": len(scenarios),
            "processes_used": list(domain.processes),
            "stress_assumption": ("压力情景把真实故障组停电模式套到无停电记录的实测过程上，"
                                  "‘该过程发生该组停电’为反事实假设，收益与历史情景分列"),
            "excluded": "未构造多组同时停电情景：资料不提供联合频率，避免伪造概率结构",
        },
        "search": {
            "method": search,
            "candidate_count": len(feasible_plans(domain)) if use_exhaustive else None,
            "exhaustive_limit": EXHAUSTIVE_LIMIT,
            "greedy_vs_exhaustive_mae_gap_mm": greedy_gap,
        },
        "stability": stability,
        "joint_vs_additive": overcount,
        "unverifiable_regions": unverifiable,
        "baseline": {
            "overall_mae": _round(baseline["overall_mae"]),
            "historical_mae": _round(baseline["by_class"]["historical"]["mae"]),
            "stress_mae": _round(baseline["by_class"]["stress"]["mae"]),
            "protected_mae": _round(baseline["protected_mae"]),
            "worst_village": baseline["worst_village"][0],
            "worst_village_mae": _round(baseline["worst_village"][1]),
            "village_coverage": {
                village: {
                    "mean_nearest_available_distance": _round(
                        baseline["by_village"][village]["mean_nearest_available_distance"]),
                    "zero_gauge_scenarios": baseline["by_village"][village]["zero_gauge_scenarios"],
                } for village in domain.villages},
        },
        "ranked_plans": plan_rows,
        "interpretation_guide": [
            "stable：历史真实停电情景有改善、相关情景符号一致且对插值幂参数不敏感",
            "mixed：部分稳健性条件成立，需要会上补充审查",
            "fragile：依赖反事实压力情景、估计器设定或无法验证区域，不能作为自动下工单依据",
            "joint_vs_additive 展示同组相邻站单站收益之和与联合收益的偏差：正值为相加高估、负值为协同，方案收益始终以联合重放为准",
        ],
    }
