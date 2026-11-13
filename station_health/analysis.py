"""按设备与共用故障组整理资产和有效参照。"""
from collections import Counter, defaultdict


def asset_profile(tables):
    groups = defaultdict(list)
    assets = {row["station_id"]: row for row in tables["assets"]}
    for row in tables["assets"]:
        if row["cost"] < 0:
            raise ValueError("检修成本不能为负")
        groups[row["failure_group"]].append(row["station_id"])
    valid = Counter()
    for row in tables["truth"]:
        if row["station_id"] not in assets:
            raise ValueError("参照引用未知设备")
        if row["quality"] == "valid" and row["rain_mm"] is not None:
            valid[row["station_id"]] += 1
    return {"failure_groups": dict(sorted(groups.items())), "valid_reference_counts": dict(sorted(valid.items())),
            "repairable_cost": sum(row["cost"] for row in assets.values() if row["repairable"])}


def summarize(tables):
    return {**asset_profile(tables), "plan_count": len(tables["plans"]),
            "scope": "资产和参照覆盖摘要，不估计检修联合收益"}
