"""区分观测缺失和设备离线。"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StationStatus:
    station_id: str
    starts_at: datetime
    ends_at: datetime | None
    reason: str


def active_at(status: StationStatus, at: datetime) -> bool:
    return status.starts_at <= at and (status.ends_at is None or at < status.ends_at)
