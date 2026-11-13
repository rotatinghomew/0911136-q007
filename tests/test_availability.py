import unittest
from datetime import datetime, timezone

from station_health.availability import StationStatus, active_at


class AvailabilityTests(unittest.TestCase):
    def test_interval_end_is_exclusive(self):
        start = datetime(2026, 8, 2, tzinfo=timezone.utc)
        end = datetime(2026, 8, 3, tzinfo=timezone.utc)
        status = StationStatus("s1", start, end, "maintenance")
        self.assertTrue(active_at(status, start))
        self.assertFalse(active_at(status, end))
