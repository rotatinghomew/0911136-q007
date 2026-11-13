"""验证现有领域摘要的业务边界。"""
import copy
import json
import unittest
from pathlib import Path

from station_health.analysis import asset_profile


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.tables = json.loads((Path(__file__).resolve().parents[1]/'fixtures/dataset.json').read_text())['tables']

    def test_unmeasured_station_has_no_reference_count(self):
        self.assertNotIn('s7',asset_profile(self.tables)['valid_reference_counts'])

    def test_unrepairable_asset_is_not_in_cost(self):
        result = asset_profile(self.tables)
        expected = sum(row['cost'] for row in self.tables['assets'] if row['repairable'])
        self.assertEqual(expected,result['repairable_cost'])

    def test_common_failure_group_retains_both_assets(self):
        self.assertEqual(['s0','s1'],asset_profile(self.tables)['failure_groups']['g0'])

    def test_invalid_cost_fails(self):
        tables = copy.deepcopy(self.tables)
        tables['assets'][0]['cost'] = -1
        with self.assertRaises(ValueError):
            asset_profile(tables)
