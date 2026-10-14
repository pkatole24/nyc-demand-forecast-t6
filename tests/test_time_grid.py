import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_trips import clock_grid

class TimeGridTests(unittest.TestCase):
    def test_dst_days_preserve_real_instants(self):
        grid = clock_grid('2025-01-01')
        local = grid.utc_hour.dt.tz_convert('America/New_York')
        spring = local[local.dt.strftime('%Y-%m-%d')=='2024-03-10']
        fall = local[local.dt.strftime('%Y-%m-%d')=='2024-11-03']
        self.assertEqual(len(spring),23)
        self.assertFalse((spring.dt.hour==2).any())
        self.assertEqual(len(fall),25)
        repeated = fall[fall.dt.hour==1]
        self.assertEqual(len(repeated),2)
        self.assertEqual(len(set(t.utcoffset() for t in repeated)),2)
        self.assertTrue(grid.utc_hour.is_unique)
        self.assertEqual(int(grid.utc_hour.diff().dropna().dt.total_seconds().min()),3600)
        self.assertEqual(int(grid.utc_hour.diff().dropna().dt.total_seconds().max()),3600)

if __name__=='__main__':
    unittest.main()
