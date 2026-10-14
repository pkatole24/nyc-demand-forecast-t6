import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import duckdb
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_trips as prep

class TripAggregationTests(unittest.TestCase):
    def test_duplicates_missing_endpoint_and_fall_back(self):
        old_raw,old_interim = prep.RAW,prep.INTERIM
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent, prefix='.tmp-') as tmp:
            prep.RAW,prep.INTERIM = Path(tmp)/'raw',Path(tmp)/'interim'
            prep.INTERIM.mkdir()
            history = prep.INTERIM/'global_ids.duckdb'
            db = duckdb.connect(str(history))
            db.execute('CREATE TABLE seen(ride_id VARCHAR,row_hash UBIGINT)')
            db.execute('CREATE TABLE completed(month VARCHAR PRIMARY KEY,archive_sha256 VARCHAR,quality_json VARCHAR)')
            db.close()
            def ride(rid,start,end,start_id='A',end_id='B'):
                return [rid,'classic_bike',start,end,'Start',start_id,'End',end_id,'40.75','-73.99','40.76','-73.98','member']
            first = ride('repeat','2024-10-31 23:55:00','2024-11-01 00:05:00')
            batches = {
                '202410':[first,first,ride('missing-end','2024-10-01 10:10:00','2024-10-01 10:20:00',end_id='')],
                '202411':[first,ride('fold','2024-11-03 01:50:00','2024-11-03 01:10:00'),
                          ride('invalid-order','2024-11-05 10:00:00','2024-11-05 09:00:00')],
                '202601':[ride('year-boundary','2025-12-31 23:55:00','2026-01-01 00:05:00'),
                          ride('test-period','2026-01-02 07:00:00','2026-01-02 07:10:00')]
            }
            try:
                for month,rows in batches.items():
                    path = prep.RAW/'extracted'/month/'trips.csv'
                    path.parent.mkdir(parents=True)
                    with path.open('w',newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(prep.FIELDS)
                        writer.writerows(rows)
                    (prep.RAW/f'{month}_receipt.json').write_text(json.dumps({'sha256':month,'csv_files':[{'name':'trips.csv'}]}))
                    prep.process(month,history,'2026-02-01',None)
                frames = pd.concat([pd.read_parquet(prep.INTERIM/m/'hourly.parquet') for m in batches])
                normal = frames[~frames.ambiguous_dst]
                self.assertEqual(int(normal.pickups.sum()),4)
                self.assertEqual(int(normal.dropoffs.sum()),3)
                self.assertEqual(int(frames[frames.ambiguous_dst].pickups.sum()),1)
                self.assertEqual(int(frames[frames.ambiguous_dst].dropoffs.sum()),1)
                oct_q = json.loads((prep.INTERIM/'202410/quality.json').read_text())
                nov_q = json.loads((prep.INTERIM/'202411/quality.json').read_text())
                self.assertEqual(oct_q['exact_duplicate_excess'],1)
                self.assertEqual(nov_q['cross_month_duplicate_ids'],1)
                self.assertEqual(nov_q['invalid_trip_order_or_parse'],1)
                # Cross-month arrivals must remain on their actual endpoint hour.
                arrival = normal[(normal.station_id=='B') & (normal.local_hour==pd.Timestamp('2024-11-01'))]
                self.assertEqual(int(arrival.dropoffs.sum()),1)
                jan_stats=pd.read_parquet(prep.INTERIM/'202601/stations.parquet')
                self.assertEqual(int(jan_stats.training_events.sum()),1)
                self.assertEqual(int(jan_stats.training_pickups.fillna(0).sum()),1)
            finally:
                prep.RAW,prep.INTERIM = old_raw,old_interim

if __name__=='__main__':
    unittest.main()
