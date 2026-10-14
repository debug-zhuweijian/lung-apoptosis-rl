import importlib.util
import math
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/digitize_martin2022_function.py'
spec=importlib.util.spec_from_file_location('martin_function',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class MartinFunctionTests(unittest.TestCase):
    def test_all_single_drug_native_bar_top_anchors(self):
        from PIL import Image
        checked=0
        for panel in module.PANELS.values():
            image=Image.open(module.SOURCE/panel['image']).convert('RGB')
            for xs,tops in zip(panel['xs'],panel['tops']):
                for x,y,(_,components) in zip(xs,tops,panel['candidates']):
                    if len(components)!=1:
                        continue
                    # Native horizontal black bar boundary spans the bar width;
                    # a vertical SD line or a colored filled row cannot pass this.
                    dark=sum(max(image.getpixel((xx,y)))<110 for xx in range(x-3,x+4))
                    self.assertGreaterEqual(dark,5,(panel['image'],x,y))
                    # Interior just below boundary must contain nonwhite bar fill.
                    interior=sum(min(image.getpixel((xx,y+1)))<245 for xx in range(x-2,x+3))
                    self.assertGreaterEqual(interior,3,(panel['image'],x,y))
                    checked+=1
        self.assertEqual(checked,20)

    def test_log_axis_ticks_and_monotonicity(self):
        for panel in module.PANELS.values():
            for y,value in panel['ticks']:
                # Tick raster rounding is covered by the declared 1-pixel axis bound.
                self.assertLess(abs(math.log10(module.value_at(y,panel['ticks'])/value)),.02)
            self.assertGreater(module.value_at(800,panel['ticks']),module.value_at(850,panel['ticks']))

    def test_candidate_blocks_and_no_pseudoreplication(self):
        rows=module.build_rows()
        self.assertEqual(len(rows),40)
        self.assertEqual(len({r['observation_id'] for r in rows}),40)
        self.assertEqual(len({r['decision_block_id'] for r in rows}),8)
        for block in {r['decision_block_id'] for r in rows}:
            members=[r for r in rows if r['decision_block_id']==block]
            self.assertEqual(len(members),6 if members[0]['panel']=='S2F' else 4)
        self.assertEqual(sum(r['candidate_type']=='single' for r in rows),20)
        for r in rows:
            self.assertLess(r['pixel_reading_lower'],r['published_mean_digitized'])
            self.assertLess(r['published_mean_digitized'],r['pixel_reading_upper'])
            self.assertEqual(r['biological_replicates'],'unknown')
            self.assertFalse(r['a549_eligible'])
            self.assertFalse(r['annexin_fraction_eligible'])

    def test_csv_readback_matches_reconstruction(self):
        import csv
        rows=module.build_rows()
        with (module.OUTPUT/'published_means.csv').open(encoding='utf-8',newline='') as f:
            saved=list(csv.DictReader(f))
        self.assertEqual(len(saved),40)
        for expected,actual in zip(rows,saved):
            self.assertEqual(expected['observation_id'],actual['observation_id'])
            self.assertEqual(expected['published_mean_digitized'],float(actual['published_mean_digitized']))

if __name__=='__main__':
    unittest.main()
