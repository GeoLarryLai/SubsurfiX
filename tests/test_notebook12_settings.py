"""Static/mocked notebook control checks; never invoke a scientific solve."""
import ast, contextlib, io, json, sys, unittest, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"examples")]
import fluid_cases as fc
import make_faulted_co2_kansas as kansas

class NotebookSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb=json.loads((ROOT/"examples/12_multiphase_fluids.ipynb").read_text())
        cls.code=["".join(c["source"]) for c in cls.nb["cells"] if c["cell_type"]=="code"]
    def test_all_cells_compile(self):
        for text in self.code:compile(text.replace("%matplotlib inline","pass"),"nb12_cell","exec")
    def test_shared_demo_defaults(self):
        cell=next(c for c in self.code if c.startswith("from fluid_cases import (DEMO_RATE"));ns={};exec(cell,ns)
        self.assertEqual(ns["DEMO_RATE_M3_S"],.008);self.assertEqual(ns["DEMO_YEARS"],40)
        self.assertEqual(ns["DEMO_P_RANGE_PA"],fc.DEMO_P_RANGE_PA)
        self.assertEqual(ns["DEMO_TABLE_POINTS"],list(fc.DEMO_TABLE_POINTS))
        self.assertEqual(fc.demo_report_times(40)[-1],40*fc.YR)
        self.assertEqual(fc.demo_report_times(41)[-1],41*fc.YR)
    def test_each_demo_call_uses_settings(self):
        calls=[c for c in self.code if any(token in c for token in ["= nat.run_fim_water_oil(","= nat.run_fim_black_oil(","= nat.run_fim_co2_brine("])]
        self.assertEqual(len(calls),8)
        for text in calls:
            self.assertIn("wells(DEMO_RATE_M3_S,",text)
            if 'runs["CO2-brine"] =' in text:
                self.assertIn("p_range=DEMO_P_RANGE_PA",text);self.assertIn("table_points=DEMO_TABLE_POINTS",text)
    def test_fluid_case_overrides_reach_native_mock(self):
        for name,fn in fc.CASES.items():
            native={"water-oil":"run_fim_water_oil","oil-gas":"run_fim_black_oil"}.get(name,"run_fim_co2_brine")
            with patch.object(fc,"_setup",return_value=({},object(),2,[0.,1.])), patch.object(fc,"_wells",return_value=[]) as wells, patch.object(fc.nat,native,return_value={}) as solve:
                kw=dict(years=12,rate=.003,report_times_s=[0.,7.,11.])
                if name=="CO2-brine":kw.update(p_range=(5e6,4e7),table_points=[48,101])
                fn(**kw)
                self.assertEqual(wells.call_args.args[1],.003);self.assertEqual(solve.call_args.args[-1],[0.,7.,11.])
                if name=="CO2-brine":self.assertEqual(solve.call_args.kwargs["table_points"],[48,101])
    def test_off_gate_does_not_call_runner(self):
        settings=next(c for c in self.code if c.startswith("RUN_FAULTED_COMPARISON ="));gate=next(c for c in self.code if c.startswith("if RUN_FAULTED_COMPARISON:"));ns={"OUT":Path("outputs")};exec(settings,ns)
        self.assertIs(ns["RUN_FAULTED_COMPARISON"],False)
        with patch.object(kansas,"run_comparison",side_effect=AssertionError("must not solve")) as solve, contextlib.redirect_stdout(io.StringIO()):exec(gate,ns)
        solve.assert_not_called()
    def test_comparison_api_forwards_all_controls(self):
        with patch.object(kansas,"_execute",side_effect=lambda a:a):
            a=kansas.run_comparison(cache_dir="outputs/test",rate=.004,p_range=(5e6,2e8),table_points=[261,201],years=100,frames=50,fault_multiplier=1e-4,inj=(5,10),prod=(9,16),inj_layers=range(8,12),prod_layers=range(1,4),prod_bhp=12e6,n_threads=8)
        self.assertEqual(a.rate,.004);self.assertEqual(a.p_range,(5e6,2e8));self.assertEqual(a.table_points,[261,201])
        self.assertEqual(a.inj_layers,[8,9,10,11]);self.assertEqual(a.prod_layers,[1,2,3]);self.assertEqual(a.prod_bhp,12e6);self.assertEqual(a.n_threads,8)
    def test_direct_comparison_passes_current_settings(self):
        text=next(c for c in self.code if "direct = run_cases(" in c)
        for token in ["years=DEMO_YEARS","rate=DEMO_RATE_M3_S","report_times_s=TIMES","p_range=DEMO_P_RANGE_PA","table_points=DEMO_TABLE_POINTS"]:self.assertIn(token,text)

    def _fake_result(self, fallback=0, bhp=20e6):
        return dict(p=np.full((3,2),20e6),saturation=np.zeros((3,2,2)),
            mass_in_place=np.array([[10.,100.],[11.,99.],[12.,98.]]),
            mass_net_in=np.array([[0.,0.],[1.,-1.],[2.,-2.]]),
            bhp=np.array([[bhp,12e6]]*3),n_fallbacks=fallback)
    def test_zero_producer_co2_is_accepted(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(cache_dir=Path(tmp));b=dict(bm=dict(phi=np.ones(2),active=np.ones(2,bool)),
                times=np.array([2.,4.]),wells=[dict(cells=[0]),dict(cells=[1])],fingerprint="unit-test-only",
                params=dict(p_range_Pa=[5e6,2e8],table_points=[261,201]))
            with patch.object(kansas.nat,"run_fim_co2_brine",return_value=self._fake_result()), patch.object(kansas,"timing_setup",return_value={}):
                with contextlib.redirect_stdout(io.StringIO()):out=kansas.solve("two_faults",object(),b,a)
            self.assertEqual(json.loads(str(out["stats"]))["producer_CO2_saturation_final"],0)
    def test_rejected_stats_are_persisted(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            a=SimpleNamespace(cache_dir=Path(tmp));b=dict(bm=dict(phi=np.ones(2),active=np.ones(2,bool)),
                times=np.array([2.,4.]),wells=[dict(cells=[0]),dict(cells=[1])],fingerprint="unit-test-only",
                params=dict(p_range_Pa=[5e6,2e8],table_points=[261,201]))
            with patch.object(kansas.nat,"run_fim_co2_brine",return_value=self._fake_result(1,61e6)), patch.object(kansas,"timing_setup",return_value={}):
                with self.assertRaises(AssertionError),contextlib.redirect_stdout(io.StringIO()):kansas.solve("two_faults",object(),b,a)
            stats=json.loads((Path(tmp)/"two_faults_stats.json").read_text())
            self.assertEqual(stats["n_fallbacks"],1);self.assertEqual(stats["injector_bhp_max_MPa"],61)
            self.assertFalse((Path(tmp)/"two_faults.npz").exists())

    def test_notebook_rotation_controls(self):
        settings=next(c for c in self.code if c.startswith("RUN_FAULTED_COMPARISON ="));ns={"OUT":Path("outputs")};exec(settings,ns)
        self.assertIs(ns["RUN_FAULTED_COMPARISON"],False);self.assertIs(ns["ROTATE"],True);self.assertEqual(ns["ROTATE_DEG"],35);self.assertEqual(ns["VIEW_DIR"],(0.88,-0.60,0.75));self.assertEqual(ns["RATE_M3_S"],.008)
        gate=next(c for c in self.code if c.startswith("if RUN_FAULTED_COMPARISON:"));self.assertIn("rotate=ROTATE, rotate_deg=ROTATE_DEG, view_dir=VIEW_DIR",gate)
        demo=next(c for c in self.code if c.startswith("from fluid_cases import (DEMO_RATE"));dn={};exec(demo,dn)
        self.assertIs(dn["DEMO_ROTATE"],True);self.assertEqual(dn["DEMO_ROTATE_DEG"],40.);self.assertEqual(dn["DEMO_RATE_M3_S"],.008)
        anim=[c for c in self.code if "animate_grid(grid, frames" in c];self.assertEqual(len(anim),2)
        for c in anim:self.assertIn("rotate=DEMO_ROTATE, rotate_deg=DEMO_ROTATE_DEG",c)

if __name__=="__main__":unittest.main()
