"""Geometry and conservation prerequisites; no simulation is hidden in these tests."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/"scripts"),str(Path(__file__).resolve().parents[1]/"examples")]
import make_faulted_co2_kansas as demo
from kansas_branching_geometry import plane_z,plane_dip

class KansasBranchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo.nat.set_flow_threads(8)
        cls.a=SimpleNamespace(years=100,frames=50,multiplier=1e-4,rate=.02)
        cls.b=demo.setup(cls.a);cls.models,cls.meta=demo.models(cls.b,cls.a)
    def test_original_model_and_notch(self):
        b=self.b["bm"];self.assertEqual(b["dims"],(22,22,12));self.assertEqual(b["active"].sum(),4656)
        act=b["active"].reshape(12,22,22);self.assertFalse(act[:,:12,14:].any());self.assertTrue(act[:,:,0].all())
    def test_approved_construction(self):
        g=self.b["geom"];np.testing.assert_allclose(g["B"][:2],[-3250,3887.5])
        self.assertAlmostEqual(plane_dip(g["main"])[0],3.7505387794,places=7)
        self.assertAlmostEqual(plane_dip(g["splay"])[0],4.6664760818,places=7)
        for name,points in [("main",["B","N","STEM","J"]),("splay",["B","S","J"])]:
            for key in points:
                p=g[key];self.assertAlmostEqual(plane_z(g[name],p[0],p[1]),p[2],places=8)
    def test_wells_and_fluid_setup(self):
        p=self.b["params"];self.assertEqual(p["rate_m3_s"],.02);self.assertEqual(p["wells"][1]["bhp"],1.2e7)
        self.assertEqual(p["wells"][0]["cells"],[5+22*(10+22*k) for k in range(8,12)])
        self.assertEqual(p["wells"][1]["cells"],[9+22*(16+22*k) for k in range(1,4)])
        self.assertEqual(p["gravity"],9.80665);self.assertEqual(p["temperature_K"],323.15)
        self.assertEqual(p["p_range_Pa"],[5e6,2e8]);self.assertEqual(p["table_points"],[261,201])
    def test_native_multiplier_and_pore_volume(self):
        self.assertEqual(self.meta["changed_connections"],477)
        np.testing.assert_array_equal(self.models["no_faults"].pore_volume(),self.models["two_faults"].pore_volume())
        self.assertTrue(self.meta["no_throw"]);self.assertEqual(self.meta["applied_multiplier"],1e-4)
    def test_branch_does_not_cross_main(self):
        self.assertGreaterEqual(self.meta["splay_min_depth_above_main_m"],0)
        self.assertEqual(sum(self.meta["face_counts"]["splay"].values()),181)
    def test_smooth_planes_and_section(self):
        surfaces,lines=demo.smooth_surfaces(self.b)
        self.assertEqual(len(surfaces),2)
        for mesh,color in surfaces:self.assertGreater(mesh.n_cells,0);self.assertGreater(mesh.n_points,0)
        self.assertEqual(len(lines),2)
        branch=lines[1][0];self.assertGreater(len(branch),2)
        g=self.b["geom"];xy=g["o"]+branch[:,0,None]*1000*g["u"]
        self.assertTrue(np.all(branch[:,1]<=plane_z(g["main"],xy[:,0],xy[:,1])+1e-8))
    def test_repeat_setup_fingerprint(self):
        b=demo.setup(self.a);self.assertEqual(b["fingerprint"],self.b["fingerprint"])
        self.assertEqual(len(b["times"]),50);np.testing.assert_allclose(np.diff(b["times"]),2)

if __name__=="__main__":unittest.main()
