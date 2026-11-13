"""Layered render-path checks, with no VTK window or scientific solve."""
import sys,tempfile,unittest,json,contextlib,io,inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
from subsurfix import viz3d as viz

class LayeredOverlayTests(unittest.TestCase):
    def test_positive_change_does_not_colour_drawdown(self):
        np.testing.assert_allclose(viz._overlay_change([12.,20.,21.],20.,True),[0.,0.,1.])
        np.testing.assert_allclose(viz._overlay_change([12.,20.,21.],20.),[8.,0.,1.])
    def test_layered_path_draws_only_requested_surfaces(self):
        mesh=MagicMock();mesh.cell_data={};mesh.copy.return_value=mesh
        grid=SimpleNamespace(shape=(1,1,1),spacing=(1.,1.,1.),n_cells=1,
            to_pyvista=lambda:mesh,to_display=lambda x:np.asarray(x))
        mesh.scale.return_value=mesh
        surface=MagicMock();scaled=MagicMock();surface.scale.return_value=scaled
        border=MagicMock();surface.extract_feature_edges.return_value=border
        pl=MagicMock();pl.screenshot.return_value=np.zeros((100,200,3),dtype=np.uint8)
        pv=SimpleNamespace(OFF_SCREEN=False,Plotter=MagicMock(return_value=pl))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{"pyvista":pv}):
            viz.animate_layered(grid,[("plain",[[1.]],{}),("faulted",[[1.]],{"fault_surfaces":[(surface,"#C2185B")],"fault_opacity":.65})],
                Path(tmp)/"mock.gif",times=[1.],shape=(1,2),panel_size=(100,100),hold_last=0,axes=False)
        surface.scale.assert_called_once_with([1,1,40.],inplace=False)
        calls=[c for c in pl.add_mesh.call_args_list if c.args[0] is scaled]
        self.assertEqual(len(calls),1);self.assertEqual(calls[0].kwargs["opacity"],.65)
        self.assertEqual(calls[0].kwargs["color"],"#C2185B")
        border.scale.assert_called_once_with([1,1,40.],inplace=False)

    def test_from_npz_never_solves_or_overwrites_solver_cache(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        import make_faulted_co2_kansas as demo
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"combined.npz"
            np.savez_compressed(source,fingerprint="test-only",**{n+"|"+k:v for n in ["no_faults","two_faults"] for k,v in dict(P=np.zeros((1,1)),S=np.zeros((1,1)),stats=np.array(json.dumps(dict(runtime_s=1.)))).items()})
            before=source.read_bytes();(root/"result.json").write_text("accepted solver report")
            a=SimpleNamespace(cache_dir=root,n_threads=8,from_npz=source,validate_only=False,multiplier=1e-4,render_output=False)
            with patch.object(demo,"setup",return_value=dict(fingerprint="test-only",params={})),patch.object(demo,"models",return_value=({"no_faults":object(),"two_faults":object()},{})),patch.object(demo,"solve",side_effect=AssertionError("forbidden")),patch.object(demo.nat,"set_flow_threads"),patch.object(demo.np,"savez_compressed",side_effect=AssertionError("cache write forbidden")),contextlib.redirect_stdout(io.StringIO()):
                demo._execute(a)
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual((root/"result.json").read_text(),"accepted solver report")
            self.assertTrue((root/"visual_refresh.json").exists())

    def test_rotation_step_switch_and_sweep(self):
        self.assertEqual(viz._rotation_step(False,20.,50),0.)
        self.assertAlmostEqual(viz._rotation_step(True,20.,51)*50,20.)
        self.assertEqual(viz._rotation_step(True,20.,1),20.)
    def _mock_layered_azimuths(self,**kw):
        mesh=MagicMock();mesh.cell_data={};mesh.copy.return_value=mesh;mesh.scale.return_value=mesh
        grid=SimpleNamespace(shape=(1,1,1),spacing=(1.,1.,1.),n_cells=1,to_pyvista=lambda:mesh,to_display=lambda x:np.asarray(x))
        pl=MagicMock();pl.camera=SimpleNamespace(azimuth=0.,zoom=lambda z:None);pl.screenshot.return_value=np.zeros((100,100,3),dtype=np.uint8)
        pv=SimpleNamespace(OFF_SCREEN=False,Plotter=MagicMock(return_value=pl))
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{"pyvista":pv}):
            viz.animate_layered(grid,[("p",np.ones((3,1)),{})],Path(tmp)/"m.gif",times=[1.,2.,3.],panel_size=(100,100),hold_last=0,axes=False,**kw)
        return pl.camera.azimuth
    def test_layered_rotation_is_controllable(self):
        self.assertAlmostEqual(self._mock_layered_azimuths(rotate=True,rotate_deg=20.),20.)
        self.assertEqual(self._mock_layered_azimuths(rotate=False,rotate_deg=20.),0.)
        self.assertAlmostEqual(self._mock_layered_azimuths(),20.)
    def test_public_helpers_expose_rotation_defaults(self):
        import inspect
        for fn,deg in [(viz.animate_grid,40.),(viz.animate_grids,30.),(viz.animate_layered,20.)]:
            ps=inspect.signature(fn).parameters;self.assertIs(ps["rotate"].default,True);self.assertEqual(ps["rotate_deg"].default,deg)
    def test_comparison_forwards_rotation(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        import make_faulted_co2_kansas as demo
        with patch.object(demo,"_execute",side_effect=lambda a:a):a=demo.run_comparison(cache_dir="x",rotate=False,rotate_deg=7)
        self.assertIs(a.rotate,False);self.assertEqual(a.rotate_deg,7.)
        with patch.object(demo,"_execute",side_effect=lambda a:a):a=demo.run_comparison(cache_dir="x")
        self.assertIs(a.rotate,True);self.assertEqual(a.rotate_deg,35.);self.assertEqual(a.view_dir,(.88,-.60,.75))
        with patch.object(demo,"_execute",side_effect=lambda a:a):a=demo.run_comparison(cache_dir="x",view_dir=(1,0,.5),rotate_deg=12)
        self.assertEqual(a.view_dir,(1.,0.,.5));self.assertEqual(a.rotate_deg,12.)
    def test_render_passes_camera_kwargs(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
        import make_faulted_co2_kansas as demo
        self.assertEqual(demo.render.__code__.co_names.count("animate_layered")>=1,True)
        src=Path(demo.__file__).read_text()
        self.assertIn('view_dir=tuple(float(x) for x in getattr(a,"view_dir",(.88,-.60,.75)))',src)
        self.assertIn('rotate_deg=float(getattr(a,"rotate_deg",35.))',src)
    def test_package_defaults_unchanged_for_water_figure(self):
        import inspect
        ps=inspect.signature(viz.animate_layered).parameters
        self.assertEqual(ps["view_dir"].default,(1.0,-0.35,0.75));self.assertEqual(ps["rotate_deg"].default,20.0)

    def test_bar_ticks_and_cell_opacity_hooks(self):
        o={"clim":(20.0,47.6),"bar_ticks":[20,25,30,40,47.6]}
        ticks=viz._colorbar_tick_values(o,5)
        self.assertEqual(list(ticks),[20.,25.,30.,40.,47.6])
        even=viz._colorbar_tick_values({"clim":(20.0,47.6)},5)
        self.assertAlmostEqual(even[0],20.0)
        self.assertAlmostEqual(even[-1],47.6)
        self.assertEqual(len(even),5)
        ps=inspect.signature(viz.animate_layered).parameters
        self.assertIn("view_dir",ps)


if __name__=="__main__":unittest.main()
