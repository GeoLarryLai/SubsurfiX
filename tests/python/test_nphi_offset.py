from subsurfix import _native


def _project(tmp_path, extra=""):
    cfg = tmp_path / "project.toml"
    cfg.write_text('[project]\ncrs = "EPSG:32614"\n' + extra)
    return str(cfg)


def test_not_run(tmp_path):
    assert _native.nphi_shift(_project(tmp_path)) == (0.0, "none")


def test_manual_override(tmp_path):
    cfg = _project(tmp_path, "[normalize]\nnphi_shift = 0.02\n")
    assert _native.nphi_shift(cfg) == (0.02, "manual")
