# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import ast
import pathlib

from subsurfix import _native


def test_stub_defs_match_native_callables() -> None:
    repo = pathlib.Path(__file__).resolve().parents[2]
    source = (repo / "python" / "subsurfix" / "_native.pyi").read_text()
    tree = ast.parse(source)
    defs = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    callables = {
        name
        for name in dir(_native)
        if not name.startswith("__") and callable(getattr(_native, name))
    }
    assert defs == callables


def test_stub_class_methods_and_parameters_match_native() -> None:
    import inspect

    repo = pathlib.Path(__file__).resolve().parents[2]
    tree = ast.parse((repo / "python" / "subsurfix" / "_native.pyi").read_text())
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        native = getattr(_native, cls.name)
        stub = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
        public = {n for n in vars(native) if not n.startswith("_")}
        assert set(stub) - {"__init__", "__new__"} == public, cls.name
        for name, fn in stub.items():
            obj = getattr(native, name, None)
            sig = getattr(obj, "__text_signature__", None)
            if not sig or name.startswith("__"):
                continue
            params = [p for p in inspect.signature(obj).parameters if p not in ("self", "$self")]
            want = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs if a.arg != "self"]
            assert params == want, f"{cls.name}.{name}: {params} != {want}"
