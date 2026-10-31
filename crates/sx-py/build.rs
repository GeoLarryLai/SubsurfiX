// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Inside a conda env, link OpenMP from the env so the extension and numpy load
//! one OpenMP runtime; two runtimes in one process abort.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=CONDA_PREFIX");
    let Ok(prefix) = std::env::var("CONDA_PREFIX") else {
        return;
    };
    let lib = Path::new(&prefix).join("lib");
    if lib.join("libomp.dylib").is_file() {
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
