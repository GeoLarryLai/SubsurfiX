// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Shared linear-solver plumbing for the flow and heat core: thread count,
//! a cached sparse Cholesky for the symmetric pressure systems, and report
//! snapshot handling (decimation and streaming to disk).
//!
//! The cache keeps the sparsity pattern, the symbolic factorization, the
//! factor storage and the workspace between solves; when the next system has
//! the same pattern (the usual case within a run) only the values are
//! scattered into the stored compressed-column matrix and the numeric
//! factorization is redone. Values are summed in insertion order, exactly as
//! the previous map-based assembly did, so results do not change.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholesky, SymmetricOrdering,
};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Conj, MatMut, Par, Side};

use crate::error::{SxError, SxResult};

/// Systems smaller than this many unknowns are factorized sequentially
/// (threading costs more than it saves there).
pub(super) const PAR_MIN_UNKNOWNS: usize = 20_000;
/// Minimum chunk length for parallel per-cell / per-face loops.
pub(super) const PAR_MIN_LEN: usize = 4096;

static N_THREADS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static CURRENT: Cell<usize> = const { Cell::new(0) };
}

/// Set the default number of threads for the flow and heat solvers;
/// 0 (the default) uses every core, 1 runs sequentially.
pub fn set_num_threads(n: usize) {
    N_THREADS.store(n, Ordering::Relaxed);
}

/// The default thread count set by [`set_num_threads`], with 0 resolved to
/// the number of cores.
pub fn num_threads() -> usize {
    resolve(N_THREADS.load(Ordering::Relaxed))
}

fn resolve(n: usize) -> usize {
    if n == 0 {
        std::thread::available_parallelism().map_or(1, |v| v.get())
    } else {
        n
    }
}

/// Thread count of the running solve.
pub(super) fn threads() -> usize {
    let c = CURRENT.with(|c| c.get());
    if c > 0 {
        c
    } else {
        num_threads()
    }
}

/// faer parallelism for a system with `n_unk` unknowns.
pub(super) fn par_for(n_unk: usize) -> Par {
    let t = threads();
    if t <= 1 || n_unk < PAR_MIN_UNKNOWNS {
        Par::Seq
    } else {
        Par::rayon(t)
    }
}

/// Whether per-cell / per-face loops of length `len` run in parallel.
pub(super) fn par_loops(len: usize) -> bool {
    threads() > 1 && len >= 2 * PAR_MIN_LEN
}

fn pool(n: usize) -> SxResult<Arc<rayon::ThreadPool>> {
    static POOL: Mutex<Option<(usize, Arc<rayon::ThreadPool>)>> = Mutex::new(None);
    let mut g = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((k, p)) = g.as_ref() {
        if *k == n {
            return Ok(p.clone());
        }
    }
    let p = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| SxError::Numerical(format!("cannot start {n} solver threads: {e}")))?,
    );
    *g = Some((n, p.clone()));
    Ok(p)
}

/// Run `f` with `n_threads` solver threads (`None` = the default from
/// [`set_num_threads`]; 0 = every core). Nested calls reuse the outer pool.
///
/// # Errors
/// `Numerical` when the thread pool cannot start; otherwise `f`'s result.
pub fn with_threads<R: Send>(n_threads: Option<usize>, f: impl FnOnce() -> SxResult<R> + Send) -> SxResult<R> {
    let outer = CURRENT.with(|c| c.get());
    let n = match n_threads {
        Some(k) => resolve(k),
        None if outer > 0 => return f(),
        None => num_threads(),
    };
    if n == outer {
        return f();
    }
    if n <= 1 {
        let prev = CURRENT.with(|c| c.replace(1));
        let r = f();
        CURRENT.with(|c| c.set(prev));
        return r;
    }
    let p = pool(n)?;
    p.install(|| {
        let prev = CURRENT.with(|c| c.replace(n));
        let r = f();
        CURRENT.with(|c| c.set(prev));
        r
    })
}

fn numerical(msg: &str) -> SxError {
    SxError::Numerical(format!("pressure solve: {msg}"))
}

/// Lower-triangle entries of a symmetric system, in insertion order.
#[derive(Debug, Default)]
pub(super) struct SpdEntries {
    keys: Vec<(usize, usize)>,
    vals: Vec<f64>,
}

impl SpdEntries {
    /// Clear, keeping the capacity.
    pub(super) fn clear(&mut self) {
        self.keys.clear();
        self.vals.clear();
    }
    /// Add `v` at (r, c); stored in the lower triangle.
    #[inline]
    pub(super) fn add(&mut self, r: usize, c: usize, v: f64) {
        self.keys.push(if r >= c { (r, c) } else { (c, r) });
        self.vals.push(v);
    }
    pub(super) fn all_finite(&self) -> bool {
        self.vals.iter().all(|v| v.is_finite())
    }
}

struct Pattern {
    n_unk: usize,
    keys: Vec<(usize, usize)>,
    slot: Vec<usize>,
    a: SparseColMat<usize, f64>,
    symbolic: SymbolicCholesky<usize>,
    l_values: Vec<f64>,
    mem: MemBuffer,
    par: Par,
}

/// Cached sparse Cholesky for a run of symmetric positive definite solves.
#[derive(Default)]
pub(super) struct SpdCache {
    pat: Option<Pattern>,
    /// Symbolic factorizations done (pattern changes).
    pub(super) n_symbolic: usize,
}

impl SpdCache {
    fn build(&mut self, n_unk: usize, e: &SpdEntries) -> SxResult<()> {
        // Unique (col, row)-sorted keys and the slot of every entry.
        let mut order: Vec<usize> = (0..e.keys.len()).collect();
        order.sort_by_key(|&i| (e.keys[i].1, e.keys[i].0));
        let mut slot = vec![0usize; e.keys.len()];
        let mut col_ptr = vec![0usize; n_unk + 1];
        let mut row_idx: Vec<usize> = Vec::with_capacity(e.keys.len());
        let mut last: Option<(usize, usize)> = None;
        for &i in &order {
            let (r, c) = e.keys[i];
            if r >= n_unk || c >= n_unk {
                return Err(numerical("matrix index out of range"));
            }
            if last != Some((r, c)) {
                row_idx.push(r);
                col_ptr[c + 1] += 1;
                last = Some((r, c));
            }
            slot[i] = row_idx.len() - 1;
        }
        for c in 0..n_unk {
            col_ptr[c + 1] += col_ptr[c];
        }
        let nnz = row_idx.len();
        let sym = SymbolicSparseColMat::new_checked(n_unk, n_unk, col_ptr, None, row_idx);
        let a = SparseColMat::new(sym, vec![0.0; nnz]);
        let symbolic = factorize_symbolic_cholesky(
            a.symbolic(),
            Side::Lower,
            SymmetricOrdering::Amd,
            CholeskySymbolicParams::default(),
        )
        .map_err(|e| numerical(&format!("symbolic Cholesky failed: {e:?}")))?;
        let par = par_for(n_unk);
        let mem = MemBuffer::try_new(StackReq::any_of(&[
            symbolic.factorize_numeric_llt_scratch::<f64>(par, Default::default()),
            symbolic.solve_in_place_scratch::<f64>(1, par),
        ]))
        .map_err(|e| numerical(&format!("cannot allocate workspace: {e:?}")))?;
        let l_values = vec![0.0f64; symbolic.len_val()];
        self.n_symbolic += 1;
        self.pat = Some(Pattern { n_unk, keys: e.keys.clone(), slot, a, symbolic, l_values, mem, par });
        Ok(())
    }

    /// Solve `A x = rhs` with `A` given by `e`; `rhs` is overwritten by `x`.
    pub(super) fn solve(&mut self, n_unk: usize, e: &SpdEntries, rhs: &mut [f64]) -> SxResult<()> {
        if !e.all_finite() || rhs.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "matrix or right-hand side has non-finite entries (check mobility)",
            ));
        }
        let same = self.pat.as_ref().is_some_and(|p| p.n_unk == n_unk && p.keys == e.keys);
        if !same {
            self.build(n_unk, e)?;
        }
        let p = self.pat.as_mut().expect("built above");
        {
            let v = p.a.val_mut();
            v.fill(0.0);
            for (&s, &x) in p.slot.iter().zip(&e.vals) {
                v[s] += x;
            }
        }
        let stack = MemStack::new(&mut p.mem);
        let llt = p
            .symbolic
            .factorize_numeric_llt::<f64>(&mut p.l_values, p.a.as_ref(), Side::Lower, LltRegularization::default(), p.par, stack, Default::default())
            .map_err(|e| numerical(&format!("Cholesky failed (is every region connected to a BHP well or fixed pressure?): {e:?}")))?;
        llt.solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(rhs, n_unk, 1),
            p.par,
            stack,
        );
        if rhs.iter().any(|v| !v.is_finite()) {
            return Err(numerical("solution is not finite"));
        }
        Ok(())
    }
}

/// One report snapshot handed to a [`ReportOptions::sink`].
#[derive(Debug, Clone, Copy)]
pub struct Snapshot<'a> {
    /// Report index (position in the requested report times).
    pub index: usize,
    /// Report time (s).
    pub time: f64,
    /// Named per-cell fields at this time (e.g. `"temperature"`, or
    /// `"sw"` and `"p"`).
    pub fields: &'a [(&'a str, &'a [f64])],
}

/// Callback receiving every kept report snapshot.
pub type SnapshotSink<'a> = &'a mut (dyn FnMut(&Snapshot<'_>) -> SxResult<()> + Send);

/// How per-cell report fields are kept. The default keeps every report in
/// memory (the previous behaviour).
pub struct ReportOptions<'a> {
    /// Keep every `every`-th report (0 and 1 keep all); the last report is
    /// always kept. Scalar series (energies, rates, BHPs) keep every report.
    pub every: usize,
    /// Store the kept fields in the result; `false` leaves the field series
    /// empty (use a sink).
    pub in_memory: bool,
    /// Called with every kept snapshot, e.g. [`NpyDirSink`].
    pub sink: Option<SnapshotSink<'a>>,
}

impl Default for ReportOptions<'_> {
    fn default() -> Self {
        Self { every: 1, in_memory: true, sink: None }
    }
}

impl ReportOptions<'_> {
    /// Whether report `i` of `n` is kept.
    pub fn keeps(&self, i: usize, n: usize) -> bool {
        self.every <= 1 || i.is_multiple_of(self.every) || i + 1 == n
    }
    /// Indices of the kept reports out of `n`.
    pub fn kept(&self, n: usize) -> Vec<usize> {
        (0..n).filter(|&i| self.keeps(i, n)).collect()
    }
    pub(super) fn emit(&mut self, s: &Snapshot<'_>) -> SxResult<()> {
        match self.sink.as_mut() {
            Some(f) => f(s),
            None => Ok(()),
        }
    }
}

/// Writes each snapshot field to `<dir>/<field>_<index:05>.npy`
/// (little-endian float64, NumPy format 1.0) and a `times.csv` index.
pub struct NpyDirSink {
    dir: PathBuf,
    index: Vec<(usize, f64)>,
}

impl NpyDirSink {
    /// Create (if needed) `dir`.
    ///
    /// # Errors
    /// `Io` when the directory cannot be created.
    pub fn new(dir: impl AsRef<Path>) -> SxResult<Self> {
        std::fs::create_dir_all(dir.as_ref()).map_err(|e| io_err(dir.as_ref(), e))?;
        Ok(Self { dir: dir.as_ref().to_path_buf(), index: Vec::new() })
    }

    /// Write one snapshot.
    ///
    /// # Errors
    /// `Io` on write failure.
    pub fn write(&mut self, s: &Snapshot<'_>) -> SxResult<()> {
        for (name, v) in s.fields {
            write_npy(&self.dir.join(format!("{name}_{:05}.npy", s.index)), v)?;
        }
        self.index.push((s.index, s.time));
        let mut csv = String::from("index,time_s\n");
        for (i, t) in &self.index {
            csv.push_str(&format!("{i},{t:e}\n"));
        }
        let p = self.dir.join("times.csv");
        std::fs::write(&p, csv).map_err(|e| io_err(&p, e))?;
        Ok(())
    }
}

/// Write `v` as a 1-D float64 `.npy` file.
///
/// # Errors
/// `Io` on write failure.
pub fn write_npy(path: &Path, v: &[f64]) -> SxResult<()> {
    use std::io::Write;
    let w = |e| io_err(path, e);
    let mut header = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}", v.len());
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(w)?);
    f.write_all(b"\x93NUMPY\x01\x00").map_err(w)?;
    f.write_all(&(header.len() as u16).to_le_bytes()).map_err(w)?;
    f.write_all(header.as_bytes()).map_err(w)?;
    for x in v {
        f.write_all(&x.to_le_bytes()).map_err(w)?;
    }
    f.flush().map_err(w)?;
    Ok(())
}

fn io_err(path: &Path, source: std::io::Error) -> SxError {
    SxError::Io { path: path.display().to_string(), source }
}
