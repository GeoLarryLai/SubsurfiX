// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Linear solvers for the fully implicit (FIM-OBL) Newton system.
//!
//! * [`LinearSolver::Iterative`] (default): restarted GMRES (Saad & Schultz
//!   1986) with a two-stage constrained pressure residual (CPR) preconditioner
//!   (Wallis 1983; Wallis, Kendall & Little 1985). The system is first scaled
//!   by the inverse of its cell diagonal blocks (alternate block
//!   factorization, Bank, Chan, Coughran & Smith 1989), which makes the
//!   pressure rows a quasi-IMPES pressure equation. Stage 1 solves that
//!   cell-pressure system approximately with one V-cycle of
//!   aggregation-based algebraic multigrid (AMG; piecewise-constant
//!   interpolation, Galerkin coarse operators, Gauss–Seidel smoothing;
//!   Notay 2010) and a dense LU on the coarsest level; stage 2 applies an incomplete LU factorization
//!   without fill, ILU(0), to the full residual. If GMRES does not reach its
//!   tolerance, the step falls back to the direct solver and the fallback is
//!   counted and logged.
//! * [`LinearSolver::Direct`]: sparse LU with partial pivoting (faer, COLAMD
//!   column ordering, Davis et al. 2004). The symbolic analysis is reused
//!   while the sparsity pattern is unchanged. Exact up to round-off.
//!
//! References:
//! * Saad, Y., Schultz, M. H. (1986). GMRES: a generalized minimal residual
//!   algorithm for solving nonsymmetric linear systems. SIAM J. Sci. Stat.
//!   Comput. 7(3), 856–869.
//! * Wallis, J. R. (1983). Incomplete Gaussian elimination as a
//!   preconditioning for generalized conjugate gradient acceleration. SPE
//!   12265.
//! * Wallis, J. R., Kendall, R. P., Little, T. E. (1985). Constrained residual
//!   acceleration of conjugate residual methods. SPE 13536.
//! * Bank, R. E., Chan, T. F., Coughran, W. M., Smith, R. K. (1989). The
//!   alternate-block-factorization procedure for systems of partial
//!   differential equations. BIT 29, 938–954.
//! * Notay, Y. (2010). An aggregation-based algebraic multigrid method.
//!   Electron. Trans. Numer. Anal. 37, 123–146.
//! * Davis, T. A., Gilbert, J. R., Larimore, S. I., Ng, E. G. (2004). A column
//!   approximate minimum degree ordering algorithm. ACM Trans. Math. Softw.
//!   30(3), 353–376.

#![allow(clippy::needless_range_loop)]

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::lu as lu_sp;
use faer::sparse::SparseColMat;
use faer::{Conj, Mat, Par};

/// Linear solver of the FIM-OBL Newton system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinearSolver {
    /// CPR-preconditioned GMRES (AMG pressure stage + ILU(0)), with fallback
    /// to the direct solver when it does not converge.
    Iterative {
        /// Relative tolerance on the scaled linear residual.
        rtol: f64,
        /// Largest number of GMRES iterations (all restarts).
        max_iter: usize,
    },
    /// Sparse direct LU (exact up to round-off).
    Direct,
}

impl Default for LinearSolver {
    fn default() -> Self {
        LinearSolver::Iterative { rtol: 1e-8, max_iter: 300 }
    }
}

impl LinearSolver {
    /// Parse `"iterative"` or `"direct"`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "iterative" | "gmres" | "cpr" => Some(Self::default()),
            "direct" | "lu" => Some(Self::Direct),
            _ => None,
        }
    }
}

/// Counters of the linear solves of one run.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LinearStats {
    /// Linear systems solved.
    pub n_solves: usize,
    /// GMRES iterations (iterative solver).
    pub n_iterations: usize,
    /// Solves that fell back from GMRES to the direct LU.
    pub n_fallbacks: usize,
}

/// Sparse direct LU with a cached symbolic analysis.
pub struct DirectLu {
    pattern: Option<(Vec<usize>, Vec<usize>)>,
    sym: Option<lu_sp::SymbolicLu<usize>>,
    num: lu_sp::NumericLu<usize, f64>,
    /// Number of symbolic analyses done (pattern changes).
    pub n_symbolic: usize,
}

impl Default for DirectLu {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectLu {
    /// New solver.
    pub fn new() -> Self {
        Self { pattern: None, sym: None, num: lu_sp::NumericLu::new(), n_symbolic: 0 }
    }

    /// Factor `a` and solve `a x = b` in place (`b` becomes `x`).
    pub fn solve(&mut self, a: &SparseColMat<usize, f64>, b: &mut Mat<f64>, par: Par) -> Result<(), String> {
        let s = a.symbolic();
        let same = self.pattern.as_ref().is_some_and(|(cp, ri)| cp.as_slice() == s.col_ptr() && ri.as_slice() == s.row_idx());
        if !same || self.sym.is_none() {
            self.n_symbolic += 1;
            self.sym = Some(lu_sp::factorize_symbolic_lu(s, Default::default()).map_err(|e| format!("symbolic lu: {e:?}"))?);
            self.pattern = Some((s.col_ptr().to_vec(), s.row_idx().to_vec()));
        }
        let sy = self.sym.as_ref().expect("symbolic LU set above");
        let req = sy.factorize_numeric_lu_scratch::<f64>(par, Default::default()).or(sy.solve_in_place_scratch::<f64>(1, par));
        let mut mem = MemBuffer::try_new(req).map_err(|e| format!("lu scratch: {e:?}"))?;
        let stack = MemStack::new(&mut mem);
        let lu = sy.factorize_numeric_lu::<f64>(&mut self.num, a.as_ref(), par, stack, Default::default()).map_err(|e| format!("lu: {e:?}"))?;
        lu.solve_in_place_with_conj(Conj::No, b.as_mut(), par, stack);
        Ok(())
    }
}

/// The FIM linear solver: iterative (with direct fallback) or direct.
pub struct FimLinear {
    kind: LinearSolver,
    direct: DirectLu,
    /// Unknowns per cell (pressure first).
    block: usize,
    /// Number of cells; unknowns past `n_cells · block` are well BHPs.
    n_cells: usize,
    /// Counters.
    pub stats: LinearStats,
}

impl FimLinear {
    /// New solver for a system with `n_cells` cell blocks of `block` unknowns
    /// (pressure first) followed by scalar well unknowns.
    pub fn new(kind: LinearSolver, block: usize, n_cells: usize) -> Self {
        Self { kind, direct: DirectLu::new(), block, n_cells, stats: LinearStats::default() }
    }

    /// Symbolic analyses of the direct LU (pattern changes).
    pub fn n_symbolic(&self) -> usize {
        self.direct.n_symbolic
    }

    /// Solve `a x = b` in place.
    pub fn solve(&mut self, a: &SparseColMat<usize, f64>, b: &mut Mat<f64>, par: Par) -> Result<(), String> {
        self.stats.n_solves += 1;
        match self.kind {
            LinearSolver::Direct => self.direct.solve(a, b, par),
            LinearSolver::Iterative { rtol, max_iter } => {
                let rhs: Vec<f64> = (0..b.nrows()).map(|i| b[(i, 0)]).collect();
                match cpr_gmres(a, &rhs, self.block, self.n_cells, rtol, max_iter) {
                    Ok((x, it)) => {
                        self.stats.n_iterations += it;
                        for (i, v) in x.into_iter().enumerate() {
                            b[(i, 0)] = v;
                        }
                        Ok(())
                    }
                    Err((why, it)) => {
                        self.stats.n_iterations += it;
                        self.stats.n_fallbacks += 1;
                        tracing::warn!("fim: GMRES-CPR {why}; falling back to the direct LU for this Newton step");
                        if std::env::var_os("SX_FIM_DEBUG").is_some() {
                            eprintln!("fim: GMRES-CPR {why}; direct LU fallback");
                        }
                        self.direct.solve(a, b, par)
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- CSR utils

/// Compressed sparse row matrix (sorted column indices).
#[derive(Debug, Clone)]
struct Csr {
    n: usize,
    ptr: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<f64>,
}

impl Csr {
    fn from_csc(a: &SparseColMat<usize, f64>) -> Result<Self, String> {
        let t = a.as_ref().transpose().to_col_major().map_err(|e| format!("transpose: {e:?}"))?;
        let s = t.symbolic();
        Ok(Csr { n: a.nrows(), ptr: s.col_ptr().to_vec(), idx: s.row_idx().to_vec(), val: t.val().to_vec() })
    }

    fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.ptr[i], self.ptr[i + 1]);
        (&self.idx[a..b], &self.val[a..b])
    }

    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            let (c, v) = self.row(i);
            *yi = c.iter().zip(v).map(|(&j, &a)| a * x[j]).sum();
        }
    }

    /// Build from per-row (col, value) lists, summing duplicates.
    fn from_rows(rows: Vec<Vec<(usize, f64)>>) -> Self {
        let mut ptr = Vec::with_capacity(rows.len() + 1);
        let mut idx = Vec::new();
        let mut val = Vec::new();
        ptr.push(0);
        for mut r in rows.into_iter() {
            r.sort_unstable_by_key(|e| e.0);
            let mut last = usize::MAX;
            for (c, v) in r {
                if c == last {
                    *val.last_mut().expect("entry pushed") += v;
                } else {
                    idx.push(c);
                    val.push(v);
                    last = c;
                }
            }
            ptr.push(idx.len());
        }
        Csr { n: ptr.len() - 1, ptr, idx, val }
    }
}

/// Sparse product `a · b` (`a` is m×k, `b` is k×n with `b_cols` columns).
fn spgemm(a: &Csr, b: &Csr, b_cols: usize) -> Csr {
    let mut acc = vec![0.0f64; b_cols];
    let mut mark = vec![usize::MAX; b_cols];
    let mut ptr = vec![0usize];
    let mut idx = Vec::new();
    let mut val = Vec::new();
    for i in 0..a.n {
        let start = idx.len();
        let (ac, av) = a.row(i);
        for (&k, &x) in ac.iter().zip(av) {
            let (bc, bv) = b.row(k);
            for (&j, &y) in bc.iter().zip(bv) {
                if mark[j] != i {
                    mark[j] = i;
                    acc[j] = 0.0;
                    idx.push(j);
                }
                acc[j] += x * y;
            }
        }
        idx[start..].sort_unstable();
        for &j in &idx[start..] {
            val.push(acc[j]);
        }
        ptr.push(idx.len());
    }
    Csr { n: a.n, ptr, idx, val }
}

fn transpose(a: &Csr, n_cols: usize) -> Csr {
    let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_cols];
    for i in 0..a.n {
        let (c, v) = a.row(i);
        for (&j, &x) in c.iter().zip(v) {
            rows[j].push((i, x));
        }
    }
    Csr::from_rows(rows)
}

// ------------------------------------------------------------ dense helpers

/// In-place inverse of a small dense row-major `k×k` matrix (Gauss–Jordan,
/// partial pivoting). `None` if singular.
fn invert_small(m: &mut [f64], k: usize) -> Option<()> {
    let mut inv = vec![0.0; k * k];
    for i in 0..k {
        inv[i * k + i] = 1.0;
    }
    for c in 0..k {
        let piv = (c..k).max_by(|&a, &b| m[a * k + c].abs().total_cmp(&m[b * k + c].abs()))?;
        let pv = m[piv * k + c];
        if pv.abs() < 1e-300 || !pv.is_finite() {
            return None;
        }
        if piv != c {
            for j in 0..k {
                m.swap(piv * k + j, c * k + j);
                inv.swap(piv * k + j, c * k + j);
            }
        }
        let d = 1.0 / m[c * k + c];
        for j in 0..k {
            m[c * k + j] *= d;
            inv[c * k + j] *= d;
        }
        for r in 0..k {
            if r != c {
                let f = m[r * k + c];
                if f != 0.0 {
                    for j in 0..k {
                        m[r * k + j] -= f * m[c * k + j];
                        inv[r * k + j] -= f * inv[c * k + j];
                    }
                }
            }
        }
    }
    m.copy_from_slice(&inv);
    Some(())
}

/// Dense LU with partial pivoting for the coarsest AMG level.
struct DenseLu {
    n: usize,
    lu: Vec<f64>,
    piv: Vec<usize>,
}

impl DenseLu {
    fn new(a: &Csr) -> Option<Self> {
        let n = a.n;
        let mut lu = vec![0.0; n * n];
        for i in 0..n {
            let (c, v) = a.row(i);
            for (&j, &x) in c.iter().zip(v) {
                lu[i * n + j] += x;
            }
        }
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let p = (k..n).max_by(|&a, &b| lu[a * n + k].abs().total_cmp(&lu[b * n + k].abs()))?;
            if lu[p * n + k].abs() < 1e-300 {
                return None;
            }
            if p != k {
                for j in 0..n {
                    lu.swap(p * n + j, k * n + j);
                }
                piv.swap(p, k);
            }
            let d = lu[k * n + k];
            for i in k + 1..n {
                let f = lu[i * n + k] / d;
                lu[i * n + k] = f;
                if f != 0.0 {
                    for j in k + 1..n {
                        lu[i * n + j] -= f * lu[k * n + j];
                    }
                }
            }
        }
        Some(DenseLu { n, lu, piv })
    }

    fn solve(&self, b: &[f64], x: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            x[i] = b[self.piv[i]];
        }
        for i in 0..n {
            let s: f64 = (0..i).map(|j| self.lu[i * n + j] * x[j]).sum();
            x[i] -= s;
        }
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|j| self.lu[i * n + j] * x[j]).sum();
            x[i] = (x[i] - s) / self.lu[i * n + i];
        }
    }
}

// --------------------------------------------------------------------- AMG

struct Level {
    a: Csr,
    p: Csr,
    r: Csr,
    nc: usize,
}

/// Aggregation AMG hierarchy (V-cycle, symmetric Gauss–Seidel).
struct Amg {
    levels: Vec<Level>,
    coarse: DenseLu,
}

const AMG_COARSE: usize = 400;
const AMG_THETA: f64 = 0.08;

fn aggregate(a: &Csr) -> (Vec<usize>, usize) {
    let n = a.n;
    let diag: Vec<f64> = (0..n).map(|i| { let (c, v) = a.row(i); c.iter().position(|&j| j == i).map_or(0.0, |k| v[k]) }).collect();
    // Symmetric strength: |a_ij| ≥ θ sqrt(|a_ii a_jj|).
    let theta = AMG_THETA;
    let strong = |i: usize, j: usize, x: f64| i != j && x.abs() >= theta * (diag[i] * diag[j]).abs().sqrt();
    let mut agg = vec![usize::MAX; n];
    let mut nc = 0usize;
    for i in 0..n {
        if agg[i] != usize::MAX {
            continue;
        }
        let (c, v) = a.row(i);
        if c.iter().zip(v).any(|(&j, &x)| strong(i, j, x) && agg[j] != usize::MAX) {
            continue;
        }
        agg[i] = nc;
        for (&j, &x) in c.iter().zip(v) {
            if strong(i, j, x) {
                agg[j] = nc;
            }
        }
        nc += 1;
    }
    for i in 0..n {
        if agg[i] != usize::MAX {
            continue;
        }
        let (c, v) = a.row(i);
        let best = c.iter().zip(v).filter(|(&j, _)| j != i && agg[j] != usize::MAX).max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()));
        match best {
            Some((&j, _)) => agg[i] = agg[j],
            None => {
                agg[i] = nc;
                nc += 1;
            }
        }
    }
    (agg, nc)
}

impl Amg {
    fn new(a: Csr) -> Option<Self> {
        let mut levels = Vec::new();
        let mut cur = a;
        while cur.n > AMG_COARSE && levels.len() < 20 {
            let (agg, nc) = aggregate(&cur);
            if nc as f64 > 0.8 * cur.n as f64 {
                break;
            }
            // Piecewise-constant interpolation from the aggregates;
            // Galerkin coarse operator Aᶜ = Pᵀ A P.
            let p = Csr::from_rows(agg.iter().map(|&g| vec![(g, 1.0)]).collect());
            let r = transpose(&p, nc);
            let ac = spgemm(&r, &spgemm(&cur, &p, nc), nc);
            levels.push(Level { a: cur, p, r, nc });
            cur = ac;
        }
        let coarse = DenseLu::new(&cur)?;
        Some(Amg { levels, coarse })
    }

    fn gs(a: &Csr, b: &[f64], x: &mut [f64], forward: bool) {
        let mut sweep = |i: usize| {
            let (c, v) = a.row(i);
            let mut s = b[i];
            let mut d = 0.0;
            for (&j, &y) in c.iter().zip(v) {
                if j == i { d = y } else { s -= y * x[j] }
            }
            if d != 0.0 {
                x[i] = s / d;
            }
        };
        if forward { (0..a.n).for_each(&mut sweep) } else { (0..a.n).rev().for_each(&mut sweep) }
    }

    fn vcycle(&self, lvl: usize, b: &[f64], x: &mut [f64]) {
        if lvl == self.levels.len() {
            self.coarse.solve(b, x);
            return;
        }
        let l = &self.levels[lvl];
        x.iter_mut().for_each(|v| *v = 0.0);
        Self::gs(&l.a, b, x, true);
        let mut res = vec![0.0; l.a.n];
        l.a.matvec(x, &mut res);
        for i in 0..l.a.n {
            res[i] = b[i] - res[i];
        }
        let mut bc = vec![0.0; l.nc];
        l.r.matvec(&res, &mut bc);
        let mut xc = vec![0.0; l.nc];
        self.vcycle(lvl + 1, &bc, &mut xc);
        let mut corr = vec![0.0; l.a.n];
        l.p.matvec(&xc, &mut corr);
        for i in 0..l.a.n {
            x[i] += corr[i];
        }
        Self::gs(&l.a, b, x, false);
    }
}

// ------------------------------------------------------------------ ILU(0)

struct Ilu0 {
    a: Csr,
    diag: Vec<usize>,
}

impl Ilu0 {
    fn new(mut a: Csr) -> Option<Self> {
        let n = a.n;
        let mut diag = vec![usize::MAX; n];
        for i in 0..n {
            let (c, _) = a.row(i);
            diag[i] = a.ptr[i] + c.iter().position(|&j| j == i)?;
        }
        let mut pos = vec![usize::MAX; n];
        for i in 0..n {
            let (s, e) = (a.ptr[i], a.ptr[i + 1]);
            for k in s..e {
                pos[a.idx[k]] = k;
            }
            for k in s..e {
                let j = a.idx[k];
                if j >= i {
                    break;
                }
                let piv = a.val[diag[j]];
                if piv == 0.0 {
                    return None;
                }
                let f = a.val[k] / piv;
                a.val[k] = f;
                for kk in diag[j] + 1..a.ptr[j + 1] {
                    let p = pos[a.idx[kk]];
                    if p != usize::MAX {
                        a.val[p] -= f * a.val[kk];
                    }
                }
            }
            for k in s..e {
                pos[a.idx[k]] = usize::MAX;
            }
            if a.val[diag[i]] == 0.0 || !a.val[diag[i]].is_finite() {
                return None;
            }
        }
        Some(Ilu0 { a, diag })
    }

    fn solve(&self, b: &[f64], x: &mut [f64]) {
        let a = &self.a;
        for i in 0..a.n {
            let mut s = b[i];
            for k in a.ptr[i]..self.diag[i] {
                s -= a.val[k] * x[a.idx[k]];
            }
            x[i] = s;
        }
        for i in (0..a.n).rev() {
            let mut s = x[i];
            for k in self.diag[i] + 1..a.ptr[i + 1] {
                s -= a.val[k] * x[a.idx[k]];
            }
            x[i] = s / a.val[self.diag[i]];
        }
    }
}

// --------------------------------------------------------------- CPR-GMRES

struct Cpr {
    amg: Amg,
    ilu: Ilu0,
    /// Pressure unknown of each pressure-system row, in the full system.
    p_dof: Vec<usize>,
    /// Columns of the scaled matrix at the pressure unknowns (row order kept),
    /// indexed by pressure-system column: `A · y1` with `y1` nonzero only there.
    a_pcols: Csr,
}

impl Cpr {
    fn apply(&self, a: &Csr, r: &[f64], y: &mut [f64]) {
        let np = self.p_dof.len();
        let rp: Vec<f64> = self.p_dof.iter().map(|&k| r[k]).collect();
        let mut xp = vec![0.0; np];
        self.amg.vcycle(0, &rp, &mut xp);
        let mut y1 = vec![0.0; a.n];
        for (q, &k) in self.p_dof.iter().enumerate() {
            y1[k] = xp[q];
        }
        let mut r2 = vec![0.0; a.n];
        self.a_pcols.matvec(&xp, &mut r2);
        for i in 0..a.n {
            r2[i] = r[i] - r2[i];
        }
        self.ilu.solve(&r2, y);
        for i in 0..a.n {
            y[i] += y1[i];
        }
    }
}

type Failure = (String, usize);

/// Left-scale by the inverse cell diagonal blocks, then CPR-GMRES.
fn cpr_gmres(a_csc: &SparseColMat<usize, f64>, b: &[f64], nb: usize, n_cells: usize, rtol: f64, max_iter: usize) -> Result<(Vec<f64>, usize), Failure> {
    let a = Csr::from_csc(a_csc).map_err(|e| (e, 0))?;
    let n = a.n;
    let nblk = n_cells * nb;
    // Block-diagonal inverses.
    let mut dinv = vec![0.0; n_cells * nb * nb];
    for c in 0..n_cells {
        let blk = &mut dinv[c * nb * nb..(c + 1) * nb * nb];
        for r in 0..nb {
            let (cols, vals) = a.row(c * nb + r);
            for (&j, &x) in cols.iter().zip(vals) {
                if j >= c * nb && j < (c + 1) * nb {
                    blk[r * nb + (j - c * nb)] = x;
                }
            }
        }
        if invert_small(blk, nb).is_none() {
            return Err((format!("singular diagonal block at cell {c}"), 0));
        }
    }
    // Scaled matrix and right-hand side.
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    let mut bs = vec![0.0; n];
    for c in 0..n_cells {
        let blk = &dinv[c * nb * nb..(c + 1) * nb * nb];
        for r in 0..nb {
            let mut row = Vec::new();
            let mut s = 0.0;
            for k in 0..nb {
                let w = blk[r * nb + k];
                if w == 0.0 {
                    continue;
                }
                s += w * b[c * nb + k];
                let (cols, vals) = a.row(c * nb + k);
                row.extend(cols.iter().zip(vals).map(|(&j, &x)| (j, w * x)));
            }
            // Keep the diagonal in the pattern for ILU(0).
            row.push((c * nb + r, 0.0));
            rows.push(row);
            bs[c * nb + r] = s;
        }
    }
    for i in nblk..n {
        let (cols, vals) = a.row(i);
        let d = cols.iter().position(|&j| j == i).map(|k| vals[k]).filter(|d| *d != 0.0 && d.is_finite()).ok_or_else(|| (format!("zero diagonal in well row {}", i - nblk), 0))?;
        let mut row: Vec<(usize, f64)> = cols.iter().zip(vals).map(|(&j, &x)| (j, x / d)).collect();
        row.push((i, 0.0));
        rows.push(row);
        bs[i] = b[i] / d;
    }
    let s = Csr::from_rows(rows);
    // Cell-pressure system: pressure rows × pressure columns of the cells.
    // Well unknowns are left to the ILU(0) stage (including them made the
    // aggregation hierarchy diverge on a thermal test system).
    let p_dof: Vec<usize> = (0..n_cells).map(|c| c * nb).collect();
    let mut pmap = vec![usize::MAX; n];
    for (q, &k) in p_dof.iter().enumerate() {
        pmap[k] = q;
    }
    let prow: Vec<Vec<(usize, f64)>> = p_dof.iter().map(|&k| { let (c, v) = s.row(k); c.iter().zip(v).filter(|(&j, _)| pmap[j] != usize::MAX).map(|(&j, &x)| (pmap[j], x)).collect() }).collect();
    let ap = Csr::from_rows(prow);
    let amg = Amg::new(ap).ok_or_else(|| ("AMG setup failed (singular coarse system)".to_string(), 0))?;
    let ilu = Ilu0::new(s.clone()).ok_or_else(|| ("ILU(0) breakdown (zero pivot)".to_string(), 0))?;
    let a_pcols = {
        let mut ptr = vec![0usize];
        let (mut idx, mut val) = (Vec::new(), Vec::new());
        for i in 0..s.n {
            let (c, v) = s.row(i);
            for (&j, &x) in c.iter().zip(v) {
                if pmap[j] != usize::MAX {
                    idx.push(pmap[j]);
                    val.push(x);
                }
            }
            ptr.push(idx.len());
        }
        Csr { n: s.n, ptr, idx, val }
    };
    let pc = Cpr { amg, ilu, p_dof, a_pcols };
    gmres(&s, &bs, &pc, rtol, max_iter)
}

/// Right-preconditioned restarted GMRES(m), modified Gram–Schmidt with
/// Givens rotations. Converged when ‖b − A x‖ ≤ rtol ‖b‖ (true residual
/// checked at the end).
fn gmres(a: &Csr, b: &[f64], pc: &Cpr, rtol: f64, max_iter: usize) -> Result<(Vec<f64>, usize), Failure> {
    const M: usize = 40;
    let n = a.n;
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let bn = norm(b);
    let mut x = vec![0.0; n];
    if bn == 0.0 {
        return Ok((x, 0));
    }
    let target = rtol * bn;
    let mut it = 0usize;
    let mut r = b.to_vec();
    loop {
        let beta = norm(&r);
        if !beta.is_finite() {
            return Err(("residual became non-finite".to_string(), it));
        }
        if beta <= target {
            return Ok((x, it));
        }
        if it >= max_iter {
            return Err((format!("did not converge in {max_iter} iterations (relative residual {:.2e}, tolerance {rtol:.1e})", beta / bn), it));
        }
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|x| x / beta).collect()];
        let mut z: Vec<Vec<f64>> = Vec::new();
        let mut h = vec![vec![0.0f64; M]; M + 1];
        let (mut cs, mut sn) = (vec![0.0f64; M], vec![0.0f64; M]);
        let mut g = vec![0.0f64; M + 1];
        g[0] = beta;
        let mut k_done = 0usize;
        for k in 0..M {
            let mut zk = vec![0.0; n];
            pc.apply(a, &v[k], &mut zk);
            let mut w = vec![0.0; n];
            a.matvec(&zk, &mut w);
            z.push(zk);
            for (j, vj) in v.iter().enumerate() {
                let hj: f64 = w.iter().zip(vj).map(|(a, b)| a * b).sum();
                h[j][k] = hj;
                w.iter_mut().zip(vj).for_each(|(wi, vi)| *wi -= hj * vi);
            }
            let hn = norm(&w);
            h[k + 1][k] = hn;
            for j in 0..k {
                let t = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
                h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
                h[j][k] = t;
            }
            let d = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            if d == 0.0 || !d.is_finite() {
                return Err(("breakdown (zero Hessenberg column)".to_string(), it));
            }
            cs[k] = h[k][k] / d;
            sn[k] = h[k + 1][k] / d;
            h[k][k] = d;
            h[k + 1][k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k] *= cs[k];
            it += 1;
            k_done = k + 1;
            if g[k + 1].abs() <= target || it >= max_iter || hn == 0.0 {
                break;
            }
            v.push(w.iter().map(|x| x / hn).collect());
        }
        // Back substitution and update.
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev() {
            let s: f64 = (i + 1..k_done).map(|j| h[i][j] * y[j]).sum();
            y[i] = (g[i] - s) / h[i][i];
        }
        for (j, yj) in y.iter().enumerate() {
            x.iter_mut().zip(&z[j]).for_each(|(xi, zi)| *xi += yj * zi);
        }
        let mut ax = vec![0.0; n];
        a.matvec(&x, &mut ax);
        r = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    }
}
