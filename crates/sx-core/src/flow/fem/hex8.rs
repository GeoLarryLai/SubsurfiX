// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Trilinear 8-node hexahedron with 2 × 2 × 2 Gauss–Legendre integration.
//!
//! Nodes follow the corner-point order `c = di + 2·dj + 4·dk` of
//! [`CornerPointGrid::corners`](crate::grid::CornerPointGrid::corners); node
//! `c` sits at the reference point `(2di − 1, 2dj − 1, 2dk − 1)` of the cube
//! `[−1, 1]³`. Shape function `N_c = ⅛ (1 + ξ ξ_c)(1 + η η_c)(1 + ζ ζ_c)`.
//! Two points per direction integrate the trilinear Jacobian exactly, so the
//! element volume is exact.

#![allow(clippy::needless_range_loop)]

use crate::error::{SxError, SxResult};

/// Nodes per element.
pub const N_NODES: usize = 8;

/// Reference coordinates of node `c`.
fn node_ref(c: usize) -> [f64; 3] {
    [(c & 1) as f64 * 2.0 - 1.0, ((c >> 1) & 1) as f64 * 2.0 - 1.0, ((c >> 2) & 1) as f64 * 2.0 - 1.0]
}

/// Shape-function values at the reference point `xi`.
pub fn shape(xi: [f64; 3]) -> [f64; N_NODES] {
    std::array::from_fn(|c| {
        let r = node_ref(c);
        0.125 * (1.0 + xi[0] * r[0]) * (1.0 + xi[1] * r[1]) * (1.0 + xi[2] * r[2])
    })
}

/// Shape-function derivatives with respect to the reference coordinates.
fn shape_grad_ref(xi: [f64; 3]) -> [[f64; 3]; N_NODES] {
    std::array::from_fn(|c| {
        let r = node_ref(c);
        let f = [1.0 + xi[0] * r[0], 1.0 + xi[1] * r[1], 1.0 + xi[2] * r[2]];
        [0.125 * r[0] * f[1] * f[2], 0.125 * r[1] * f[0] * f[2], 0.125 * r[2] * f[0] * f[1]]
    })
}

/// One integration point of an element in physical space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussPoint {
    /// Shape-function values.
    pub n: [f64; N_NODES],
    /// Shape-function gradients (1/m).
    pub dn: [[f64; 3]; N_NODES],
    /// Weight × |det J| (m³).
    pub w: f64,
}

/// The 2 × 2 × 2 Gauss points of the element with corners `x` (m).
///
/// # Errors
/// `InvalidInput` when the element is degenerate (zero volume) or folded
/// (the Jacobian changes sign inside it).
pub fn gauss_points(x: &[[f64; 3]; N_NODES]) -> SxResult<[GaussPoint; 8]> {
    let g = 1.0 / 3f64.sqrt();
    let mut out = [GaussPoint { n: [0.0; N_NODES], dn: [[0.0; 3]; N_NODES], w: 0.0 }; 8];
    let mut sign = 0.0f64;
    for (q, gp) in out.iter_mut().enumerate() {
        let xi = [if q & 1 == 1 { g } else { -g }, if (q >> 1) & 1 == 1 { g } else { -g }, if (q >> 2) & 1 == 1 { g } else { -g }];
        let dr = shape_grad_ref(xi);
        // J[a][b] = ∂x_b / ∂ξ_a.
        let mut j = [[0.0f64; 3]; 3];
        for c in 0..N_NODES {
            for a in 0..3 {
                for b in 0..3 {
                    j[a][b] += dr[c][a] * x[c][b];
                }
            }
        }
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        if !(det.is_finite() && det != 0.0) || det * sign < 0.0 {
            return Err(SxError::InvalidInput("fem: degenerate or folded hexahedron".into()));
        }
        sign = det.signum();
        // Inverse of J by cofactors; ∇N = J⁻¹ ∂N/∂ξ.
        let inv = [
            [(j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det, (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det, (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det],
            [(j[1][2] * j[2][0] - j[1][0] * j[2][2]) / det, (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det, (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det],
            [(j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det, (j[0][1] * j[2][0] - j[0][0] * j[2][1]) / det, (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det],
        ];
        gp.n = shape(xi);
        for c in 0..N_NODES {
            for b in 0..3 {
                gp.dn[c][b] = (0..3).map(|a| inv[b][a] * dr[c][a]).sum();
            }
        }
        gp.w = det.abs();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sheared, stretched box (affine map), corners in corner-point order.
    fn skewed() -> [[f64; 3]; 8] {
        std::array::from_fn(|c| {
            let (i, j, k) = ((c & 1) as f64, ((c >> 1) & 1) as f64, ((c >> 2) & 1) as f64);
            [100.0 * i + 20.0 * k, 50.0 * j + 10.0 * i, 1000.0 + 5.0 * k + 2.0 * i]
        })
    }

    #[test]
    fn partition_of_unity() {
        for gp in gauss_points(&skewed()).unwrap() {
            assert!((gp.n.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            for b in 0..3 {
                assert!(gp.dn.iter().map(|d| d[b]).sum::<f64>().abs() < 1e-12);
            }
        }
    }

    #[test]
    fn linear_field_gradient_is_exact() {
        let x = skewed();
        let grad = [0.3, -1.2, 4.0];
        let f: Vec<f64> = x.iter().map(|p| 7.0 + grad[0] * p[0] + grad[1] * p[1] + grad[2] * p[2]).collect();
        for gp in gauss_points(&x).unwrap() {
            for b in 0..3 {
                let g: f64 = (0..N_NODES).map(|c| gp.dn[c][b] * f[c]).sum();
                assert!((g - grad[b]).abs() < 1e-10, "{g} vs {}", grad[b]);
            }
        }
    }

    #[test]
    fn volume_of_affine_element() {
        // Edge vectors (100, 10, 2), (0, 50, 0), (20, 0, 5): volume = |det|.
        let v: f64 = gauss_points(&skewed()).unwrap().iter().map(|gp| gp.w).sum();
        let det = 100.0 * (50.0 * 5.0) - 10.0 * (0.0) + 2.0 * (0.0 - 50.0 * 20.0);
        assert!((v - f64::abs(det)).abs() < 1e-8 * v, "{v} vs {det}");
    }

    #[test]
    fn volume_matches_corner_point_grid() {
        // One cell between parallel inclined pillars, tilted top and base: a parallelepiped.
        let coord = vec![
            0.0, 0.0, 0.0, 40.0, 0.0, 100.0, //
            100.0, 0.0, 0.0, 140.0, 0.0, 100.0, //
            0.0, 50.0, 0.0, 40.0, 50.0, 100.0, //
            100.0, 50.0, 0.0, 140.0, 50.0, 100.0,
        ];
        let zcorn = vec![20.0, 22.0, 20.0, 22.0, 30.0, 32.0, 30.0, 32.0];
        let g = crate::grid::CornerPointGrid::new([1, 1, 1], coord, zcorn, None).unwrap();
        let v: f64 = gauss_points(&g.corners(0)).unwrap().iter().map(|gp| gp.w).sum();
        assert!((v - g.volume(0)).abs() < 1e-9 * v, "{v} vs {}", g.volume(0));
    }

    #[test]
    fn patch_test_linear_pressure_has_no_interior_flux() {
        // 2 × 2 × 2 elements on a 3 × 3 × 3 lattice with the centre node and one
        // corner moved: a linear field gives zero Galerkin flux ∫ ∇N_a · K ∇p at the interior node.
        let node = |i: usize, j: usize, k: usize| -> [f64; 3] {
            let mut p = [10.0 * i as f64, 20.0 * j as f64, 1000.0 + 5.0 * k as f64];
            if (i, j, k) == (1, 1, 1) {
                p = [p[0] + 2.0, p[1] - 3.0, p[2] + 0.7];
            }
            if (i, j, k) == (2, 2, 2) {
                p = [p[0] + 1.5, p[1] + 2.5, p[2] - 0.4];
            }
            p
        };
        let (grad, kd) = ([0.3, -1.2, 4.0], [1.0, 2.0, 0.1]);
        let field = |p: [f64; 3]| 7.0 + grad[0] * p[0] + grad[1] * p[1] + grad[2] * p[2];
        let (mut flux, mut scale) = (0.0, 0.0);
        for e in 0..8 {
            let (ei, ej, ek) = (e & 1, (e >> 1) & 1, (e >> 2) & 1);
            let x: [[f64; 3]; 8] = std::array::from_fn(|c| node(ei + (c & 1), ej + ((c >> 1) & 1), ek + ((c >> 2) & 1)));
            // The centre node is corner (1 − ei, 1 − ej, 1 − ek) of this element.
            let a = (1 - ei) + 2 * (1 - ej) + 4 * (1 - ek);
            let f: Vec<f64> = x.iter().map(|&p| field(p)).collect();
            for gp in gauss_points(&x).unwrap() {
                let gh: [f64; 3] = std::array::from_fn(|b| (0..N_NODES).map(|c| gp.dn[c][b] * f[c]).sum());
                let term: f64 = (0..3).map(|b| gp.dn[a][b] * kd[b] * gh[b]).sum::<f64>() * gp.w;
                flux += term;
                scale += term.abs();
            }
        }
        assert!(flux.abs() < 1e-12 * scale, "interior flux {flux} (scale {scale})");
    }

    #[test]
    fn flat_element_is_rejected() {
        let mut x = skewed();
        for c in 4..8 {
            x[c] = x[c - 4];
        }
        assert!(gauss_points(&x).is_err());
    }
}
