// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Process-wide thread pool setup.

use std::sync::OnceLock;

use crate::error::{SxError, SxResult};

static POOL_SIZE: OnceLock<usize> = OnceLock::new();

/// Build the global rayon pool once with `n_threads` workers.
///
/// `n_threads == 0` means "use the number of available cores".
/// Calling again with the same resolved size is a no-op. Calling again with a
/// different size returns `SxError::Config`.
pub fn init_threads(n_threads: usize) -> SxResult<usize> {
    let wanted = if n_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        n_threads
    };
    if let Some(&size) = POOL_SIZE.get() {
        return if size == wanted {
            Ok(size)
        } else {
            Err(SxError::Config(format!(
                "thread pool already built with {size} threads; asked for {wanted}"
            )))
        };
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(wanted)
        .build_global()
        .map_err(|e| SxError::Config(format!("cannot build thread pool: {e}")))?;
    let _ = POOL_SIZE.set(wanted);
    Ok(wanted)
}

/// Run `f` on the global pool if [`init_thread_pool`]-style setup already
/// built it; otherwise on a short-lived local pool, so library calls never
/// initialise the global pool behind the pipeline's back.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if POOL_SIZE.get().is_some() {
        return f();
    }
    match rayon::ThreadPoolBuilder::new().build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// Run `f` with `n_threads` workers (`0` = all cores). The first call builds
/// the global pool; a later call with the same size reuses it; a later call
/// with another size runs on its own short-lived pool, so two projects with
/// different thread counts can run in one process.
///
/// # Errors
/// `SxError::Config` when no pool can be built; otherwise the result of `f`.
pub fn with_threads<R: Send>(n_threads: usize, f: impl FnOnce() -> SxResult<R> + Send) -> SxResult<R> {
    let wanted = if n_threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        n_threads
    };
    let local = |f: Box<dyn FnOnce() -> SxResult<R> + Send + '_>| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(wanted)
            .build()
            .map_err(|e| SxError::Config(format!("cannot build thread pool: {e}")))?
            .install(f)
    };
    match POOL_SIZE.get() {
        Some(&size) if size == wanted => f(),
        Some(&size) => {
            tracing::info!(global = size, wanted, "running on a separate thread pool of the asked size");
            local(Box::new(f))
        }
        // The global pool may already exist without us (rayon builds it on
        // first use); then run on a pool of our own.
        None => match init_threads(wanted) {
            Ok(_) => f(),
            Err(_) => local(Box::new(f)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_threads_allows_a_second_size() {
        let a = with_threads(2, || Ok(rayon::current_num_threads())).unwrap();
        let b = with_threads(3, || Ok(rayon::current_num_threads())).unwrap();
        assert_eq!(b, 3);
        assert_eq!(a, 2);
    }
}
