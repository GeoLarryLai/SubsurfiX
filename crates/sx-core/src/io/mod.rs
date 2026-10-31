// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! Generic file I/O for stage products: Parquet batches, CSV reports, atomic
//! writes, the ingest products, the column glossary and per-well LAS export.

pub mod atomic;
pub mod csv;
pub mod glossary;
pub mod las;
pub mod parquet;
pub mod products;
