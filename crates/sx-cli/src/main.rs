// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
//! `subsurfix` command line.

mod cli;
mod exit;
mod init;
mod nphi_offset;
mod pipeline_seam;
mod run;
mod stages;
mod status;

use std::process::ExitCode;

use clap::Parser;
use sx_core::SxResult;
use tracing_subscriber::filter::LevelFilter;

use cli::{Cli, Command};

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_logging(cli.verbose, cli.quiet);
    match dispatch(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(exit::exit_code(&e))
        }
    }
}

fn init_logging(verbose: bool, quiet: bool) {
    let level = if verbose {
        LevelFilter::DEBUG
    } else if quiet {
        LevelFilter::WARN
    } else {
        LevelFilter::INFO
    };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_max_level(level)
        .init();
}

fn dispatch(cli: &Cli) -> SxResult<()> {
    match &cli.command {
        Command::Init { dir } => init::execute(dir),
        Command::Run {
            config,
            from,
            to,
            force,
            threads,
        } => run::execute(config, from.as_deref(), to.as_deref(), *force, *threads),
        Command::Status { config } => status::execute(config),
        Command::NphiOffset { config } => nphi_offset::execute(config),
        Command::Version => {
            println!("subsurfix {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
    }
}
