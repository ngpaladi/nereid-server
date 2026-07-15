//! Inference backends — one self-contained folder each.
//!
//! Backends are *discovered, not listed*: build.rs globs the subfolders of this
//! directory and emits one `mod` declaration per `<name>/mod.rs`, which this
//! module `include!`s. So adding a backend is just dropping a new folder here
//! (which can be a git submodule) — nothing in the tree needs editing, and there
//! is no central enum, match, detection list, or module list.
//!
//! Each backend's `mod.rs` holds its detection predicate and its
//! `inventory::submit!` registration (pure and dependency-free, so the core
//! discovers it at link time); only the engine in its feature-gated `imp`
//! submodule pulls in the backend's crate. The core registry that collects these
//! registrations lives in [`crate::backend`].

include!(concat!(env!("OUT_DIR"), "/backend_modules.rs"));
