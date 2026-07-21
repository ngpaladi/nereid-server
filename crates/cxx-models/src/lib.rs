//! Compile-time C++ inference models for nereid-server, bridged with [`cxx`].
//!
//! A C++ model implements the `nereid::Model` interface in `cpp/` and is
//! registered by name in `create_model`. This crate is meant to be vendored as a
//! git submodule / workspace member: adding a model is dropping in its C++ and
//! one registry line, then rebuilding — the boundary is type-safe (no hand-rolled
//! FFI), and the server links this crate behind its `cxx` feature.
//!
//! The server only ever sees the plain-Rust [`CxxModel`] interface below; no
//! `cxx` types leak across the crate boundary.

#[cxx::bridge(namespace = "nereid")]
mod ffi {
    unsafe extern "C++" {
        include!("cxx-models/cpp/models.h");

        /// An opaque, compiled-in C++ model.
        type Model;

        /// Whether a model with this name is compiled in.
        fn model_exists(name: &str) -> bool;

        /// Construct the named model, or a null pointer if it isn't registered.
        fn create_model(name: &str) -> UniquePtr<Model>;

        /// Run one inference. `data` is row-major little-endian for `dtype`; the
        /// output tensor is written to the `out_*` buffers. A thrown C++
        /// exception becomes an `Err` (its `what()` message).
        fn run(
            self: &Model,
            dtype: &str,
            shape: &[i64],
            data: &[u8],
            out_shape: &mut Vec<i64>,
            out_dtype: &mut String,
            out_data: &mut Vec<u8>,
        ) -> Result<()>;
    }
}

/// A loaded C++ model exposing a plain-Rust interface (no `cxx` types leak to
/// callers). Returns `(shape, canonical dtype, row-major little-endian bytes)`.
pub trait CxxModel: Send {
    fn run(
        &self,
        dtype: &str,
        shape: &[i64],
        data: &[u8],
    ) -> Result<(Vec<i64>, String, Vec<u8>), String>;
}

/// Whether a C++ model with this name is compiled into the binary.
pub fn model_exists(name: &str) -> bool {
    ffi::model_exists(name)
}

/// Construct the named compiled-in C++ model, or `None` if it isn't registered.
pub fn create(name: &str) -> Option<Box<dyn CxxModel>> {
    let model = ffi::create_model(name);
    if model.is_null() {
        return None;
    }
    Some(Box::new(Loaded(model)))
}

struct Loaded(cxx::UniquePtr<ffi::Model>);

// SAFETY: the model is only ever accessed from the single worker thread that
// owns it (the server runs each model on its own thread), and inference reads
// immutable model state and touches no shared or thread-local data.
unsafe impl Send for Loaded {}

impl CxxModel for Loaded {
    fn run(
        &self,
        dtype: &str,
        shape: &[i64],
        data: &[u8],
    ) -> Result<(Vec<i64>, String, Vec<u8>), String> {
        let model = self
            .0
            .as_ref()
            .ok_or_else(|| "null C++ model".to_string())?;
        let mut out_shape: Vec<i64> = Vec::new();
        let mut out_dtype = String::new();
        let mut out_data: Vec<u8> = Vec::new();
        model
            .run(
                dtype,
                shape,
                data,
                &mut out_shape,
                &mut out_dtype,
                &mut out_data,
            )
            .map_err(|e| e.to_string())?;
        Ok((out_shape, out_dtype, out_data))
    }
}
