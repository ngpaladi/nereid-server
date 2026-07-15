//! Small helpers shared by the in-process native engines (ONNX, TensorFlow):
//! POD byte<->slice reinterpretation and the canonical-dtype → Rust element type
//! dispatch macros.

use tonic::Status;

/// Reinterpret a little-endian byte buffer as `Vec<T>` (T: POD). Validates the
/// length is a whole number of elements.
pub(crate) fn bytes_to_vec<T: Copy>(bytes: &[u8], model: &str) -> Result<Vec<T>, Status> {
    let esz = std::mem::size_of::<T>();
    if esz == 0 || !bytes.len().is_multiple_of(esz) {
        return Err(Status::invalid_argument(format!(
            "model '{model}': input byte length {} is not a multiple of element size {esz}",
            bytes.len()
        )));
    }
    let n = bytes.len() / esz;
    let mut out = Vec::<T>::with_capacity(n);
    // SAFETY: T is POD; we copy exactly n*esz bytes from a validated buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        out.set_len(n);
    }
    Ok(out)
}

/// Reinterpret a `&[T]` (POD) as its little-endian bytes.
pub(crate) fn slice_to_bytes<T: Copy>(slice: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(slice);
    let mut out = vec![0u8; byte_len];
    // SAFETY: copying POD bytes out of a valid slice into an equally-sized buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(slice.as_ptr() as *const u8, out.as_mut_ptr(), byte_len);
    }
    out
}

/// Applies `$body` with `$t` bound to the Rust element type for a canonical
/// dtype name. Unknown names hit `$err`.
#[cfg(feature = "onnx")]
macro_rules! dispatch_dtype {
    ($dtype:expr, |$t:ident| $body:expr, $err:expr) => {
        match $dtype {
            "float32" => {
                type $t = f32;
                $body
            }
            "float64" => {
                type $t = f64;
                $body
            }
            "int8" => {
                type $t = i8;
                $body
            }
            "int16" => {
                type $t = i16;
                $body
            }
            "int32" => {
                type $t = i32;
                $body
            }
            "int64" => {
                type $t = i64;
                $body
            }
            "uint8" => {
                type $t = u8;
                $body
            }
            "uint16" => {
                type $t = u16;
                $body
            }
            "uint32" => {
                type $t = u32;
                $body
            }
            "uint64" => {
                type $t = u64;
                $body
            }
            "bool" => {
                type $t = bool;
                $body
            }
            "float16" => {
                type $t = half::f16;
                $body
            }
            "bfloat16" => {
                type $t = half::bf16;
                $body
            }
            _ => $err,
        }
    };
}
#[cfg(feature = "onnx")]
pub(crate) use dispatch_dtype;

/// Like [`dispatch_dtype`] but without `bfloat16` — the `tensorflow` crate does
/// not implement `TensorType` for `half::bf16`.
#[cfg(feature = "tensorflow")]
macro_rules! dispatch_dtype_tf {
    ($dtype:expr, |$t:ident| $body:expr, $err:expr) => {
        match $dtype {
            "float32" => {
                type $t = f32;
                $body
            }
            "float64" => {
                type $t = f64;
                $body
            }
            "int8" => {
                type $t = i8;
                $body
            }
            "int16" => {
                type $t = i16;
                $body
            }
            "int32" => {
                type $t = i32;
                $body
            }
            "int64" => {
                type $t = i64;
                $body
            }
            "uint8" => {
                type $t = u8;
                $body
            }
            "uint16" => {
                type $t = u16;
                $body
            }
            "uint32" => {
                type $t = u32;
                $body
            }
            "uint64" => {
                type $t = u64;
                $body
            }
            "bool" => {
                type $t = bool;
                $body
            }
            "float16" => {
                type $t = half::f16;
                $body
            }
            _ => $err,
        }
    };
}
#[cfg(feature = "tensorflow")]
pub(crate) use dispatch_dtype_tf;
