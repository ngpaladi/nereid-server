//! KServe v2 datatype helpers.
//!
//! nereid's Python tensor path is byte-passthrough, so for a *fixed-width*
//! datatype it only needs the element byte size (to validate that a raw buffer
//! matches its shape) plus a canonical lowercase name that it hands to
//! `main.py` (`NEREID_INPUT_DTYPE`) and reads back from the framed output header.
//!
//! `BYTES` (variable-length strings) is intentionally unsupported — it isn't a
//! fixed-width element type — as is any datatype not in KServe's set.

/// For a fixed-width KServe datatype string (e.g. `"INT32"`), return its element
/// byte size and nereid's canonical lowercase name (e.g. `(4, "int32")`).
/// `None` for `BYTES` and any unknown datatype.
pub fn kserve_fixed_width(datatype: &str) -> Option<(usize, &'static str)> {
    Some(match datatype {
        "BOOL" => (1, "bool"),
        "UINT8" => (1, "uint8"),
        "UINT16" => (2, "uint16"),
        "UINT32" => (4, "uint32"),
        "UINT64" => (8, "uint64"),
        "INT8" => (1, "int8"),
        "INT16" => (2, "int16"),
        "INT32" => (4, "int32"),
        "INT64" => (8, "int64"),
        "FP16" => (2, "float16"),
        "FP32" => (4, "float32"),
        "FP64" => (8, "float64"),
        "BF16" => (2, "bfloat16"),
        _ => return None,
    })
}

/// Inverse of [`kserve_fixed_width`]: map a canonical lowercase name (as written
/// in a Python model's framed output header) back to its KServe datatype string
/// and element byte size. `None` for any unknown name.
pub fn canonical_to_kserve(canonical: &str) -> Option<(&'static str, usize)> {
    Some(match canonical {
        "bool" => ("BOOL", 1),
        "uint8" => ("UINT8", 1),
        "uint16" => ("UINT16", 2),
        "uint32" => ("UINT32", 4),
        "uint64" => ("UINT64", 8),
        "int8" => ("INT8", 1),
        "int16" => ("INT16", 2),
        "int32" => ("INT32", 4),
        "int64" => ("INT64", 8),
        "float16" => ("FP16", 2),
        "float32" => ("FP32", 4),
        "float64" => ("FP64", 8),
        "bfloat16" => ("BF16", 2),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{canonical_to_kserve, kserve_fixed_width};

    #[test]
    fn kserve_and_canonical_round_trip() {
        for dt in [
            "BOOL", "UINT8", "UINT16", "UINT32", "UINT64", "INT8", "INT16", "INT32", "INT64",
            "FP16", "FP32", "FP64", "BF16",
        ] {
            let (size, canonical) = kserve_fixed_width(dt).expect("known datatype");
            let (back, size2) = canonical_to_kserve(canonical).expect("known canonical");
            assert_eq!(back, dt, "round trip for {dt}");
            assert_eq!(size, size2, "size agreement for {dt}");
        }
    }

    #[test]
    fn unknown_and_variable_are_rejected() {
        assert_eq!(
            kserve_fixed_width("BYTES"),
            None,
            "BYTES is variable-length"
        );
        assert_eq!(kserve_fixed_width("NOPE"), None);
        assert_eq!(canonical_to_kserve("string"), None);
    }
}
