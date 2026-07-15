//! A model's declared tensor **contract** and all request/response validation.
//!
//! Parsed from `model_inference.textproto`, backend-agnostic, and owned by the
//! core server — so dtype/shape/batch validation is identical for every backend.
//! A single-tensor model is represented as one input named `"input"` and one
//! output named `"output"`; an output-only model has no inputs. The nested
//! `input {}` / `output {}` form yields the multi-tensor case directly.

use std::path::Path;

use tonic::Status;

use crate::dtype;

/// One named tensor in a model's contract. `dims` excludes the batch dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    /// KServe datatype, e.g. `"FP32"`.
    pub dtype: String,
    pub dims: Vec<i64>,
}

/// A model's declared inputs and outputs plus its max batch size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Contract {
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
    pub max_batch_size: i64,
    /// Whether output datatypes are authoritative and must match what the model
    /// returns. True for the nested (multi-tensor) form, where each output's
    /// `data_type` is declared; false for the flat single-tensor form, where the
    /// output dtype is whatever the model produces.
    pub strict_output_dtype: bool,
}

impl Contract {
    /// Parse the `model_inference.textproto` in `model_dir`.
    pub fn parse(model_dir: &Path) -> Result<Contract, Status> {
        if textproto_is_multi(model_dir) {
            parse_multi(model_dir)
        } else {
            parse_flat(model_dir)
        }
    }

    /// Whether the model consumes a tensor input.
    pub fn has_input(&self) -> bool {
        !self.inputs.is_empty()
    }

    /// True when a batch dimension is declared.
    pub fn has_batch_dim(&self) -> bool {
        self.max_batch_size > 0
    }

    /// Validate a request tensor's shape against `spec`, allowing an omitted
    /// batch dimension (auto-expanded later) exactly as the model declares.
    pub fn validate_input_shape(
        &self,
        spec: &TensorSpec,
        request_shape: &[i64],
        model_name: &str,
    ) -> Result<(), Status> {
        validate_tensor_shape(&spec.dims, self.max_batch_size, request_shape, model_name)
    }

    /// Return the effective shape to feed the model, inserting a leading batch
    /// dimension of 1 when the request omitted the declared batch dimension.
    /// Expects `request_shape` to have already passed [`Self::validate_input_shape`].
    pub fn normalize_request_shape(&self, spec: &TensorSpec, request_shape: Vec<i64>) -> Vec<i64> {
        if self.has_batch_dim() && request_shape.len() == spec.dims.len() {
            let mut expanded = Vec::with_capacity(request_shape.len() + 1);
            expanded.push(1);
            expanded.extend(request_shape);
            expanded
        } else {
            request_shape
        }
    }

    /// The expected request batch size (leading dim) if the request carries a
    /// batch dimension, else `None`. `normalized` must be batch-normalized.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub fn expected_batch(&self, normalized: &[i64]) -> Option<i64> {
        if self.has_batch_dim() {
            normalized.first().copied()
        } else {
            None
        }
    }

    /// Validate a model's output tensor shape against `spec` (dims), and — when
    /// the batch dimension is present and `expected_batch` is `Some` — that its
    /// batch size matches the request. A wrong-shaped output is a model/config
    /// bug, so this returns `Status::internal`.
    pub fn validate_output_shape(
        &self,
        spec: &TensorSpec,
        actual: &[i64],
        expected_batch: Option<i64>,
        model_name: &str,
    ) -> Result<(), Status> {
        let declared = &spec.dims;
        let offset = if self.has_batch_dim() && actual.len() == declared.len() + 1 {
            1
        } else if actual.len() == declared.len() {
            0
        } else {
            let allowed = if self.has_batch_dim() {
                format!(
                    "{} (or {} with a batch dimension)",
                    declared.len(),
                    declared.len() + 1
                )
            } else {
                declared.len().to_string()
            };
            return Err(Status::internal(format!(
                "model '{model_name}' output rank mismatch: expected {allowed}, got {}",
                actual.len()
            )));
        };

        if offset == 1
            && let Some(expected) = expected_batch
            && actual[0] != expected
        {
            return Err(Status::internal(format!(
                "model '{model_name}' output batch size {} does not match input batch size {expected}",
                actual[0]
            )));
        }

        for (index, expected_dim) in declared.iter().enumerate() {
            let actual_dim = actual[index + offset];
            if *expected_dim != -1 && *expected_dim != actual_dim {
                return Err(Status::internal(format!(
                    "model '{model_name}' output shape mismatch at dimension {}: declared {}, got {}",
                    index + offset,
                    expected_dim,
                    actual_dim
                )));
            }
        }
        Ok(())
    }

    /// The advertised metadata shape for `spec`: a leading `-1` (variable batch)
    /// when a batch dimension is declared, followed by the declared dims.
    pub fn metadata_shape(&self, spec: &TensorSpec) -> Vec<i64> {
        let mut shape = Vec::with_capacity(spec.dims.len() + 1);
        if self.has_batch_dim() {
            shape.push(-1);
        }
        shape.extend_from_slice(&spec.dims);
        shape
    }
}

fn validate_tensor_shape(
    input_shape: &[i64],
    max_batch_size: i64,
    request_shape: &[i64],
    model_name: &str,
) -> Result<(), Status> {
    if request_shape.is_empty() {
        return Err(Status::invalid_argument("tensor chunk shape is required"));
    }
    if request_shape.iter().any(|dim| *dim <= 0) {
        return Err(Status::invalid_argument(
            "tensor chunk shape dimensions must be positive",
        ));
    }

    let has_batch = max_batch_size > 0;
    let batched_rank = input_shape.len() + usize::from(has_batch);

    // A request may carry the declared batch dimension or omit it (auto-expanded
    // to size 1).
    let shape_offset = if request_shape.len() == batched_rank {
        if has_batch {
            let batch_size = request_shape[0];
            if batch_size > max_batch_size {
                return Err(Status::invalid_argument(format!(
                    "batch size {batch_size} exceeds max_batch_size {max_batch_size} for model '{model_name}'"
                )));
            }
            1
        } else {
            0
        }
    } else if has_batch && request_shape.len() == input_shape.len() {
        0
    } else {
        let allowed = if has_batch {
            format!(
                "expected {} (or {} without the batch dimension)",
                batched_rank,
                input_shape.len()
            )
        } else {
            format!("expected {batched_rank}")
        };
        return Err(Status::invalid_argument(format!(
            "input tensor rank mismatch for model '{model_name}': {allowed}, got {}",
            request_shape.len()
        )));
    };

    for (index, expected_dim) in input_shape.iter().enumerate() {
        let actual_dim = request_shape[index + shape_offset];
        if *expected_dim != -1 && *expected_dim != actual_dim {
            return Err(Status::invalid_argument(format!(
                "input tensor shape mismatch for model '{model_name}' at dimension {}: expected {}, got {}",
                index + shape_offset,
                expected_dim,
                actual_dim
            )));
        }
    }
    Ok(())
}

/// Whether a model's textproto uses the nested `input {}` / `output {}` blocks.
pub fn textproto_is_multi(model_dir: &Path) -> bool {
    let path = model_dir.join("model_inference.textproto");
    let Ok(contents) = std::fs::read_to_string(path) else {
        return false;
    };
    contents.lines().any(|raw| {
        let line = raw.split('#').next().unwrap_or("").trim();
        line.starts_with("input {")
            || line.starts_with("input{")
            || line.starts_with("output {")
            || line.starts_with("output{")
    })
}

/// Parse the flat single-tensor form (`input_shape` / `output_shape` /
/// `data_type` / `max_batch_size`) into a one-input, one-output `Contract`.
fn parse_flat(model_dir: &Path) -> Result<Contract, Status> {
    let config_path = model_dir.join("model_inference.textproto");
    let contents = std::fs::read_to_string(&config_path).map_err(|err| {
        Status::failed_precondition(format!("failed to read {}: {err}", config_path.display()))
    })?;

    fn parse_shape_dims(
        field: &str,
        raw_value: &str,
        raw_line: &str,
        path: &Path,
    ) -> Result<Vec<i64>, Status> {
        let inner = raw_value
            .trim()
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| {
                Status::failed_precondition(format!(
                    "invalid {field} in {}: '{raw_line}'. expected bracketed dimensions such as `{field}: [1, 16]`",
                    path.display()
                ))
            })?
            .trim();
        let dims_str: Vec<&str> = inner
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if dims_str.is_empty() {
            return Err(Status::failed_precondition(format!(
                "invalid {field} in {}: '{raw_line}'. expected bracketed dimensions such as `{field}: [1, 16]`",
                path.display()
            )));
        }
        let mut dims = Vec::with_capacity(dims_str.len());
        for dim_str in dims_str {
            let dim = dim_str.parse::<i64>().map_err(|err| {
                Status::failed_precondition(format!(
                    "failed parsing {field} dimension '{dim_str}' in {}: {err}",
                    path.display()
                ))
            })?;
            if dim == 0 || dim < -1 {
                return Err(Status::failed_precondition(format!(
                    "{field} dimensions in {} must be positive or -1",
                    path.display()
                )));
            }
            dims.push(dim);
        }
        Ok(dims)
    }

    let mut input_shape = None::<Vec<i64>>;
    let mut output_shape = None::<Vec<i64>>;
    let mut data_type = None::<String>;
    let mut max_batch_size = 0i64;
    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (field, raw_value) = line.split_once(':').ok_or_else(|| {
            Status::failed_precondition(format!(
                "invalid model_inference line in {}: '{raw_line}'",
                config_path.display()
            ))
        })?;
        match field.trim() {
            "input_shape" => {
                input_shape = Some(parse_shape_dims(
                    "input_shape",
                    raw_value,
                    raw_line,
                    &config_path,
                )?)
            }
            "output_shape" => {
                output_shape = Some(parse_shape_dims(
                    "output_shape",
                    raw_value,
                    raw_line,
                    &config_path,
                )?)
            }
            "data_type" => {
                let value = raw_value.trim().trim_matches('"').to_string();
                if dtype::kserve_fixed_width(&value).is_none() {
                    return Err(Status::failed_precondition(format!(
                        "unsupported data_type '{value}' in {}",
                        config_path.display()
                    )));
                }
                data_type = Some(value);
            }
            "max_batch_size" => {
                max_batch_size = raw_value.trim().parse::<i64>().map_err(|err| {
                    Status::failed_precondition(format!(
                        "failed parsing max_batch_size in {}: {err}",
                        config_path.display()
                    ))
                })?;
                if max_batch_size < 0 {
                    return Err(Status::failed_precondition(format!(
                        "max_batch_size in {} must be >= 0",
                        config_path.display()
                    )));
                }
            }
            _ => {}
        }
    }

    if input_shape.is_none() && output_shape.is_none() {
        return Err(Status::failed_precondition(format!(
            "{} must declare input_shape and/or output_shape",
            config_path.display()
        )));
    }

    let dt = data_type.unwrap_or_else(|| "FP32".to_string());
    let inputs = input_shape
        .map(|dims| {
            vec![TensorSpec {
                name: "input".to_string(),
                dtype: dt.clone(),
                dims,
            }]
        })
        .unwrap_or_default();
    let outputs = output_shape
        .map(|dims| {
            vec![TensorSpec {
                name: "output".to_string(),
                dtype: dt.clone(),
                dims,
            }]
        })
        .unwrap_or_default();

    Ok(Contract {
        inputs,
        outputs,
        max_batch_size,
        strict_output_dtype: false,
    })
}

/// Parse the nested multi-tensor form into a `Contract` (output dtypes are
/// authoritative).
fn parse_multi(model_dir: &Path) -> Result<Contract, Status> {
    let config_path = model_dir.join("model_inference.textproto");
    let contents = std::fs::read_to_string(&config_path).map_err(|err| {
        Status::failed_precondition(format!("failed to read {}: {err}", config_path.display()))
    })?;

    let fail =
        |msg: String| Status::failed_precondition(format!("{msg} in {}", config_path.display()));

    #[derive(Default)]
    struct Pending {
        name: Option<String>,
        dtype: Option<String>,
        dims: Option<Vec<i64>>,
    }

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut max_batch_size = 0i64;
    let mut block: Option<bool> = None; // Some(true)=input, Some(false)=output
    let mut pending = Pending::default();
    let unquote = |s: &str| s.trim().trim_matches('"').to_string();

    for raw_line in contents.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("input {") || line.starts_with("input{") {
            if block.is_some() {
                return Err(fail("nested/unclosed block".to_string()));
            }
            block = Some(true);
            pending = Pending::default();
            continue;
        }
        if line.starts_with("output {") || line.starts_with("output{") {
            if block.is_some() {
                return Err(fail("nested/unclosed block".to_string()));
            }
            block = Some(false);
            pending = Pending::default();
            continue;
        }
        if line == "}" {
            let is_input = block.ok_or_else(|| fail("stray '}'".to_string()))?;
            let name = pending
                .name
                .take()
                .ok_or_else(|| fail("block missing name".to_string()))?;
            let dims = pending
                .dims
                .take()
                .ok_or_else(|| fail(format!("input/output '{name}' missing dims")))?;
            let dtype = pending.dtype.take().unwrap_or_else(|| "FP32".to_string());
            if dtype::kserve_fixed_width(&dtype).is_none() {
                return Err(fail(format!(
                    "unsupported data_type '{dtype}' for '{name}'"
                )));
            }
            let spec = TensorSpec { name, dtype, dims };
            if is_input {
                inputs.push(spec);
            } else {
                outputs.push(spec);
            }
            block = None;
            continue;
        }

        let (field, value) = line
            .split_once(':')
            .ok_or_else(|| fail(format!("invalid line '{raw_line}'")))?;
        match (block, field.trim()) {
            (None, "max_batch_size") => {
                max_batch_size = value
                    .trim()
                    .parse::<i64>()
                    .map_err(|err| fail(format!("bad max_batch_size: {err}")))?;
                if max_batch_size < 0 {
                    return Err(fail("max_batch_size must be >= 0".to_string()));
                }
            }
            (Some(_), "name") => pending.name = Some(unquote(value)),
            (Some(_), "data_type") => pending.dtype = Some(unquote(value)),
            (Some(_), "dims") => {
                let inner = value
                    .trim()
                    .strip_prefix('[')
                    .and_then(|s| s.strip_suffix(']'))
                    .ok_or_else(|| fail(format!("dims must be bracketed: '{raw_line}'")))?;
                let mut dims = Vec::new();
                for d in inner.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let dim = d
                        .parse::<i64>()
                        .map_err(|err| fail(format!("bad dim '{d}': {err}")))?;
                    if dim == 0 || dim < -1 {
                        return Err(fail("dims must be positive or -1".to_string()));
                    }
                    dims.push(dim);
                }
                if dims.is_empty() {
                    return Err(fail("dims must be non-empty".to_string()));
                }
                pending.dims = Some(dims);
            }
            _ => {}
        }
    }

    if block.is_some() {
        return Err(fail("unclosed block".to_string()));
    }
    if inputs.is_empty() || outputs.is_empty() {
        return Err(fail(
            "multi-tensor model needs at least one input and one output".to_string(),
        ));
    }

    Ok(Contract {
        inputs,
        outputs,
        max_batch_size,
        strict_output_dtype: true,
    })
}

#[cfg(test)]
mod tests {
    use super::{Contract, TensorSpec, textproto_is_multi};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("nereid-contract-{prefix}-{nanos}"));
        fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn spec(name: &str, dims: Vec<i64>) -> TensorSpec {
        TensorSpec {
            name: name.to_string(),
            dtype: "FP32".to_string(),
            dims,
        }
    }

    /// A single-input contract, as parsed from the flat form.
    fn input_contract(dims: Vec<i64>, max_batch_size: i64) -> Contract {
        Contract {
            inputs: vec![spec("input", dims)],
            outputs: Vec::new(),
            max_batch_size,
            strict_output_dtype: false,
        }
    }

    #[test]
    fn flat_contract_allows_variable_dims_and_max_batch_size() {
        let base = temp_dir("flat-variable");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [-1, 16]\nmax_batch_size: 8\n",
        )
        .expect("write textproto");

        let contract = Contract::parse(&base).expect("parse contract");
        assert_eq!(contract.inputs.len(), 1);
        assert_eq!(contract.inputs[0].dims, vec![-1, 16]);
        assert_eq!(contract.max_batch_size, 8);
        assert!(contract.outputs.is_empty());
        assert!(!contract.strict_output_dtype);

        let spec = &contract.inputs[0];
        contract
            .validate_input_shape(spec, &[4, 10, 16], "model")
            .expect("shape should match");
        assert!(
            contract
                .validate_input_shape(spec, &[9, 10, 16], "model")
                .is_err()
        );
        assert!(
            contract
                .validate_input_shape(spec, &[4, 10, 15], "model")
                .is_err()
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn multi_contract_parses_nested_blocks() {
        let base = temp_dir("multi");
        fs::write(
            base.join("model_inference.textproto"),
            b"max_batch_size: 4\n\
              input {\n  name: \"a\"\n  data_type: \"FP32\"\n  dims: [4]\n}\n\
              input {\n  name: \"b\"\n  dims: [4]\n}\n\
              output {\n  name: \"sum\"\n  data_type: \"FP32\"\n  dims: [4]\n}\n",
        )
        .expect("write textproto");

        assert!(textproto_is_multi(&base), "nested blocks -> multi");
        let contract = Contract::parse(&base).expect("parse multi contract");
        assert_eq!(contract.max_batch_size, 4);
        assert!(contract.strict_output_dtype);
        assert_eq!(contract.inputs.len(), 2);
        assert_eq!(contract.inputs[0].name, "a");
        assert_eq!(contract.inputs[0].dtype, "FP32");
        assert_eq!(contract.inputs[0].dims, vec![4]);
        // data_type defaults to FP32 when omitted.
        assert_eq!(contract.inputs[1].name, "b");
        assert_eq!(contract.inputs[1].dtype, "FP32");
        assert_eq!(contract.outputs.len(), 1);
        assert_eq!(contract.outputs[0].name, "sum");

        // A flat single-tensor textproto is NOT detected as multi.
        let flat = temp_dir("flat");
        fs::write(
            flat.join("model_inference.textproto"),
            b"input_shape: [16]\nmax_batch_size: 8\n",
        )
        .expect("write flat");
        assert!(!textproto_is_multi(&flat), "flat form -> not multi");

        let _ = fs::remove_dir_all(&base);
        let _ = fs::remove_dir_all(&flat);
    }

    #[test]
    fn multi_contract_rejects_block_missing_dims() {
        let base = temp_dir("multi-bad");
        fs::write(
            base.join("model_inference.textproto"),
            b"input {\n  name: \"a\"\n}\noutput {\n  name: \"y\"\n  dims: [4]\n}\n",
        )
        .expect("write");
        assert!(
            Contract::parse(&base).is_err(),
            "input without dims must be rejected"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn invalid_output_shape_error_names_output_shape() {
        let base = temp_dir("bad-output-shape");
        // A malformed output_shape must report `output_shape`, not `input_shape`.
        fs::write(
            base.join("model_inference.textproto"),
            b"output_shape: not-a-list\n",
        )
        .expect("write textproto");
        let err = Contract::parse(&base).expect_err("should reject");
        assert!(
            err.message().contains("output_shape"),
            "error should name output_shape, got: {}",
            err.message()
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn flat_contract_parses_output_shape_when_present() {
        let base = temp_dir("output-shape");
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [4]\nmax_batch_size: 4\noutput_shape: [4]\n",
        )
        .expect("write textproto");

        let contract = Contract::parse(&base).expect("parse contract");
        assert_eq!(contract.outputs.len(), 1);
        assert_eq!(contract.outputs[0].dims, vec![4]);

        // A contract with no output_shape line has no declared output.
        fs::write(
            base.join("model_inference.textproto"),
            b"input_shape: [4]\nmax_batch_size: 4\n",
        )
        .expect("rewrite textproto");
        let no_output = Contract::parse(&base).expect("parse contract");
        assert!(no_output.outputs.is_empty());

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn validate_output_shape_accepts_declared_and_rejects_mismatch() {
        let contract = Contract {
            inputs: vec![spec("input", vec![4])],
            outputs: vec![spec("output", vec![4])],
            max_batch_size: 4,
            strict_output_dtype: false,
        };
        let out = &contract.outputs[0];

        // With or without the optional batch dimension.
        contract
            .validate_output_shape(out, &[1, 4], Some(1), "m")
            .expect("batched output ok");
        contract
            .validate_output_shape(out, &[4], None, "m")
            .expect("unbatched output ok");

        // Wrong trailing dim and wrong rank are both rejected.
        assert!(
            contract
                .validate_output_shape(out, &[1, 5], Some(1), "m")
                .is_err()
        );
        assert!(
            contract
                .validate_output_shape(out, &[1, 1, 4], Some(1), "m")
                .is_err()
        );

        // Output batch that disagrees with the input batch is rejected.
        assert!(
            contract
                .validate_output_shape(out, &[3, 4], Some(2), "m")
                .is_err(),
            "output batch 3 must not pass for input batch 2"
        );
        contract
            .validate_output_shape(out, &[2, 4], Some(2), "m")
            .expect("matching batch ok");

        // A -1 declared dim matches any positive size.
        let variable = Contract {
            inputs: vec![spec("input", vec![4])],
            outputs: vec![spec("output", vec![-1])],
            max_batch_size: 0,
            strict_output_dtype: false,
        };
        variable
            .validate_output_shape(&variable.outputs[0], &[7], None, "m")
            .expect("variable output dim ok");
    }

    #[test]
    fn flat_contract_without_batch_uses_request_shape_directly() {
        let contract = input_contract(vec![-1, 16], 0);
        let spec = &contract.inputs[0];
        contract
            .validate_input_shape(spec, &[10, 16], "model")
            .expect("shape should match");
        assert!(
            contract
                .validate_input_shape(spec, &[1, 10, 16], "model")
                .is_err()
        );
    }

    #[test]
    fn flat_contract_auto_expands_missing_batch_dim() {
        let contract = input_contract(vec![16], 10);
        let spec = &contract.inputs[0];

        // Bare shape (batch omitted) and explicit batch shape both validate.
        contract
            .validate_input_shape(spec, &[16], "model")
            .expect("bare shape should be accepted");
        contract
            .validate_input_shape(spec, &[2, 16], "model")
            .expect("explicit batch shape should be accepted");

        // The bare shape is expanded to a leading batch dimension of 1, while an
        // explicit batch shape is passed through untouched.
        assert_eq!(
            contract.normalize_request_shape(spec, vec![16]),
            vec![1, 16]
        );
        assert_eq!(
            contract.normalize_request_shape(spec, vec![2, 16]),
            vec![2, 16]
        );

        // A genuinely wrong rank is still rejected.
        assert!(
            contract
                .validate_input_shape(spec, &[1, 1, 16], "model")
                .is_err()
        );
    }

    #[test]
    fn flat_contract_allows_multiple_variable_dims() {
        let contract = input_contract(vec![-1, -1, 16], 4);
        let spec = &contract.inputs[0];
        contract
            .validate_input_shape(spec, &[2, 5, 7, 16], "model")
            .expect("shape should match");
        assert!(
            contract
                .validate_input_shape(spec, &[2, 5, 7, 15], "model")
                .is_err()
        );
    }
}
