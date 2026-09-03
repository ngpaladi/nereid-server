//! Prometheus metrics for inference requests, in Triton's names.
//!
//! nereid exposes the *inference request* subset of NVIDIA Triton's metrics —
//! the `nv_inference_*` counts, cumulative latencies, and the pending-request
//! gauge — rendered in the text exposition format for the `/metrics` endpoint
//! (`crate::http`, on `server.http_addr`), so a Prometheus scrape config,
//! Grafana dashboard, or alert rule written for Triton reads nereid unchanged. Every series carries Triton's `model` and `version`
//! labels; nereid's single implicit version is `"1"`.
//!
//! Deliberately out of scope: Triton's GPU/CPU/memory utilization metrics, the
//! response-cache metrics, and the opt-in latency summaries/histograms.
//!
//! How a request is accounted:
//!
//! - A [`RequestTimer`] is opened when a request has named a configured model
//!   (both the KServe `ModelInfer` and native `Checkpoint` surfaces open one),
//!   and the model's pending gauge goes up.
//! - The frontend marks the input built ([`RequestTimer::input_ready`]:
//!   `compute_input`); the `ModelManager` marks the execution
//!   ([`RequestTimer::executed`]: `queue` = waiting for a blocking worker,
//!   `compute_infer` = the backend's `infer`, plus the batch size).
//! - The request ends with [`RequestTimer::succeed`] (the remainder is
//!   `compute_output`; counts and durations are committed, as in Triton, only
//!   for successful requests) or [`RequestTimer::fail`] (one failure counted
//!   under a Triton `reason`). A timer dropped before either — the handler's
//!   future was cancelled because the client went away — counts as `CANCELED`.
//!
//! The counters are plain atomics and the model set is fixed at startup, so
//! recording is lock-free and the registry never grows.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tonic::{Code, Status};

use crate::triton::MODEL_VERSION;

/// The Prometheus text exposition format's content type, for whoever serves
/// [`InferenceMetrics::render`] over HTTP.
pub const TEXT_FORMAT: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Triton's `reason` label on `nv_inference_request_failure`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureReason {
    /// The scheduler refused the request (nereid: the model's queue was full).
    Rejected,
    /// The client cancelled / went away before the response.
    Canceled,
    /// The backend failed while executing the model.
    Backend,
    /// Anything else — chiefly request validation failures.
    Other,
}

impl FailureReason {
    const ALL: [FailureReason; 4] = [
        FailureReason::Rejected,
        FailureReason::Canceled,
        FailureReason::Backend,
        FailureReason::Other,
    ];

    fn label(self) -> &'static str {
        match self {
            FailureReason::Rejected => "REJECTED",
            FailureReason::Canceled => "CANCELED",
            FailureReason::Backend => "BACKEND",
            FailureReason::Other => "OTHER",
        }
    }

    /// Classify a failing gRPC status. Frontends use this for failures whose
    /// origin the `ModelManager` did not already pin down: a full queue is a
    /// scheduler rejection, a cancelled call is the client's doing, and
    /// `Internal`/`FailedPrecondition` are the codes nereid reserves for
    /// model/backend bugs (a wrong-shaped or wrong-typed output, a crashed
    /// worker). Everything else — `InvalidArgument` above all — is a bad
    /// request.
    fn from_status(status: &Status) -> Self {
        match status.code() {
            Code::ResourceExhausted => FailureReason::Rejected,
            Code::Cancelled => FailureReason::Canceled,
            Code::Internal | Code::FailedPrecondition => FailureReason::Backend,
            _ => FailureReason::Other,
        }
    }
}

/// One model's counters. Field names track the metric names minus the
/// `nv_inference_` prefix.
#[derive(Default)]
struct ModelMetrics {
    request_success: AtomicU64,
    /// Indexed by `FailureReason::ALL` position.
    request_failure: [AtomicU64; 4],
    count: AtomicU64,
    exec_count: AtomicU64,
    request_duration_us: AtomicU64,
    queue_duration_us: AtomicU64,
    compute_input_duration_us: AtomicU64,
    compute_infer_duration_us: AtomicU64,
    compute_output_duration_us: AtomicU64,
    pending_request_count: AtomicU64,
}

/// The registry: one [`ModelMetrics`] per configured model, in config order.
pub struct InferenceMetrics {
    models: Vec<(String, Arc<ModelMetrics>)>,
    by_name: HashMap<String, Arc<ModelMetrics>>,
}

impl InferenceMetrics {
    /// Build the registry for a fixed set of models. Every series is emitted
    /// from the start (at zero), as Triton does, so `rate()` queries have a
    /// baseline before the first request.
    pub fn new<I, S>(model_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut models = Vec::new();
        let mut by_name = HashMap::new();
        for name in model_names {
            let name: String = name.into();
            let metrics = Arc::new(ModelMetrics::default());
            by_name.insert(name.clone(), metrics.clone());
            models.push((name, metrics));
        }
        Self { models, by_name }
    }

    /// Open the accounting for one request to `model_name`, or `None` when the
    /// model is not registered. The model's pending gauge goes up until the
    /// request executes or ends.
    pub fn begin(&self, model_name: &str) -> Option<RequestTimer> {
        let model = self.by_name.get(model_name)?.clone();
        model.pending_request_count.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        Some(RequestTimer {
            model,
            started: now,
            phase: now,
            input_us: 0,
            queue_us: 0,
            infer_us: 0,
            executions: 0,
            inferences: 0,
            pending: true,
            reason: None,
            finished: false,
        })
    }

    /// Render every series in the Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.family(
            &mut out,
            "nv_inference_request_success",
            "counter",
            "Number of successful inference requests, all batch sizes",
            |m| m.request_success.load(Ordering::Relaxed),
        );

        // The failure counter carries an extra `reason` label, so it is laid
        // out by hand: every reason for every model, zero or not.
        let _ = writeln!(
            out,
            "# HELP nv_inference_request_failure Number of failed inference requests, all batch sizes"
        );
        let _ = writeln!(out, "# TYPE nv_inference_request_failure counter");
        for (model, metrics) in &self.models {
            for (i, reason) in FailureReason::ALL.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "nv_inference_request_failure{{model=\"{}\",version=\"{MODEL_VERSION}\",reason=\"{}\"}} {}",
                    escape_label(model),
                    reason.label(),
                    metrics.request_failure[i].load(Ordering::Relaxed)
                );
            }
        }

        self.family(
            &mut out,
            "nv_inference_count",
            "counter",
            "Number of inferences performed (a batch of \"n\" is counted as \"n\" inferences)",
            |m| m.count.load(Ordering::Relaxed),
        );
        self.family(
            &mut out,
            "nv_inference_exec_count",
            "counter",
            "Number of model executions performed",
            |m| m.exec_count.load(Ordering::Relaxed),
        );
        self.family(
            &mut out,
            "nv_inference_request_duration_us",
            "counter",
            "Cumulative inference request duration in microseconds",
            |m| m.request_duration_us.load(Ordering::Relaxed),
        );
        self.family(
            &mut out,
            "nv_inference_queue_duration_us",
            "counter",
            "Cumulative inference queuing duration in microseconds",
            |m| m.queue_duration_us.load(Ordering::Relaxed),
        );
        self.family(
            &mut out,
            "nv_inference_compute_input_duration_us",
            "counter",
            "Cumulative compute input duration in microseconds",
            |m| m.compute_input_duration_us.load(Ordering::Relaxed),
        );
        self.family(
            &mut out,
            "nv_inference_compute_infer_duration_us",
            "counter",
            "Cumulative compute inference duration in microseconds",
            |m| m.compute_infer_duration_us.load(Ordering::Relaxed),
        );
        self.family(
            &mut out,
            "nv_inference_compute_output_duration_us",
            "counter",
            "Cumulative inference compute output duration in microseconds",
            |m| m.compute_output_duration_us.load(Ordering::Relaxed),
        );
        self.family(
            &mut out,
            "nv_inference_pending_request_count",
            "gauge",
            "Instantaneous number of pending requests awaiting execution per-model.",
            |m| m.pending_request_count.load(Ordering::Relaxed),
        );
        out
    }

    /// Append one metric family — HELP, TYPE, then a `{model, version}` sample
    /// per model — reading each model's value with `value`.
    fn family(
        &self,
        out: &mut String,
        name: &str,
        kind: &str,
        help: &str,
        value: impl Fn(&ModelMetrics) -> u64,
    ) {
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} {kind}");
        for (model, metrics) in &self.models {
            let _ = writeln!(
                out,
                "{name}{{model=\"{}\",version=\"{MODEL_VERSION}\"}} {}",
                escape_label(model),
                value(metrics)
            );
        }
    }
}

/// Escape a label value per the text exposition format: backslash, double
/// quote, and newline. Model names are folder names, so this rarely fires, but
/// a name containing `"` must not corrupt the scrape.
fn escape_label(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            other => escaped.push(other),
        }
    }
    escaped
}

fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

/// The in-flight accounting for one inference request. See the module docs
/// for the phase model. Must end with [`Self::succeed`] or [`Self::fail`];
/// dropping it unfinished records a `CANCELED` failure.
pub struct RequestTimer {
    model: Arc<ModelMetrics>,
    started: Instant,
    /// Start of the current phase (input → queue/infer → output).
    phase: Instant,
    input_us: u64,
    queue_us: u64,
    infer_us: u64,
    executions: u64,
    inferences: u64,
    /// Whether this request still counts toward the pending gauge.
    pending: bool,
    /// A failure reason pinned by whoever observed the failure (e.g. the
    /// `ModelManager` on a backend error); overrides status-code classification.
    reason: Option<FailureReason>,
    finished: bool,
}

impl RequestTimer {
    /// The input tensors are built and validated: end of the `compute_input`
    /// phase.
    pub fn input_ready(&mut self) {
        let now = Instant::now();
        self.input_us += micros(now.duration_since(self.phase));
        self.phase = now;
    }

    /// Time elapsed in the current phase — for a backend that runs the whole
    /// execution as one opaque step (no separate queue/infer split).
    pub fn phase_elapsed(&self) -> Duration {
        self.phase.elapsed()
    }

    /// The backend ran: `queue` is how long the request waited for a worker,
    /// `infer` how long the backend's `infer` took, `batch` the number of
    /// inferences in this execution (the request's batch size, or 1). Ends the
    /// pending state and starts the `compute_output` phase.
    pub fn executed(&mut self, queue: Duration, infer: Duration, batch: u64) {
        self.queue_us += micros(queue);
        self.infer_us += micros(infer);
        self.executions += 1;
        self.inferences += batch.max(1);
        self.clear_pending();
        self.phase = Instant::now();
    }

    /// Pin the failure reason to `BACKEND`: the model itself failed, whatever
    /// status code the error carries.
    pub fn backend_failed(&mut self) {
        self.reason = Some(FailureReason::Backend);
    }

    /// The request completed. Commits the request's counts and durations; the
    /// time since the last phase mark is `compute_output`.
    pub fn succeed(mut self) {
        let now = Instant::now();
        let output_us = micros(now.duration_since(self.phase));
        let request_us = micros(now.duration_since(self.started));
        let m = &self.model;
        m.request_success.fetch_add(1, Ordering::Relaxed);
        m.count.fetch_add(self.inferences, Ordering::Relaxed);
        m.exec_count.fetch_add(self.executions, Ordering::Relaxed);
        m.request_duration_us
            .fetch_add(request_us, Ordering::Relaxed);
        m.queue_duration_us
            .fetch_add(self.queue_us, Ordering::Relaxed);
        m.compute_input_duration_us
            .fetch_add(self.input_us, Ordering::Relaxed);
        m.compute_infer_duration_us
            .fetch_add(self.infer_us, Ordering::Relaxed);
        m.compute_output_duration_us
            .fetch_add(output_us, Ordering::Relaxed);
        self.close();
    }

    /// The request failed with `status`. Counted once under the pinned reason
    /// if there is one, else the reason implied by the status code.
    pub fn fail(mut self, status: &Status) {
        let reason = self
            .reason
            .unwrap_or_else(|| FailureReason::from_status(status));
        self.record_failure(reason);
        self.close();
    }

    fn record_failure(&self, reason: FailureReason) {
        let index = FailureReason::ALL
            .iter()
            .position(|r| *r == reason)
            .expect("every reason is listed in ALL");
        self.model.request_failure[index].fetch_add(1, Ordering::Relaxed);
    }

    fn clear_pending(&mut self) {
        if self.pending {
            self.pending = false;
            self.model
                .pending_request_count
                .fetch_sub(1, Ordering::Relaxed);
        }
    }

    fn close(&mut self) {
        self.clear_pending();
        self.finished = true;
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        if !self.finished {
            // The handler's future was dropped mid-request: the client hung up
            // or the call was cancelled before a response could be recorded.
            // That is a cancellation whatever was observed before the drop —
            // a pinned reason is only ever committed by an explicit `fail`.
            self.record_failure(FailureReason::Canceled);
            self.close();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value of the series `name{labels}` in rendered text.
    fn series(text: &str, name: &str, labels: &str) -> Option<u64> {
        let prefix = format!("{name}{{{labels}}} ");
        text.lines()
            .find_map(|line| line.strip_prefix(&prefix))
            .map(|v| v.parse().expect("integer sample"))
    }

    const M3: &str = "model=\"model3\",version=\"1\"";

    #[test]
    fn renders_every_series_at_zero_for_each_model() {
        let metrics = InferenceMetrics::new(["model3", "pymul"]);
        let text = metrics.render();
        for name in [
            "nv_inference_request_success",
            "nv_inference_count",
            "nv_inference_exec_count",
            "nv_inference_request_duration_us",
            "nv_inference_queue_duration_us",
            "nv_inference_compute_input_duration_us",
            "nv_inference_compute_infer_duration_us",
            "nv_inference_compute_output_duration_us",
            "nv_inference_pending_request_count",
        ] {
            assert!(
                text.contains(&format!("# TYPE {name} ")),
                "missing TYPE line for {name}"
            );
            assert_eq!(series(&text, name, M3), Some(0), "{name} for model3");
            assert_eq!(
                series(&text, name, "model=\"pymul\",version=\"1\""),
                Some(0),
                "{name} for pymul"
            );
        }
        assert!(text.contains("# TYPE nv_inference_request_failure counter"));
        for reason in ["REJECTED", "CANCELED", "BACKEND", "OTHER"] {
            assert_eq!(
                series(
                    &text,
                    "nv_inference_request_failure",
                    &format!("{M3},reason=\"{reason}\"")
                ),
                Some(0),
                "failure reason {reason} must be pre-registered"
            );
        }
        assert!(text.contains("# TYPE nv_inference_pending_request_count gauge"));
    }

    #[test]
    fn success_commits_counts_and_durations() {
        let metrics = InferenceMetrics::new(["model3"]);
        let mut timer = metrics.begin("model3").expect("registered model");
        assert_eq!(
            series(&metrics.render(), "nv_inference_pending_request_count", M3),
            Some(1),
            "pending while in flight"
        );
        timer.input_ready();
        timer.executed(Duration::from_micros(7), Duration::from_micros(1500), 4);
        assert_eq!(
            series(&metrics.render(), "nv_inference_pending_request_count", M3),
            Some(0),
            "no longer pending once executing"
        );
        timer.succeed();

        let text = metrics.render();
        assert_eq!(series(&text, "nv_inference_request_success", M3), Some(1));
        assert_eq!(
            series(&text, "nv_inference_count", M3),
            Some(4),
            "batch of 4"
        );
        assert_eq!(series(&text, "nv_inference_exec_count", M3), Some(1));
        assert_eq!(series(&text, "nv_inference_queue_duration_us", M3), Some(7));
        assert_eq!(
            series(&text, "nv_inference_compute_infer_duration_us", M3),
            Some(1500)
        );
        assert!(
            series(&text, "nv_inference_compute_output_duration_us", M3).is_some(),
            "output phase is recorded"
        );
        for reason in ["REJECTED", "CANCELED", "BACKEND", "OTHER"] {
            assert_eq!(
                series(
                    &text,
                    "nv_inference_request_failure",
                    &format!("{M3},reason=\"{reason}\"")
                ),
                Some(0)
            );
        }
    }

    #[test]
    fn failures_are_classified_by_reason() {
        let metrics = InferenceMetrics::new(["model3"]);
        let failure = |reason: &str| {
            series(
                &metrics.render(),
                "nv_inference_request_failure",
                &format!("{M3},reason=\"{reason}\""),
            )
        };

        metrics
            .begin("model3")
            .unwrap()
            .fail(&Status::invalid_argument("bad shape"));
        assert_eq!(failure("OTHER"), Some(1));

        metrics
            .begin("model3")
            .unwrap()
            .fail(&Status::resource_exhausted("queue full"));
        assert_eq!(failure("REJECTED"), Some(1));

        metrics
            .begin("model3")
            .unwrap()
            .fail(&Status::internal("model returned 2 outputs"));
        assert_eq!(failure("BACKEND"), Some(1));

        // A pinned reason wins over the status code.
        let mut timer = metrics.begin("model3").unwrap();
        timer.backend_failed();
        timer.fail(&Status::invalid_argument("backend rejected dtype"));
        assert_eq!(failure("BACKEND"), Some(2));

        // Dropping an unfinished timer is a cancellation — even when a reason
        // was pinned: the pin only takes effect through an explicit `fail`.
        drop(metrics.begin("model3").unwrap());
        assert_eq!(failure("CANCELED"), Some(1));
        let mut pinned_then_dropped = metrics.begin("model3").unwrap();
        pinned_then_dropped.backend_failed();
        drop(pinned_then_dropped);
        assert_eq!(failure("CANCELED"), Some(2));
        assert_eq!(failure("BACKEND"), Some(2), "the pin was not committed");

        let text = metrics.render();
        assert_eq!(series(&text, "nv_inference_request_success", M3), Some(0));
        assert_eq!(series(&text, "nv_inference_count", M3), Some(0));
        assert_eq!(
            series(&text, "nv_inference_pending_request_count", M3),
            Some(0),
            "every failure path must release its pending slot"
        );
    }

    #[test]
    fn unknown_model_has_no_timer() {
        let metrics = InferenceMetrics::new(["model3"]);
        assert!(metrics.begin("ghost").is_none());
    }

    #[test]
    fn label_values_are_escaped() {
        let metrics = InferenceMetrics::new(["we\"ird\\name"]);
        let text = metrics.render();
        assert!(
            text.contains(
                "nv_inference_request_success{model=\"we\\\"ird\\\\name\",version=\"1\"} 0"
            ),
            "{text}"
        );
    }
}
