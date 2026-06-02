# Python Mock ED Client

The Python client is a YAML-configured mock ED producer runner. It creates random float32 input tensors and sends them to `Nereid/Checkpoint`.

## Install

Run from the repository root:

```bash
python3 -m venv python-client/.venv
source python-client/.venv/bin/activate
pip install -r python-client/requirements.txt
```

## Configure

Create a local config:

```bash
cp python-client/client.yaml.example python-client/client.yaml
```

The local `client.yaml` is ignored by Git.

Basic fixed-shape MLP example:

```yaml
host: "[::1]"
port: 50051
model: "model3"
shape: [1, 16]
producers: 1
inputs_per_producer: 1
chunk_bytes: 65536
sleep_seconds: 0
```

For shape selection, configure exactly one of `shape`, `shapes`, or `random_shape`.

`shape` sends the same full request tensor shape every time:

```yaml
model: "variable-shape-cnn"
shape: [1, 3, 224, 224]
```

`shapes` picks one full request tensor shape per input:

```yaml
model: "variable-shape-cnn"
shapes:
  - [1, 3, 224, 224]
  - [4, 3, 128, 256]
  - [8, 3, 480, 640]
```

`random_shape` generates one full request tensor shape per input. Each dimension can be a fixed positive integer or a `[min, max]` inclusive range:

```yaml
model: "variable-shape-cnn"
random_shape:
  - [1, 8]      # batch
  - 3           # RGB channels
  - [128, 512]  # height
  - [128, 512]  # width
```

The selected shape must match the target model's `model_inference.textproto` contract. For `variable-shape-cnn`, the server-side contract is `input_shape: [3, -1, -1]` plus `max_batch_size: 10`, so request shapes must be `[batch, 3, height, width]` with batch no greater than `10`.

## Run

Run from the repository root:

```bash
python3 python-client/client.py
```

To use a different config file:

```bash
python3 python-client/client.py --config path/to/client.yaml
```

## Behavior

The client:

- starts `producers` separate OS processes
- opens one persistent gRPC channel per producer
- sends `inputs_per_producer` separate tensors per producer
- sends one `Checkpoint` stream per tensor
- logs the configured shape source at startup
- logs the last selected shape on periodic success messages and the selected shape on failures
- does not decode output tensors

