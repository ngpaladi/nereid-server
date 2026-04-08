# Python gRPC Client

Simple client for `inference.Nereid/Checkpoint`.

## Setup

```bash
cd python-client
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (model3 example)

```bash
python client.py \
  --host '[::1]' \
  --port 50051 \
  --model model3 \
  --shape 1,16 \
  --values 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
```

The client:
- connects to gRPC server
- converts input float values to little-endian float32 bytes
- streams request to `Checkpoint`
- collects `output_chunk` bytes from responses
- decodes output back to numeric float values and prints them
