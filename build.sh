#/usr/bin/env bash
set -e

echo "NOTE: Current build script is designed for Purdue Gilbreth cluster."

module load cuda/12.6.0

export LIBTORCH="/depot/cms/private/users/colberte/SONIC/nereid/torch_lib/libtorch"
cargo build

echo "Build complete for grpc-test executable. To run, do:"
echo "    export LD_LIBRARY_PATH=\"$LIBTORCH/lib:\$LD_LIBRARY_PATH\""
echo "    ./target/debug/grpc-test"
