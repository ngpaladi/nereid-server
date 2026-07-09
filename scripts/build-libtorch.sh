#!/usr/bin/env bash
#
# build-libtorch.sh — build a *static* libtorch from source.
#
# The official libtorch distributions ship shared libraries only; PyTorch has
# not published a prebuilt `static-with-deps` archive since 2.1.2. So the only
# way to statically link libtorch at the version `tch` expects (2.5.1) is to
# build it yourself. This script does that: it checks out PyTorch at the tag,
# runs the CMake libtorch build with BUILD_SHARED_LIBS=OFF, and installs a
# clean `include/` + `lib/` tree that `LIBTORCH` can point straight at.
#
# It is invoked automatically by `./build.sh --build-libtorch`, or on its own:
#
#   scripts/build-libtorch.sh --version 2.5.1 --out /opt/libtorch-static
#   LIBTORCH=/opt/libtorch-static LIBTORCH_STATIC=1 cargo build
#
# This is a heavy build: a multi-gigabyte checkout with submodules, a C++
# toolchain, CMake + Ninja, and a Python with PyYAML/typing_extensions for the
# code generators. Expect tens of minutes to well over an hour depending on the
# machine. If you manage Python with conda/pyenv, activate that first (or use
# ./build.sh's --conda / --pyenv, which activate before calling this).

set -euo pipefail

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

VERSION="2.5.1"
DEVICE="cpu"                 # cpu | cuda | cuda:<ver>
OUT_DIR=""
SRC_DIR=""
JOBS=""
KEEP_SRC=0
CXX11_ABI=1                  # tch 0.18.x wants the cxx11 ABI

usage() {
  cat <<'EOF'
Usage: scripts/build-libtorch.sh [options]

Build a static libtorch from source and install it to --out.

Options:
  --version <v>   PyTorch/libtorch version tag to build (default: 2.5.1).
  --device <dev>  cpu (default), cuda, or cuda:<ver>. cuda builds with USE_CUDA=ON.
  --out <dir>     Install prefix (default: .cache/libtorch-static/<device>-<version>).
  --src <dir>     Where to check out PyTorch (default: <out>/../src/pytorch-<version>).
  -j, --jobs <n>  Parallel build jobs (default: nproc).
  --keep-src      Keep the PyTorch checkout after building (default: reuse if present).
  --pre-cxx11     Build the pre-cxx11 ABI (default: cxx11, which tch 0.18 expects).
  -h, --help      Show this help.

Prerequisites: git, cmake (>=3.18), a C++17 compiler, Ninja (recommended), and a
Python 3 with `pip install pyyaml typing_extensions` available on PATH.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)   VERSION="${2:?}"; shift 2 ;;
    --version=*) VERSION="${1#*=}"; shift ;;
    --device)    DEVICE="${2:?}"; shift 2 ;;
    --device=*)  DEVICE="${1#*=}"; shift ;;
    --out)       OUT_DIR="${2:?}"; shift 2 ;;
    --out=*)     OUT_DIR="${1#*=}"; shift ;;
    --src)       SRC_DIR="${2:?}"; shift 2 ;;
    --src=*)     SRC_DIR="${1#*=}"; shift ;;
    -j|--jobs)   JOBS="${2:?}"; shift 2 ;;
    --jobs=*)    JOBS="${1#*=}"; shift ;;
    --keep-src)  KEEP_SRC=1; shift ;;
    --pre-cxx11) CXX11_ABI=0; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)           die "unknown option: $1 (see --help)" ;;
  esac
done

case "$DEVICE" in
  cpu)         USE_CUDA=OFF ;;
  cuda|cuda:*) USE_CUDA=ON ;;
  *)           die "invalid --device: '$DEVICE' (want cpu|cuda|cuda:<ver>)" ;;
esac

# device tag for default paths (cuda:cu124 -> cu124)
DEVICE_TAG="$DEVICE"
[[ "$DEVICE" == cuda:* ]] && DEVICE_TAG="${DEVICE#cuda:}"
[[ "$DEVICE" == "cuda" ]] && DEVICE_TAG="cuda"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -n "$OUT_DIR" ]] || OUT_DIR="$REPO_ROOT/.cache/libtorch-static/${DEVICE_TAG}-${VERSION}"
[[ -n "$SRC_DIR" ]] || SRC_DIR="$REPO_ROOT/.cache/libtorch-src/pytorch-${VERSION}"
[[ -n "$JOBS" ]] || JOBS="$(nproc 2>/dev/null || echo 4)"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

# Already built? Reuse it.
if [[ -f "$OUT_DIR/lib/libtorch.a" ]]; then
  log "static libtorch already present: $OUT_DIR (delete it to rebuild)"
  printf '%s\n' "$OUT_DIR"
  exit 0
fi

# ---------------------------------------------------------------------------
# Prerequisite checks — fail early with an actionable message.
# ---------------------------------------------------------------------------
need() { command -v "$1" >/dev/null 2>&1 || die "missing prerequisite: $1 ($2)"; }
need git    "install git"
need cmake   "install cmake >= 3.18"
need python3 "a Python 3 for PyTorch's code generators"

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import yaml, typing_extensions' >/dev/null 2>&1; then
  die "the selected python ($("$PYTHON" --version 2>&1)) is missing build deps.
    Run: $PYTHON -m pip install pyyaml typing_extensions
    (or activate a suitable env: ./build.sh --conda <env> / --pyenv <ver> --build-libtorch)"
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
else
  warn "ninja not found; falling back to make (slower). Install ninja-build for speed."
fi

if [[ "$USE_CUDA" == "ON" ]] && ! command -v nvcc >/dev/null 2>&1; then
  warn "--device $DEVICE but nvcc is not on PATH; the CUDA build will likely fail."
  warn "load your CUDA toolkit first (e.g. ./build.sh --module cuda/12.6.0 --build-libtorch)."
fi

# ---------------------------------------------------------------------------
# Check out PyTorch at the tag (with submodules).
# ---------------------------------------------------------------------------
if [[ -d "$SRC_DIR/.git" ]]; then
  log "reusing PyTorch checkout: $SRC_DIR"
else
  log "cloning PyTorch v$VERSION (with submodules; this is large)..."
  rm -rf "$SRC_DIR"
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --depth 1 --branch "v$VERSION" --recurse-submodules --shallow-submodules \
    https://github.com/pytorch/pytorch.git "$SRC_DIR" \
    || die "git clone failed (is v$VERSION a valid tag?)"
fi

# ---------------------------------------------------------------------------
# Configure + build + install libtorch (static). CMake install-prefix method
# from PyTorch's docs/libtorch.rst: produces a clean include/ + lib/ tree.
# ---------------------------------------------------------------------------
BUILD_DIR="$SRC_DIR/build-libtorch-static-${DEVICE_TAG}"
mkdir -p "$BUILD_DIR"

log "configuring CMake (static, ABI=$([[ $CXX11_ABI -eq 1 ]] && echo cxx11 || echo pre-cxx11), CUDA=$USE_CUDA)"
cmake "${GENERATOR_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_TEST=OFF \
  -DUSE_DISTRIBUTED=OFF \
  -DUSE_CUDA="$USE_CUDA" \
  -DGLIBCXX_USE_CXX11_ABI="$CXX11_ABI" \
  -DPYTHON_EXECUTABLE="$(command -v "$PYTHON")" \
  -DCMAKE_INSTALL_PREFIX="$OUT_DIR" \
  -S "$SRC_DIR" -B "$BUILD_DIR" \
  || die "cmake configure failed"

log "building + installing with $JOBS jobs (go get a coffee)..."
cmake --build "$BUILD_DIR" --target install -j "$JOBS" || die "libtorch build failed"

# ---------------------------------------------------------------------------
# Verify the static archives landed.
# ---------------------------------------------------------------------------
if [[ ! -f "$OUT_DIR/lib/libtorch.a" ]]; then
  die "build finished but $OUT_DIR/lib/libtorch.a is missing — the static build did not produce archives."
fi
log "static libtorch installed: $OUT_DIR"
log "archives: $(cd "$OUT_DIR/lib" && ls libtorch.a libc10.a 2>/dev/null | tr '\n' ' ')"

if [[ "$KEEP_SRC" -ne 1 ]]; then
  log "removing build tree $BUILD_DIR (pass --keep-src to keep it)"
  rm -rf "$BUILD_DIR"
fi

# The install prefix is what LIBTORCH should point at.
printf '%s\n' "$OUT_DIR"
