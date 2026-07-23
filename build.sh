#!/usr/bin/env bash
#
# build.sh — portable build driver for nereid-server.
#
# nereid-server links libtorch through `tch`/`torch-sys`. Getting a working
# build therefore comes down to answering three questions: which libtorch, how
# is it linked, and how does the binary find it at runtime. This script answers
# all three so `cargo build` doesn't leave you to.
#
# It grew out of a cluster-specific recipe (Purdue's Gilbreth: `module load
# cuda`, point LIBTORCH at a shared install, build) and generalizes it — the
# same script drives a laptop CPU build, an HPC CUDA node, a self-contained
# release tarball, or a fully static binary.
#
# Linking modes (--link):
#   dynamic  (default)  Ordinary build. Finds the libtorch it linked and writes
#                       a run wrapper that sets LD_LIBRARY_PATH for you.
#   bundled             Copies libtorch's .so's next to the binary and sets an
#                       $ORIGIN rpath, so it runs with NO LD_LIBRARY_PATH and
#                       relocates as a unit — good for containers and tarballs.
#   static              Statically links libtorch (LIBTORCH_STATIC=1). Needs a
#                       static-lib libtorch; build one with --build-libtorch.
#
# libtorch sources, in precedence order:
#   --build-libtorch    build a static libtorch from source (scripts/build-libtorch.sh)
#   --fetch-libtorch    download the official libtorch and verify its sha256
#   --libtorch / $LIBTORCH   use an existing install (the HPC path)
#   (default)           let `tch` download a CPU libtorch itself
#
# See --help for the full option list.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_NAME="grpc-test"           # cargo package name -> target/<profile>/grpc-test

LINK_MODE="dynamic"            # dynamic | bundled | static
LINK_EXPLICIT=0                # was --link passed on the command line?
PROFILE="debug"               # debug | release
DEVICE="cpu"                  # cpu | cuda | cuda:<ver> (e.g. cuda:cu124)
LIBTORCH_PATH="${LIBTORCH:-}"  # external libtorch dir (overrides download)
OUT_DIR=""                    # bundled output dir (default: dist/<binary>)
JOBS=""                       # cargo -j value (empty = cargo default)
DO_RUN=0                       # run the server after building
declare -a EXTRA_CARGO_ARGS=() # passthrough args after `--`
declare -a FEATURES=()         # extra cargo features added on top of the selection
BACKENDS=""                    # exact backend set (implies --no-default-features)

# libtorch source resolution
LIBTORCH_VERSION="2.5.1"       # the version tch 0.18.x links against
FETCH_LIBTORCH=0               # download + hash-check libtorch ourselves
BUILD_LIBTORCH=0               # build a static libtorch from source (implies static)
LIBTORCH_SHA256_OVERRIDE=""    # override the pinned sha256 (for un-pinned combos)
NO_VERIFY=0                    # skip the sha256 check (not recommended)
LIBTORCH_CACHE="${LIBTORCH_CACHE:-$REPO_ROOT/.cache/libtorch}"

# Environment managers / HPC modules (applied before building)
declare -a MODULES=()          # `module load` specs (Lmod / environment-modules)
CONDA_ENV=""                  # `conda activate <env>`
PYENV_VERSION=""              # `pyenv shell <version>`
VENV_PATH=""                  # `source <path>/bin/activate`

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: ./build.sh [options] [-- <extra cargo args>]

Build nereid-server, taking care of the libtorch dependency.

Linking:
  --link <mode>       dynamic (default) | bundled | static.
                        dynamic  ordinary build; writes an LD_LIBRARY_PATH wrapper.
                        bundled  copy libtorch .so's beside the binary, rpath=$ORIGIN;
                                 runs with no env setup and relocates as a unit.
                        static   static link (LIBTORCH_STATIC=1); needs a static libtorch.
  --release           Build the release profile (default: debug).
  --debug             Build the debug profile (explicit default).
  --out <dir>         Output dir for --link bundled (default: dist/<binary>).

libtorch source (highest precedence first):
  --build-libtorch    Build a static libtorch from source (scripts/build-libtorch.sh),
                        then static-link against it. Long; see that script's --help.
  --fetch-libtorch    Download the official libtorch and verify its sha256, instead
                        of letting `tch` download it opaquely. Feeds dynamic/bundled.
  --libtorch <dir>    Use an existing libtorch at <dir> (same as exporting LIBTORCH).
  --libtorch-version <v>   libtorch version to fetch/build (default: 2.5.1).
  --libtorch-sha256 <hex>  Expected sha256 for --fetch-libtorch (un-pinned combos).
  --no-verify         Skip the sha256 check for --fetch-libtorch (not recommended).

Backend selection (build only the backends you want):
  --backends <csv>    Exact backend set, e.g. --backends onnx (turns OFF the
                        default torch+python). Valid: torch, python, onnx,
                        tensorflow, tensorflow-gpu, cxx, cpp. An
                        ONNX/TF/cxx/cpp-only build links no libtorch at all.
  --onnx              Add the ONNX backend (ort / ONNX Runtime, CUDA-capable).
  --tensorflow        Add the TensorFlow backend (libtensorflow SavedModel).
  --tensorflow-gpu    Add TensorFlow linked against the libtensorflow GPU build.
  --cxx               Add compile-time C++ models (cxx-bridged, linked in; needs
                        a C++ compiler at build time).
  --cpp               Add the C++ subprocess backend (compiles a model's
                        main.cpp at startup; needs a C++ compiler at runtime).
  --features <list>   Add an arbitrary comma-separated cargo feature list.
  # With no --backends, the default torch+python backends stay on and --onnx /
  # --tensorflow add to them. bundled builds bundle whichever runtimes are linked.
  #
  # $NEREID_BACKENDS (or a backends.conf beside Cargo.toml) independently selects
  # which discovered backend folders are compiled in, by name pattern — the way
  # to include or exclude an out-of-tree backend that has no Cargo feature.
  # e.g. NEREID_BACKENDS='!torch' ./build.sh    (see the README)

Device:
  --device <dev>      cpu (default), cuda, or cuda:<cuXYZ>. The version is the
                        compact TORCH_CUDA_VERSION form, e.g. --device cuda:cu124
                        (not cuda:12.4). Bare 'cuda' defaults to cu124. Selects the
                        CUDA libtorch download / TORCH_CUDA_VERSION.

Environment (applied before building; useful on HPC / managed Python):
  --module <spec>     `module load <spec>` (repeatable), e.g. --module cuda/12.6.0.
  --conda <env>       `conda activate <env>` before building.
  --pyenv <version>   `pyenv shell <version>` before building.
  --venv <dir>        `source <dir>/bin/activate` before building.

Other:
  -j, --jobs <n>      Pass -j <n> to cargo.
  --run               Run the server after a successful build.
  -h, --help          Show this help.

Examples:
  ./build.sh                                   # plain dynamic build
  ./build.sh --release --link bundled          # self-contained release
  ./build.sh --release --link bundled --fetch-libtorch   # + verified libtorch
  ./build.sh --module cuda/12.6.0 --libtorch /depot/.../libtorch --device cuda
  ./build.sh --release --link static --build-libtorch     # static, from source
  ./build.sh --conda nereid --release          # build inside a conda env
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --link)             LINK_MODE="${2:?--link needs a value}"; LINK_EXPLICIT=1; shift 2 ;;
    --link=*)           LINK_MODE="${1#*=}"; LINK_EXPLICIT=1; shift ;;
    --backends)         BACKENDS="${2:?--backends needs a value}"; shift 2 ;;
    --backends=*)       BACKENDS="${1#*=}"; shift ;;
    --onnx)             FEATURES+=("onnx"); shift ;;
    --tensorflow)       FEATURES+=("tensorflow"); shift ;;
    --tensorflow-gpu)   FEATURES+=("tensorflow-gpu"); shift ;;
    --cxx)              FEATURES+=("cxx"); shift ;;
    --cpp)              FEATURES+=("cpp"); shift ;;
    --features)         FEATURES+=("${2:?--features needs a value}"); shift 2 ;;
    --features=*)       FEATURES+=("${1#*=}"); shift ;;
    --release)          PROFILE="release"; shift ;;
    --debug)            PROFILE="debug"; shift ;;
    --device)           DEVICE="${2:?--device needs a value}"; shift 2 ;;
    --device=*)         DEVICE="${1#*=}"; shift ;;
    --libtorch)         LIBTORCH_PATH="${2:?--libtorch needs a path}"; shift 2 ;;
    --libtorch=*)       LIBTORCH_PATH="${1#*=}"; shift ;;
    --libtorch-version) LIBTORCH_VERSION="${2:?}"; shift 2 ;;
    --libtorch-version=*) LIBTORCH_VERSION="${1#*=}"; shift ;;
    --libtorch-sha256)  LIBTORCH_SHA256_OVERRIDE="${2:?}"; shift 2 ;;
    --libtorch-sha256=*) LIBTORCH_SHA256_OVERRIDE="${1#*=}"; shift ;;
    --fetch-libtorch)   FETCH_LIBTORCH=1; shift ;;
    --build-libtorch)   BUILD_LIBTORCH=1; shift ;;
    --no-verify)        NO_VERIFY=1; shift ;;
    --out)              OUT_DIR="${2:?--out needs a path}"; shift 2 ;;
    --out=*)            OUT_DIR="${1#*=}"; shift ;;
    --module)           MODULES+=("${2:?--module needs a spec}"); shift 2 ;;
    --module=*)         MODULES+=("${1#*=}"); shift ;;
    --conda)            CONDA_ENV="${2:?--conda needs an env}"; shift 2 ;;
    --conda=*)          CONDA_ENV="${1#*=}"; shift ;;
    --pyenv)            PYENV_VERSION="${2:?--pyenv needs a version}"; shift 2 ;;
    --pyenv=*)          PYENV_VERSION="${1#*=}"; shift ;;
    --venv)             VENV_PATH="${2:?--venv needs a path}"; shift 2 ;;
    --venv=*)           VENV_PATH="${1#*=}"; shift ;;
    -j|--jobs)          JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --jobs=*)           JOBS="${1#*=}"; shift ;;
    --run)              DO_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    --)                 shift; EXTRA_CARGO_ARGS+=("$@"); break ;;
    *)                  die "unknown option: $1 (see --help)" ;;
  esac
done

# --build-libtorch produces a static libtorch, which can only be static-linked.
# Default --link to static for it; but if the user explicitly asked for a
# different mode, that's a contradiction — fail loudly rather than silently
# overriding their choice.
if [[ $BUILD_LIBTORCH -eq 1 ]]; then
  if [[ $LINK_EXPLICIT -eq 1 && "$LINK_MODE" != "static" ]]; then
    die "--build-libtorch produces a static libtorch, which cannot be linked '$LINK_MODE'.
    Use '--link static' (or drop --link to default to it), or drop --build-libtorch
    and use --fetch-libtorch for a $LINK_MODE build."
  fi
  LINK_MODE="static"
fi

case "$LINK_MODE" in
  dynamic|bundled|static) ;;
  *) die "invalid --link mode: '$LINK_MODE' (want dynamic|bundled|static)" ;;
esac

# ---------------------------------------------------------------------------
# Device -> TORCH_CUDA_VERSION + download subdir
# ---------------------------------------------------------------------------
declare -a BUILD_ENV=()
DEVICE_SUBDIR="cpu"            # download.pytorch.org/libtorch/<subdir>
case "$DEVICE" in
  cpu)    ;;
  cuda)   DEVICE_SUBDIR="cu124"; BUILD_ENV+=("TORCH_CUDA_VERSION=cu124") ;;
  cuda:*) cuver="${DEVICE#cuda:}"
          # tch's TORCH_CUDA_VERSION and PyTorch's download URLs both use the
          # compact "cuXYZ" form (e.g. cu124), not a dotted "12.4".
          [[ "$cuver" =~ ^cu[0-9]+$ ]] || die \
            "invalid --device '$DEVICE': the CUDA version must be the compact 'cuXYZ' form,
    e.g. --device cuda:cu124 (not cuda:12.4). This matches TORCH_CUDA_VERSION."
          DEVICE_SUBDIR="$cuver"; BUILD_ENV+=("TORCH_CUDA_VERSION=$cuver") ;;
  *)      die "invalid --device: '$DEVICE' (want cpu | cuda | cuda:cuXYZ, e.g. cuda:cu124)" ;;
esac

# ---------------------------------------------------------------------------
# Environment managers (HPC modules, conda, pyenv, venv) — applied to THIS
# shell so the cargo build (and any python codegen) inherits them.
# ---------------------------------------------------------------------------
activate_environment() {
  if [[ ${#MODULES[@]} -gt 0 ]]; then
    if type module >/dev/null 2>&1; then
      for m in "${MODULES[@]}"; do log "module load $m"; module load "$m"; done
    else
      warn "'module' command not found; skipping --module (${MODULES[*]}). On HPC, run this from a login shell."
    fi
  fi

  if [[ -n "$VENV_PATH" ]]; then
    [[ -f "$VENV_PATH/bin/activate" ]] || die "--venv: no activate script at $VENV_PATH/bin/activate"
    log "activating venv: $VENV_PATH"
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
  fi

  if [[ -n "$PYENV_VERSION" ]]; then
    command -v pyenv >/dev/null 2>&1 || die "--pyenv given but 'pyenv' is not on PATH"
    log "pyenv shell $PYENV_VERSION"
    eval "$(pyenv init -)"
    pyenv shell "$PYENV_VERSION"
  fi

  if [[ -n "$CONDA_ENV" ]]; then
    command -v conda >/dev/null 2>&1 || die "--conda given but 'conda' is not on PATH"
    log "conda activate $CONDA_ENV"
    # Load conda's shell function, then activate — works in non-interactive shells.
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  fi
}

activate_environment

# ---------------------------------------------------------------------------
# Pinned libtorch checksums (filename -> sha256). Extend as combos are verified.
# Fill a new one in by running: ./build.sh --fetch-libtorch --no-verify ...
# then sha256sum the cached zip and paste it here.
# ---------------------------------------------------------------------------
libtorch_pinned_sha256() {
  case "$1" in
    libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip)
      echo "618ca54eef82a1dca46ff1993d5807d9c0deb0bae147da4974166a147cb562fa" ;;
    *) echo "" ;;
  esac
}

# Download + verify + extract the official libtorch. Echoes the libtorch dir.
fetch_libtorch() {
  local subdir="$1" ver="$2"
  command -v curl >/dev/null 2>&1 || die "curl is required for --fetch-libtorch"
  command -v unzip >/dev/null 2>&1 || die "unzip is required for --fetch-libtorch"

  local fname="libtorch-cxx11-abi-shared-with-deps-${ver}+${subdir}.zip"
  local url="https://download.pytorch.org/libtorch/${subdir}/${fname/+/%2B}"
  local dest="$LIBTORCH_CACHE/${subdir}-${ver}"

  # Already extracted?
  if [[ -d "$dest/libtorch/lib" ]]; then
    log "using cached libtorch: $dest/libtorch"
    printf '%s\n' "$dest/libtorch"; return 0
  fi

  local want="$LIBTORCH_SHA256_OVERRIDE"
  [[ -z "$want" ]] && want="$(libtorch_pinned_sha256 "$fname")"
  if [[ -z "$want" && "$NO_VERIFY" -ne 1 ]]; then
    die "no pinned sha256 for $fname.
    Pass --libtorch-sha256 <hex> to pin it, or --no-verify to skip (not recommended)."
  fi

  mkdir -p "$LIBTORCH_CACHE"
  local zip="$LIBTORCH_CACHE/$fname"
  log "downloading libtorch: $url"
  curl -fSL --retry 3 -o "$zip" "$url" || die "download failed: $url"

  if [[ "$NO_VERIFY" -ne 1 ]]; then
    local got; got="$(sha256sum "$zip" | cut -d' ' -f1)"
    if [[ "$got" != "$want" ]]; then
      rm -f "$zip"
      die "sha256 mismatch for $fname
    expected: $want
    got:      $got"
    fi
    log "sha256 verified: $got"
  else
    warn "skipping sha256 verification (--no-verify)"
  fi

  log "extracting into $dest"
  mkdir -p "$dest"
  unzip -q -o "$zip" -d "$dest"
  [[ -d "$dest/libtorch/lib" ]] || die "unexpected libtorch archive layout under $dest"
  printf '%s\n' "$dest/libtorch"
}

# ---------------------------------------------------------------------------
# Resolve which libtorch to build against.
# ---------------------------------------------------------------------------
if [[ $BUILD_LIBTORCH -eq 1 ]]; then
  [[ -z "$LIBTORCH_PATH" ]] || warn "--build-libtorch overrides --libtorch ($LIBTORCH_PATH)"
  builder="$REPO_ROOT/scripts/build-libtorch.sh"
  [[ -x "$builder" ]] || die "missing helper: $builder"
  static_out="$REPO_ROOT/.cache/libtorch-static/${DEVICE_SUBDIR}-${LIBTORCH_VERSION}"
  declare -a builder_args=(--version "$LIBTORCH_VERSION" --out "$static_out" --device "$DEVICE")
  [[ -n "$JOBS" ]] && builder_args+=(--jobs "$JOBS")
  log "building static libtorch from source (this takes a while)..."
  "$builder" "${builder_args[@]}"
  LIBTORCH_PATH="$static_out"
elif [[ $FETCH_LIBTORCH -eq 1 && -z "$LIBTORCH_PATH" ]]; then
  LIBTORCH_PATH="$(fetch_libtorch "$DEVICE_SUBDIR" "$LIBTORCH_VERSION")"
fi

# External / fetched / built libtorch overrides tch's own download.
if [[ -n "$LIBTORCH_PATH" ]]; then
  [[ -d "$LIBTORCH_PATH" ]] || die "libtorch path does not exist: $LIBTORCH_PATH"
  LIBTORCH_PATH="$(cd "$LIBTORCH_PATH" && pwd)"
  BUILD_ENV+=("LIBTORCH=$LIBTORCH_PATH")
fi

# Static link needs the archive files, not just the .so's.
if [[ "$LINK_MODE" == "static" ]]; then
  [[ -n "$LIBTORCH_PATH" ]] || die \
    "--link static requires a static libtorch. Use --build-libtorch, or pass --libtorch <dir>
    pointing at a static build. The official/downloaded libtorch is shared-only."
  if ! ls "$LIBTORCH_PATH"/lib/libtorch.a >/dev/null 2>&1; then
    die "no static libtorch archives under $LIBTORCH_PATH/lib (expected libtorch.a, libc10.a, ...).
    That looks like a shared-only libtorch. Build a static one with: ./build.sh --build-libtorch"
  fi
  BUILD_ENV+=("LIBTORCH_STATIC=1")
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
# Resolve the final cargo feature selection. `--backends <csv>` picks an exact
# set (turning off the torch+python defaults); `--onnx`/`--tensorflow` add on top
# of whatever's selected. With neither, the default backends (torch+python) apply.
declare -a CARGO_ARGS=(build)
[[ "$PROFILE" == "release" ]] && CARGO_ARGS+=(--release)
[[ -n "$JOBS" ]] && CARGO_ARGS+=(-j "$JOBS")

declare -a SELECTED=()
if [[ -n "$BACKENDS" ]]; then
  IFS=',' read -r -a explicit <<< "$BACKENDS"
  SELECTED+=("${explicit[@]}")
fi
SELECTED+=("${FEATURES[@]}")

# Is the torch (.pt / libtorch) backend part of this build? It's on by default
# unless an explicit --backends set omits it.
if [[ -n "$BACKENDS" ]]; then
  case ",$BACKENDS,${FEATURES[*]}," in *torch*) TORCH_ENABLED=1 ;; *) TORCH_ENABLED=0 ;; esac
else
  TORCH_ENABLED=1
fi

if [[ -n "$BACKENDS" ]]; then
  CARGO_ARGS+=(--no-default-features)
fi
if [[ ${#SELECTED[@]} -gt 0 ]]; then
  feature_csv="$(IFS=,; echo "${SELECTED[*]}")"
  CARGO_ARGS+=(--features "$feature_csv")
fi
CARGO_ARGS+=("${EXTRA_CARGO_ARGS[@]}")

# A non-torch build must not try to fetch/point-at libtorch.
if [[ $TORCH_ENABLED -eq 0 ]]; then
  FETCH_LIBTORCH=0
  LIBTORCH_PATH=""
fi

log "Building nereid-server (profile=$PROFILE, device=$DEVICE, link=$LINK_MODE)"
[[ -n "$BACKENDS" ]] && log "Backends: $BACKENDS${FEATURES[*]:+ (+ ${FEATURES[*]})}"
[[ ${#FEATURES[@]} -gt 0 ]] && log "Features: ${FEATURES[*]}"
[[ ${#BUILD_ENV[@]} -gt 0 ]] && log "Build env: ${BUILD_ENV[*]}"

( cd "$REPO_ROOT" && env "${BUILD_ENV[@]}" cargo "${CARGO_ARGS[@]}" )

BIN_PATH="$REPO_ROOT/target/$PROFILE/$BIN_NAME"
[[ -x "$BIN_PATH" ]] || die "expected binary not found at $BIN_PATH"
log "Built: $BIN_PATH"

# ---------------------------------------------------------------------------
# Locate the libtorch lib/ directory that was linked.
#
# For an external/fetched libtorch it's <LIBTORCH>/lib. For tch's own download,
# it's under target/.../build/torch-sys-*/out/... — a rebuild can leave several
# torch-sys build dirs, so pick the one holding the largest (fully-extracted)
# libtorch_cpu.so, matching the README/CI approach.
# ---------------------------------------------------------------------------
find_libtorch_lib_dir() {
  if [[ -n "$LIBTORCH_PATH" && -d "$LIBTORCH_PATH/lib" ]]; then
    printf '%s\n' "$LIBTORCH_PATH/lib"; return 0
  fi
  local so
  so="$(find "$REPO_ROOT/target" -name 'libtorch_cpu.so' -printf '%s %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)"
  [[ -n "$so" ]] && dirname "$so"
}

# ---------------------------------------------------------------------------
# Post-build per mode
# ---------------------------------------------------------------------------
RUN_CMD=("$BIN_PATH")

case "$LINK_MODE" in

  static)
    if command -v ldd >/dev/null 2>&1; then
      if ldd "$BIN_PATH" 2>/dev/null | grep -qiE 'libtorch|libc10'; then
        warn "binary still lists a libtorch/libc10 shared dependency:"
        ldd "$BIN_PATH" 2>/dev/null | grep -iE 'libtorch|libc10' >&2 || true
        warn "the static link may not have fully taken effect."
      else
        log "No libtorch/libc10 shared dependency — static link looks good."
      fi
    fi
    ;;

  bundled)
    [[ -z "$OUT_DIR" ]] && OUT_DIR="$REPO_ROOT/dist/$BIN_NAME"
    mkdir -p "$OUT_DIR/lib"
    copied=0

    # libtorch's shared objects (only when the torch backend is in the build).
    if [[ $TORCH_ENABLED -eq 1 ]]; then
      lib_dir="$(find_libtorch_lib_dir || true)"
      [[ -n "$lib_dir" && -d "$lib_dir" ]] \
        || die "could not locate the libtorch lib/ directory to bundle from."
      log "Bundling libtorch shared objects from: $lib_dir"
      shopt -s nullglob
      for so in "$lib_dir"/*.so "$lib_dir"/*.so.*; do
        cp -L "$so" "$OUT_DIR/lib/" && copied=$((copied + 1))
      done
      shopt -u nullglob
      [[ $copied -gt 0 ]] || die "found no .so files to bundle under $lib_dir"
    fi

    # Native backend runtimes (ONNX Runtime, libtensorflow) live outside the
    # libtorch dir. Resolve them from the binary's actual load map and bundle
    # them too, so a native-only build is equally self-contained.
    if command -v ldd >/dev/null 2>&1; then
      while read -r native_so; do
        [[ -n "$native_so" && -f "$native_so" ]] || continue
        cp -L "$native_so" "$OUT_DIR/lib/" && copied=$((copied + 1))
        log "Bundled native runtime: $(basename "$native_so")"
      done < <(ldd "$BIN_PATH" 2>/dev/null \
        | grep -iE 'libonnxruntime|libtensorflow|libtensorflow_framework' \
        | awk '{print $3}')
    fi

    cp "$BIN_PATH" "$OUT_DIR/$BIN_NAME"
    log "Copied $copied shared object(s) and the binary into $OUT_DIR"

    if command -v patchelf >/dev/null 2>&1; then
      patchelf --set-rpath '$ORIGIN/lib' "$OUT_DIR/$BIN_NAME"
      log "Patched rpath to \$ORIGIN/lib — the bundle is relocatable."
      if command -v ldd >/dev/null 2>&1 \
         && ldd "$OUT_DIR/$BIN_NAME" 2>/dev/null | grep -qi 'libtorch_cpu.so.*not found'; then
        warn "libtorch_cpu.so still not resolved after rpath patch; check the bundle."
      fi
      RUN_CMD=("$OUT_DIR/$BIN_NAME")
    else
      warn "patchelf not found — writing a run wrapper that sets LD_LIBRARY_PATH instead."
      warn "install patchelf for a truly env-free binary (e.g. apt-get install patchelf)."
      cat > "$OUT_DIR/run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
here="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="\$here/lib:\${LD_LIBRARY_PATH:-}"
exec "\$here/$BIN_NAME" "\$@"
EOF
      chmod +x "$OUT_DIR/run.sh"
      RUN_CMD=("$OUT_DIR/run.sh")
    fi
    log "Self-contained build ready in: $OUT_DIR"
    ;;

  dynamic)
    # Only torch builds need the libtorch loader-path wrapper. Native-only builds
    # (ONNX/TensorFlow) resolve their runtimes without it.
    lib_dir=""
    [[ $TORCH_ENABLED -eq 1 ]] && lib_dir="$(find_libtorch_lib_dir || true)"
    if [[ $TORCH_ENABLED -eq 0 ]]; then
      log "No torch backend in this build — the binary needs no libtorch loader path."
      log "Run it directly: $BIN_PATH"
    elif [[ -n "$lib_dir" && -d "$lib_dir" ]]; then
      wrapper="$REPO_ROOT/target/$PROFILE/run-$BIN_NAME.sh"
      cat > "$wrapper" <<EOF
#!/usr/bin/env bash
# Auto-generated by build.sh — runs the dynamically linked binary with libtorch
# on the loader path.
set -euo pipefail
export LD_LIBRARY_PATH="$lib_dir:\${LD_LIBRARY_PATH:-}"
exec "$BIN_PATH" "\$@"
EOF
      chmod +x "$wrapper"
      log "libtorch lib dir: $lib_dir"
      log "Run wrapper written: $wrapper"
      {
        echo
        echo "To run the server:"
        echo "    $wrapper"
        echo "or manually:"
        echo "    export LD_LIBRARY_PATH=\"$lib_dir:\$LD_LIBRARY_PATH\""
        echo "    $BIN_PATH"
      } >&2
      RUN_CMD=("$wrapper")
    else
      warn "could not locate the extracted libtorch lib/ directory."
      warn "you'll need to set LD_LIBRARY_PATH manually before running $BIN_PATH."
    fi
    ;;
esac

# ---------------------------------------------------------------------------
# Optionally run
# ---------------------------------------------------------------------------
if [[ "$DO_RUN" -eq 1 ]]; then
  log "Starting server: ${RUN_CMD[*]}"
  cd "$REPO_ROOT"
  exec "${RUN_CMD[@]}"
fi
