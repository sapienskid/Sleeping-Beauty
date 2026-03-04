#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=scripts/lib/load-dotenv.sh
source "${PROJECT_DIR}/scripts/lib/load-dotenv.sh"
load_dotenv_file "${PROJECT_DIR}/.env"

SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"
REPO_URL="https://github.com/ggml-org/llama.cpp.git"
REPO_DIR="${LLAMA_CPP_REPO_DIR:-/home/sapiens/llama.cpp}"
BUILD_DIR="${LLAMA_CPP_BUILD_DIR:-}"
JOBS="${LLAMA_CPP_BUILD_JOBS:-}"
UPDATE_REPO=true
CLEAN_BUILD=false
MODELS_DIR="${MODELS_DIR:-/home/sapiens/models}"

usage() {
  cat <<'EOF'
Build llama.cpp with CUDA support and produce the llama-server binary.

Usage:
  ./scripts/build-llama-cpp-gpu.sh [options]

Options:
  --repo-dir <path>   llama.cpp repository location (default: LLAMA_CPP_REPO_DIR or /home/sapiens/llama.cpp)
  --build-dir <path>  build directory (default: LLAMA_CPP_BUILD_DIR or <repo-dir>/build-cuda)
  --jobs <count>      parallel build jobs (default: LLAMA_CPP_BUILD_JOBS or detected CPU count)
  --no-update         do not fetch/pull if repository already exists
  --clean             remove build directory before configure/build
  -h, --help          show this help
EOF
}

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi

  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || printf '4\n'
    return
  fi

  printf '4\n'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      [[ $# -ge 2 ]] || die "--repo-dir requires a value"
      REPO_DIR="$2"
      shift 2
      ;;
    --build-dir)
      [[ $# -ge 2 ]] || die "--build-dir requires a value"
      BUILD_DIR="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || die "--jobs requires a value"
      JOBS="$2"
      shift 2
      ;;
    --no-update)
      UPDATE_REPO=false
      shift
      ;;
    --clean)
      CLEAN_BUILD=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

require_command git
require_command cmake

if [[ -z "${BUILD_DIR}" ]]; then
  BUILD_DIR="${REPO_DIR%/}/build-cuda"
fi

if [[ -z "${JOBS}" ]]; then
  JOBS="$(detect_jobs)"
fi

if [[ ! "$JOBS" =~ ^[0-9]+$ ]]; then
  die "--jobs must be a positive integer"
fi

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  log "Cloning llama.cpp into ${REPO_DIR}"
  mkdir -p "$(dirname -- "$REPO_DIR")"
  git clone "$REPO_URL" "$REPO_DIR"
elif [[ "$UPDATE_REPO" == "true" ]]; then
  log "Updating llama.cpp repository at ${REPO_DIR}"
  git -C "$REPO_DIR" fetch --prune origin
  git -C "$REPO_DIR" pull --ff-only
fi

if [[ "$CLEAN_BUILD" == "true" && -d "$BUILD_DIR" ]]; then
  log "Removing build directory: ${BUILD_DIR}"
  rm -rf -- "$BUILD_DIR"
fi

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

log "Configuring CUDA build in ${BUILD_DIR}"
cmake -S "$REPO_DIR" -B "$BUILD_DIR" \
  "${GENERATOR_ARGS[@]}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=ON

log "Building llama-server with ${JOBS} jobs"
cmake --build "$BUILD_DIR" --target llama-server -j "$JOBS"

BINARY_PATH="${BUILD_DIR%/}/bin/llama-server"
[[ -x "$BINARY_PATH" ]] || die "Build completed but binary not found at ${BINARY_PATH}"

cat <<EOF

Build complete.
Binary: ${BINARY_PATH}

Example run:
  ${BINARY_PATH} \\
    --host ${LLAMA_HOST:-0.0.0.0} \\
    --port ${LLAMA_PORT:-10000} \\
    -m ${MODELS_DIR}/<owner>/<repo>/<model>.gguf \\
    -ngl 999 \\
    -fa on \\
    -c 8192
EOF
