#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=scripts/lib/load-dotenv.sh
source "${PROJECT_DIR}/scripts/lib/load-dotenv.sh"
load_dotenv_file "${PROJECT_DIR}/.env"

SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"
OUTPUT_ROOT="${MODELS_DIR:-/home/sapiens/models}"
REPO_ID=""
MODEL_FILE=""
INCLUDE_PATTERN="${HF_DOWNLOAD_INCLUDE_PATTERN:-*.gguf}"
REVISION=""
HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-30}"
ENABLE_XET_HIGH_PERFORMANCE="${HF_DOWNLOAD_ENABLE_XET_HIGH_PERFORMANCE:-true}"
QUIET=false
HF_CLI=""

usage() {
  cat <<'EOF'
Download GGUF models from Hugging Face into /home/sapiens/models by default.

Usage:
  ./scripts/download-hf-model.sh <repo-id> [options]

Options:
  --file <filename>       download only one file from the repo
  --include <pattern>     glob include pattern when --file is not provided (default: *.gguf)
  --revision <ref>        branch, tag, or commit SHA
  --output-root <path>    root directory for downloads (default: /home/sapiens/models)
  --token <hf_token>      Hugging Face token (or use HF_TOKEN in .env)
  --timeout <seconds>     set HF_HUB_DOWNLOAD_TIMEOUT (default: 30)
  --no-xet-high-perf      disable HF_XET_HIGH_PERFORMANCE=1 optimization
  --quiet                 only print final downloaded path(s)
  -h, --help              show this help

Examples:
  # Download Qwen 2.5 1.5B Instruct (Top pick for fast RAG & code)
  ./scripts/download-hf-model.sh Qwen/Qwen2.5-1.5B-Instruct-GGUF --file qwen2.5-1.5b-instruct-q8_0.gguf

  # Download DeepSeek-R1 Distill 1.5B (Micro reasoning model with <think> blocks)
  ./scripts/download-hf-model.sh unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF --file DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf

  # Download Gemma 2 2B Instruct
  ./scripts/download-hf-model.sh bartowski/gemma-2-2b-it-GGUF --file gemma-2-2b-it-Q8_0.gguf
EOF
}

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*" >&2
}

warn() {
  printf '[%s] WARN: %s\n' "$SCRIPT_NAME" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

install_hf_cli() {
  require_command python3
  python3 -m pip --version >/dev/null 2>&1 || die "pip is required. Install python3-pip first."

  log "Installing Hugging Face CLI (huggingface_hub[hf_xet]) for current user."
  python3 -m pip install --user --upgrade "huggingface_hub[hf_xet]"
  export PATH="$HOME/.local/bin:$PATH"
}

resolve_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
    return
  fi

  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
    return
  fi

  install_hf_cli

  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
  else
    die "Unable to find Hugging Face CLI after installation"
  fi
}

has_hf_xet_module() {
  require_command python3
  python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("hf_xet") else 1)
PY
}

enable_xet_high_performance() {
  if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not found; cannot verify hf_xet module. Continuing with default transfer mode."
    return
  fi

  if ! has_hf_xet_module; then
    warn "hf_xet module not found. Attempting installation."
    python3 -m pip --version >/dev/null 2>&1 || {
      warn "pip unavailable; continuing with default transfer mode."
      return
    }
    python3 -m pip install --user --upgrade "huggingface_hub[hf_xet]" || {
      warn "Failed to install hf_xet extras. Continuing with default transfer mode."
      return
    }
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if has_hf_xet_module; then
    export HF_XET_HIGH_PERFORMANCE=1
    log "Enabled HF_XET_HIGH_PERFORMANCE=1"
  else
    warn "hf_xet still unavailable; continuing with default transfer mode."
  fi
}

validate_positive_int() {
  local value="$1"
  local name="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a positive integer"
  [[ "$value" -gt 0 ]] || die "${name} must be greater than 0"
}

check_auth_state() {
  if [[ -n "$HF_TOKEN_VALUE" ]]; then
    export HF_TOKEN="$HF_TOKEN_VALUE"
    log "Using HF token from --token/HF_TOKEN for authenticated downloads."
    return
  fi

  if hf_whoami >/dev/null 2>&1; then
    log "Using token from existing 'hf auth login' session."
    return
  fi

  warn "No Hugging Face token detected. Downloads will use anonymous limits and may be throttled."
  warn "Set HF_TOKEN or pass --token for higher resolver rate limits."
}

hf_whoami() {
  if [[ "$HF_CLI" == "hf" ]]; then
    "$HF_CLI" auth whoami
    return
  fi

  "$HF_CLI" whoami
}

is_truthy() {
  local value="${1:-}"
  case "${value,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      [[ $# -ge 2 ]] || die "--file requires a value"
      MODEL_FILE="$2"
      shift 2
      ;;
    --include)
      [[ $# -ge 2 ]] || die "--include requires a value"
      INCLUDE_PATTERN="$2"
      shift 2
      ;;
    --revision)
      [[ $# -ge 2 ]] || die "--revision requires a value"
      REVISION="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires a value"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --token)
      [[ $# -ge 2 ]] || die "--token requires a value"
      HF_TOKEN_VALUE="$2"
      shift 2
      ;;
    --timeout)
      [[ $# -ge 2 ]] || die "--timeout requires a value"
      DOWNLOAD_TIMEOUT="$2"
      shift 2
      ;;
    --no-xet-high-perf)
      ENABLE_XET_HIGH_PERFORMANCE=false
      shift
      ;;
    --quiet)
      QUIET=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      if [[ -z "$REPO_ID" ]]; then
        REPO_ID="$1"
      else
        die "Unexpected positional argument: $1"
      fi
      shift
      ;;
  esac
done

[[ -n "$REPO_ID" ]] || {
  usage
  die "repo-id is required"
}

validate_positive_int "$DOWNLOAD_TIMEOUT" "timeout"
export HF_HUB_DOWNLOAD_TIMEOUT="$DOWNLOAD_TIMEOUT"

resolve_hf_cli
if is_truthy "$ENABLE_XET_HIGH_PERFORMANCE"; then
  enable_xet_high_performance
else
  log "Xet high-performance mode disabled by config."
fi
check_auth_state

TARGET_DIR="${OUTPUT_ROOT%/}/${REPO_ID}"
mkdir -p "$TARGET_DIR"

QUIET_ARGS=()
if [[ "$QUIET" == "true" ]]; then
  QUIET_ARGS=(--quiet)
fi

if [[ -n "$MODEL_FILE" ]]; then
  log "Downloading ${MODEL_FILE} from ${REPO_ID} to ${TARGET_DIR}"
  CMD=("$HF_CLI" download "$REPO_ID" "$MODEL_FILE" \
    --repo-type model \
    --local-dir "$TARGET_DIR")
else
  log "Downloading files matching '${INCLUDE_PATTERN}' from ${REPO_ID} to ${TARGET_DIR}"
  CMD=("$HF_CLI" download "$REPO_ID" \
    --repo-type model \
    --local-dir "$TARGET_DIR" \
    --include "$INCLUDE_PATTERN")
fi

if [[ -n "$REVISION" ]]; then
  CMD+=(--revision "$REVISION")
fi

CMD+=("${QUIET_ARGS[@]}")
"${CMD[@]}"

log "Download complete: ${TARGET_DIR}"
