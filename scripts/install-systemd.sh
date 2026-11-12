#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=scripts/lib/load-dotenv.sh
source "${PROJECT_DIR}/scripts/lib/load-dotenv.sh"
load_dotenv_file "${PROJECT_DIR}/.env"

MODE="user"
AUTO_START=true
ACTION="install"

usage() {
  cat <<'EOF'
Install or uninstall systemd daemon units for Sleeping Beauty:
  - sleeping-beauty-llm.service    (Qwen3.5-2B via llama-server on :10000)
  - sleeping-beauty-embed.service  (EmbeddingGemma-2 on :10001)
  - sleeping-beauty.service        (Web Hub + API on :8000; starts/stops the others)

Usage:
  ./scripts/install-systemd.sh              # Install & start user services (systemctl --user)
  sudo ./scripts/install-systemd.sh --system  # Install & start system-wide units (systemctl)
  ./scripts/install-systemd.sh --no-start   # Install & enable without starting now
  ./scripts/install-systemd.sh --uninstall  # Stop, disable, and remove installed units
EOF
}

log() {
  printf '[%s] %s\n' "$SCRIPT_NAME" "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)
      MODE="user"
      shift
      ;;
    --system)
      MODE="system"
      shift
      ;;
    --no-start)
      AUTO_START=false
      shift
      ;;
    --uninstall)
      ACTION="uninstall"
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

command -v systemctl >/dev/null 2>&1 || die "systemctl not found"

UNITS=(
  "sleeping-beauty-llm.service"
  "sleeping-beauty-embed.service"
  "sleeping-beauty.service"
)

if [[ "$MODE" == "system" ]]; then
  [[ "${EUID}" -eq 0 ]] || die "--system requires root (run with sudo)"
  UNIT_DIR="/etc/systemd/system"
  PRESET_PATH="/etc/sleeping-beauty/models-preset.ini"
  RUN_USER="${SUDO_USER:-$(id -un)}"
  RUN_GROUP="$(id -gn "$RUN_USER")"
  SYSTEM_USER_LINES="User=${RUN_USER}"$'\n'"Group=${RUN_GROUP}"$'\n'
  NETWORK_UNIT_LINES="After=network-online.target"$'\n'"Wants=network-online.target"$'\n'
  WANTED_BY="multi-user.target"
  SYSCTL=(systemctl)
  PREFIX_CMD="systemctl"
  JOURNAL_CMD="journalctl"
else
  [[ "${EUID}" -ne 0 ]] || die "Do not run --user mode as root"
  UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
  PRESET_PATH="${XDG_CONFIG_HOME:-$HOME/.config}/sleeping-beauty/models-preset.ini"
  SYSTEM_USER_LINES=""
  NETWORK_UNIT_LINES=""
  WANTED_BY="default.target"
  SYSCTL=(systemctl --user)
  PREFIX_CMD="systemctl --user"
  JOURNAL_CMD="journalctl --user"
fi

if [[ "$ACTION" == "uninstall" ]]; then
  log "Stopping and disabling Sleeping Beauty systemd units (${MODE})..."
  "${SYSCTL[@]}" stop sleeping-beauty.service >/dev/null 2>&1 || true
  "${SYSCTL[@]}" disable "${UNITS[@]}" >/dev/null 2>&1 || true
  for u in "${UNITS[@]}"; do
    rm -f -- "${UNIT_DIR}/${u}"
  done
  rm -f -- "$PRESET_PATH"
  "${SYSCTL[@]}" daemon-reload
  log "Uninstalled units from ${UNIT_DIR}"
  exit 0
fi

# Validate runtime binaries and models
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/home/sapiens/llama.cpp/build-cuda/bin/llama-server}"
MODELS_DIR="${MODELS_DIR:-/home/sapiens/models}"
LLM_MODEL="${LLM_GGUF_PATH:-${MODELS_DIR}/Qwen3.5-2B-Q8_0.gguf}"
EMBED_MODEL="${EMBEDDING_GGUF_PATH:-${MODELS_DIR}/embeddinggemma-2-Q8_0.gguf}"
LLAMA_HOST="${LLAMA_HOST:-0.0.0.0}"
LLM_PORT="${LLAMA_PORT:-10000}"
LLAMA_NGL="${LLAMA_N_GPU_LAYERS:-99}"
LLAMA_CTX="${LLAMA_CONTEXT_SIZE:-102400}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-512}"
LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-64}"
LLAMA_KV_CACHE_TYPE_K="${LLAMA_KV_CACHE_TYPE_K:-q8_0}"
LLAMA_KV_CACHE_TYPE_V="${LLAMA_KV_CACHE_TYPE_V:-q8_0}"
SB_HOST="${HOST:-0.0.0.0}"
SB_PORT="${PORT:-8000}"

if [[ "${EMBEDDING_SERVER_URL:-}" =~ :([0-9]+)(/|$) ]]; then
  EMBED_PORT="${BASH_REMATCH[1]}"
else
  EMBED_PORT="10001"
fi

VENV_PY="${PROJECT_DIR}/.venv/bin/python"
[[ -x "$VENV_PY" ]] || die "Python venv not found at ${VENV_PY}. Run ./sb --help first."
[[ -x "$LLAMA_SERVER_BIN" ]] || die "llama-server binary not found at ${LLAMA_SERVER_BIN}"
[[ -f "$LLM_MODEL" ]] || die "LLM GGUF model not found at ${LLM_MODEL}"
[[ -f "$EMBED_MODEL" ]] || die "Embedding GGUF model not found at ${EMBED_MODEL}"

LLAMA_BIN_DIR="$(dirname -- "$LLAMA_SERVER_BIN")"
mkdir -p "$UNIT_DIR"
rm -f -- "$PRESET_PATH"

render_template() {
  local src="$1"
  local dst="$2"
  "$VENV_PY" - "$src" "$dst" \
    "$PROJECT_DIR" "$MODELS_DIR" "$LLAMA_SERVER_BIN" "$LLAMA_BIN_DIR" \
    "$LLM_MODEL" "$EMBED_MODEL" "$LLAMA_HOST" "$LLM_PORT" "$EMBED_PORT" \
    "$LLAMA_NGL" "$LLAMA_CTX" "$SB_HOST" "$SB_PORT" "$WANTED_BY" \
    "$SYSTEM_USER_LINES" "$NETWORK_UNIT_LINES" \
    "$LLAMA_PARALLEL" "$LLAMA_BATCH_SIZE" "$LLAMA_UBATCH_SIZE" \
    "$LLAMA_KV_CACHE_TYPE_K" "$LLAMA_KV_CACHE_TYPE_V" <<'PY'
import sys
from pathlib import Path

(_, src, dst, proj, models_dir, llama_bin, llama_dir,
 llm_m, emb_m, l_host, l_port, e_port, ngl, ctx, sb_host, sb_port,
 wanted, user_lines, net_lines, parallel, batch, ubatch, ctk, ctv) = sys.argv
text = Path(src).read_text(encoding="utf-8")
repl = {
    "@PROJECT_DIR@": proj,
    "@MODELS_DIR@": models_dir,
    "@LLAMA_SERVER_BIN@": llama_bin,
    "@LLAMA_BIN_DIR@": llama_dir,
    "@LLM_MODEL@": llm_m,
    "@EMBED_MODEL@": emb_m,
    "@LLAMA_HOST@": l_host,
    "@LLM_PORT@": l_port,
    "@EMBED_PORT@": e_port,
    "@LLAMA_NGL@": ngl,
    "@LLAMA_CTX@": ctx,
    "@LLAMA_PARALLEL@": parallel,
    "@LLAMA_BATCH_SIZE@": batch,
    "@LLAMA_UBATCH_SIZE@": ubatch,
    "@LLAMA_KV_CACHE_TYPE_K@": ctk,
    "@LLAMA_KV_CACHE_TYPE_V@": ctv,
    "@SB_HOST@": sb_host,
    "@SB_PORT@": sb_port,
    "@WANTED_BY@": wanted,
    "@SYSTEM_USER_LINES@": user_lines,
    "@NETWORK_UNIT_LINES@": net_lines,
}
for k, v in repl.items():
    text = text.replace(k, v)
Path(dst).write_text(text, encoding="utf-8")
PY
  chmod 0644 "$dst"
}

for u in "${UNITS[@]}"; do
  src="${PROJECT_DIR}/systemd/${u}.in"
  [[ -f "$src" ]] || die "Template not found: ${src}"
  log "Installing ${UNIT_DIR}/${u}"
  render_template "$src" "${UNIT_DIR}/${u}"
done

"${SYSCTL[@]}" daemon-reload
"${SYSCTL[@]}" enable sleeping-beauty.service

if [[ "$AUTO_START" == "true" ]]; then
  log "Starting sleeping-beauty.service (brings up LLM :${LLM_PORT}, Embed :${EMBED_PORT}, Web :${SB_PORT})..."
  "${SYSCTL[@]}" restart sleeping-beauty.service
  log "All services started:"
  "${SYSCTL[@]}" --no-pager status "${UNITS[@]}" || true
fi

cat <<EOF

Sleeping Beauty systemd daemon installed (${MODE} mode).
Manage with:
  ${PREFIX_CMD} start sleeping-beauty     # Start all (LLM + Embedding + Web UI)
  ${PREFIX_CMD} stop sleeping-beauty      # Stop all (frees GPU VRAM)
  ${PREFIX_CMD} restart sleeping-beauty   # Restart all
  ${PREFIX_CMD} status sleeping-beauty sleeping-beauty-llm sleeping-beauty-embed
  ${JOURNAL_CMD} -u sleeping-beauty -u sleeping-beauty-llm -u sleeping-beauty-embed -f
EOF
