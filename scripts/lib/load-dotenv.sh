#!/usr/bin/env bash

# Load KEY=VALUE pairs from a dotenv file without evaluating shell expressions.
load_dotenv_file() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0

  local raw_line=""
  local line=""
  local key=""
  local value=""

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line%$'\r'}"

    [[ -z "${line//[[:space:]]/}" ]] && continue

    line="${line#"${line%%[![:space:]]*}"}"
    [[ "${line:0:1}" == "#" ]] && continue
    [[ "$line" == *=* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"

    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    if [[ "$value" =~ ^\"(.*)\"$ ]]; then
      value="${BASH_REMATCH[1]}"
    fi

    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    export "$key=$value"
  done < "$env_file"
}
