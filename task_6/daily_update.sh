#!/bin/bash

set -e

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_ROOT="$(
    cd "$SCRIPT_DIR/.."
    pwd
)"

PYTHON_ROOT="$(
    cd "$PROJECT_ROOT/.."
    pwd
)"

PYTHON="$PYTHON_ROOT/.venv/bin/python"

START_TIME=$(date +%s)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
}

log "Daily update started"

log "Fetching new pages..."
"$PYTHON" -m task_6.fetch_new_pages

log "Updating vector index..."
"$PYTHON" -m task_6.update_index

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

log "Daily update finished | duration=${DURATION}s"