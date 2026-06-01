#!/bin/bash

set -euo pipefail

INSTANCE_ID="${1:-0}"
REMOTE_DIR="/home/ubuntu/mps"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

command -v tnr >/dev/null || { echo "tnr CLI not installed (pip install thundercompute)" >&2; exit 1; }
command -v rsync >/dev/null || { echo "rsync not installed" >&2; exit 1; }

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN env var not set." >&2
    echo "Get a token from https://huggingface.co/settings/tokens then run:" >&2
    echo "  export HF_TOKEN=hf_xxxxxxxxxxxx" >&2
    echo "  bash deploy_to_thunder.sh $INSTANCE_ID" >&2
    exit 1
fi

for f in run_all.sh remote_setup.sh requirements.txt src configs; do
    [ -e "$PROJECT_DIR/$f" ] || { echo "missing $PROJECT_DIR/$f — run from project root" >&2; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "Staging project (excluding .venv, caches, results.zip)..."
rsync -a \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='results.zip' \
    --exclude='.hf_token' \
    "$PROJECT_DIR/" "$STAGE/"

printf '%s' "$HF_TOKEN" > "$STAGE/.hf_token"
chmod 600 "$STAGE/.hf_token"

STAGE_SIZE=$(du -sh "$STAGE" | cut -f1)
echo "Staged size: $STAGE_SIZE."

echo "Uploading to instance $INSTANCE_ID:$REMOTE_DIR/..."
tnr scp "$STAGE/" "${INSTANCE_ID}:${REMOTE_DIR}/"

echo
echo "Upload complete."
echo
echo "Next, on the instance, run:"
echo "  tnr connect $INSTANCE_ID"
echo "  bash $REMOTE_DIR/remote_setup.sh"
echo
echo "remote_setup.sh handles venv, deps, HF auth, verification, and launches run_all.sh detached."
echo "When it prints 'Safe to disconnect', close the SSH session with Ctrl+D."
echo
echo "To monitor later:"
echo "  tnr connect $INSTANCE_ID"
echo "  tail -f $REMOTE_DIR/run_all.log"
echo
echo "When the sweep finishes, pull results back:"
echo "  tnr scp ${INSTANCE_ID}:${REMOTE_DIR}/src/results/ $PROJECT_DIR/src/"
