#!/bin/bash
set -euo pipefail

REMOTE_DIR=/home/ubuntu/mps
HF_HOME_DIR=/ephemeral/hf_cache
RUN_SCRIPT="${1:-run_all.sh}"

echo "remote_setup.sh starting at $(date -u). Run script: $RUN_SCRIPT."
cd "$REMOTE_DIR"

if [ -f .hf_token ]; then
    HF_TOKEN="$(cat .hf_token)"
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "No HF token found." >&2
    echo "Set HF_TOKEN env var, or put your token in $REMOTE_DIR/.hf_token, then rerun." >&2
    exit 1
fi

export HF_HOME="$HF_HOME_DIR"
mkdir -p "$HF_HOME"
printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
chmod 600 "$HF_HOME/token"
shred -u .hf_token 2>/dev/null || rm -f .hf_token

if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing deps (about 3 min)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet pydantic

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch sees:', torch.cuda.get_device_name(0))"

echo "Verifying sensitivity CSVs..."
missing=0
for csv in \
    src/results/exp01_meta-llama_Llama-3.2-3B_int4.csv \
    src/results/exp01_meta-llama_Llama-3.1-8B_int4.csv \
    src/results/exp01_google_gemma-4-E4B_int4.csv \
    src/results/exp01_Qwen_Qwen3-8B_int4.csv; do
    if [ ! -f "$csv" ]; then
        echo "  missing: $csv" >&2
        missing=1
    fi
done
[ $missing -eq 0 ] || { echo "Required CSVs missing — aborting." >&2; exit 1; }

[ -f "$RUN_SCRIPT" ] || { echo "$RUN_SCRIPT missing — aborting." >&2; exit 1; }
chmod +x "$RUN_SCRIPT"

echo "Validating HF token..."
python -c "from huggingface_hub import HfApi; u=HfApi().whoami(); print('HF user:', u.get('name', '?'))"

LOG_FILE="${RUN_SCRIPT%.sh}.log"
echo "Launching $RUN_SCRIPT detached..."
rm -f "$LOG_FILE"
nohup bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1 &
LAUNCH_PID=$!
disown
echo "$RUN_SCRIPT PID: $LAUNCH_PID."

sleep 20
if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo
    echo "$RUN_SCRIPT is still running."
    echo
    echo "Latest log:"
    tail -15 "$LOG_FILE"
    echo
    echo "Safe to disconnect (Ctrl+D)."
    echo "Reconnect anytime with: tnr connect <id>"
    echo "Monitor:  tail -f $REMOTE_DIR/$LOG_FILE"
    echo "Results:  ls -la $REMOTE_DIR/src/results/*/exp04/*.json"
else
    echo
    echo "$RUN_SCRIPT died. $LOG_FILE:"
    cat "$LOG_FILE"
    exit 1
fi
