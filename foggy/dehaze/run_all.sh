#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./run_all.sh [--data_dir PATH] [--tag NAME]
  --data_dir  Path to hazy input images (default: data/real)
  --tag       Identifier for this run; when omitted, results land in outputs/original|dcp|aodnet and figs/.
              For custom tags (e.g., sots_o), outputs go to outputs/<tag>/*, figs/<tag>, and NIQE files get tagged suffixes.
USAGE
}

DATA_DIR="data/real"
TAG="main"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$TAG" == "main" ]]; then
    ORIG_OUT="outputs/original"
    DCP_OUT="outputs/dcp"
    AOD_OUT="outputs/aodnet"
    FIG_OUT="figs"
    NIQE_CSV="niqe_results.csv"
    NIQE_SUMMARY="niqe_summary.txt"
else
    ORIG_OUT="outputs/${TAG}/original"
    DCP_OUT="outputs/${TAG}/dcp"
    AOD_OUT="outputs/${TAG}/aodnet"
    FIG_OUT="figs/${TAG}"
    NIQE_CSV="niqe_results_${TAG}.csv"
    NIQE_SUMMARY="niqe_summary_${TAG}.txt"
fi

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Cannot find python interpreter. Set PYTHON_BIN env var to your Python executable."
        exit 1
    fi
fi

"$PYTHON_BIN" src/preprocess.py --in "$DATA_DIR" --out "$ORIG_OUT" --long_side 1024
"$PYTHON_BIN" src/dcp.py --in "$ORIG_OUT" --out "$DCP_OUT" --window 15 --omega 0.95 --radius 60 --eps 1e-3 --t0 0.1
"$PYTHON_BIN" src/aodnet_infer.py --in "$ORIG_OUT" --out "$AOD_OUT" --weights ckpt/aodnet.pth
"$PYTHON_BIN" src/eval_niqe.py --orig "$ORIG_OUT" --dcp "$DCP_OUT" --aod "$AOD_OUT" --csv "$NIQE_CSV" --summary "$NIQE_SUMMARY"
"$PYTHON_BIN" src/viz.py --orig "$ORIG_OUT" --dcp "$DCP_OUT" --aod "$AOD_OUT" --out "$FIG_OUT" --max_examples 8 --niqe_csv "$NIQE_CSV"
