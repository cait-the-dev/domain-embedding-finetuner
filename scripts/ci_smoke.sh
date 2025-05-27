set -euo pipefail

CFG=${1:-config.yaml}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="outputs"
OUT_FILE="${OUT_DIR}/ci_report_${TIMESTAMP}.txt"
mkdir -p "$OUT_DIR"

echo "[CI] Ingest …"
python -m src.ingest   --config "$CFG"

echo "[CI] Generate synthetic QA …"
python -m src.generate --config "$CFG"

BASE_GGUF=$(yq '.paths.base_gguf'      "$CFG")
TUNE_GGUF=$(yq '.paths.finetuned_gguf' "$CFG")

have_base() { [[ -f "$BASE_GGUF" ]]; }
have_tune() { [[ -f "$TUNE_GGUF" ]]; }

use_tune_as_base() {
  echo "[CI] Using finetuned GGUF as stand-in for base."
  mkdir -p "$(dirname "$BASE_GGUF")"
  ln -sf "$(realpath "$TUNE_GGUF")" "$BASE_GGUF"
}

if have_base; then
  echo "[CI] Base model found → finetune + evaluate."
  python -m src.finetune --config "$CFG"
  python -m src.evaluate --config "$CFG" | tee "$OUT_FILE"
else
  echo "[CI] Base missing but finetuned present → evaluate only."
  python -m src.evaluate --config "$CFG" | tee "$OUT_FILE"
fi

if [[ -f "$OUT_FILE" ]]; then
  grep -E "Hits@|LLM-judge" "$OUT_FILE" || true
  echo "Report saved to $OUT_FILE"
fi
