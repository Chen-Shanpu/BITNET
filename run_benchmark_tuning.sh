#!/usr/bin/env bash
# =============================================================================
# run_benchmark_tuning.sh
#
# End-to-end BitNet kernel benchmark-tuning runner for all 6 LLaMA model
# scales: 80M, 120M, 300M, 500M, 700M, 1B.
#
# Steps executed
# --------------
#   1. Environment check (Python version, required packages, model configs)
#   2. Heuristic-only tuning (fast, no GPU/build required) for all models
#   3. (Optional) Full benchmark-driven tuning via --benchmark flag
#   4. Progress monitoring – summarises generated INI files and CSV logs
#
# Usage
# -----
#   # Fast heuristic tuning only (no benchmark):
#   bash run_benchmark_tuning.sh
#
#   # Full benchmark-driven tuning (recommended for production):
#   bash run_benchmark_tuning.sh --benchmark
#
#   # Benchmark a subset of models:
#   bash run_benchmark_tuning.sh --benchmark --models "Llama-80M Llama-1B"
#
#   # Control benchmark parameters:
#   bash run_benchmark_tuning.sh --benchmark --seq-len 256 --num-runs 100
#
# Outputs
# -------
#   tuned_kernels/<model>/kernel_config_tl{1,2}.ini  ← optimal kernel configs
#   preset_kernels/<model>/kernel_config_tl{1,2}.ini ← (benchmark mode only)
#   tuning_logs/<model>/tuning_log_tl{1,2}.json      ← full candidate logs
#   tuning_logs/<model>/tuning_log_tl{1,2}_summary_desc.csv
#   tuning_report.md                                  ← markdown summary
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BENCHMARK=false
MODELS="Llama-80M Llama-120M Llama-300M Llama-500M Llama-700M Llama-1B"
SEQ_LEN=128
NUM_RUNS=50
OUTPUT_DIR="tuned_kernels"
TUNING_LOGS_DIR="tuning_logs"
REPORT="tuning_report.md"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)    BENCHMARK=true;  shift ;;
        --models)       MODELS="$2";     shift 2 ;;
        --seq-len)      SEQ_LEN="$2";    shift 2 ;;
        --num-runs)     NUM_RUNS="$2";   shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --report)       REPORT="$2";     shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^# ---/p' "$0" | head -20
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}  ✓${RESET}  $*"; }
warn() { echo -e "${YELLOW}  ⚠${RESET}  $*"; }
err()  { echo -e "${RED}  ✗${RESET}  $*"; }
info() { echo -e "${CYAN}  →${RESET}  $*"; }
hdr()  { echo -e "\n${BOLD}$*${RESET}"; }

# ---------------------------------------------------------------------------
# Step 1 – Environment check
# ---------------------------------------------------------------------------
hdr "Step 1: Environment check"

# Python version
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then
    err "Python 3 not found. Install Python 3.8+ and retry."
    exit 1
fi
PY_VER=$("$PYTHON" -c "import sys; print('%d.%d' % sys.version_info[:2])")
ok "Python $PY_VER at $PYTHON"

# Required packages
MISSING_PKGS=()
for pkg in torch transformers configparser; do
    if "$PYTHON" -c "import $pkg" 2>/dev/null; then
        ok "Package '$pkg' available"
    else
        err "Package '$pkg' not found"
        MISSING_PKGS+=("$pkg")
    fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    warn "Installing missing packages: ${MISSING_PKGS[*]}"
    "$PYTHON" -m pip install "${MISSING_PKGS[@]}" --quiet
    ok "Packages installed"
fi

# Model config files
hdr "  Checking model config files"
ALL_CONFIGS_PRESENT=true
for MODEL in $MODELS; do
    CFG="$SCRIPT_DIR/model_configs/$MODEL/config.json"
    if [[ -f "$CFG" ]]; then
        ok "$MODEL  →  $CFG"
    else
        err "$MODEL  →  config.json NOT FOUND at $CFG"
        ALL_CONFIGS_PRESENT=false
    fi
done

if [[ "$ALL_CONFIGS_PRESENT" == false ]]; then
    err "One or more model configs are missing. Aborting."
    exit 1
fi

# tune_all_models.py
if [[ -f "$SCRIPT_DIR/tune_all_models.py" ]]; then
    ok "tune_all_models.py found"
else
    err "tune_all_models.py not found in $SCRIPT_DIR"
    exit 1
fi

echo ""
ok "Environment checks passed."

# ---------------------------------------------------------------------------
# Step 2 / 3 – Run tuning
# ---------------------------------------------------------------------------
hdr "Step 2: Running kernel tuning"

CMD=("$PYTHON" "$SCRIPT_DIR/tune_all_models.py"
     --models $MODELS
     --output-dir "$OUTPUT_DIR"
     --report     "$REPORT"
     --tuning-logs-dir "$TUNING_LOGS_DIR"
)

if [[ "$BENCHMARK" == true ]]; then
    CMD+=(--benchmark --seq-len "$SEQ_LEN" --num-runs "$NUM_RUNS")
    info "Mode: benchmark-driven tuning (seq_len=$SEQ_LEN, num_runs=$NUM_RUNS)"
else
    info "Mode: heuristic tuning (pass --benchmark for full benchmark-driven tuning)"
fi

info "Command: ${CMD[*]}"
echo ""

START_TIME=$(date +%s)
"${CMD[@]}"
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
ok "Tuning completed in ${ELAPSED}s."

# ---------------------------------------------------------------------------
# Step 4 – Progress monitoring: verify outputs
# ---------------------------------------------------------------------------
hdr "Step 3: Verifying outputs"

PASS=0; FAIL=0

check_file() {
    local path="$1"
    local label="$2"
    if [[ -f "$path" ]]; then
        ok "$label"
        PASS=$(( PASS + 1 ))
    else
        err "MISSING: $label  ($path)"
        FAIL=$(( FAIL + 1 ))
    fi
}

for MODEL in $MODELS; do
    for ARCH in tl1 tl2; do
        check_file "$SCRIPT_DIR/$OUTPUT_DIR/$MODEL/kernel_config_$ARCH.ini" \
            "tuned_kernels/$MODEL/kernel_config_$ARCH.ini"

        if [[ "$BENCHMARK" == true ]]; then
            check_file "$SCRIPT_DIR/$TUNING_LOGS_DIR/$MODEL/tuning_log_${ARCH}.json" \
                "tuning_logs/$MODEL/tuning_log_$ARCH.json"
            check_file "$SCRIPT_DIR/$TUNING_LOGS_DIR/$MODEL/tuning_log_${ARCH}_summary_desc.csv" \
                "tuning_logs/$MODEL/tuning_log_${ARCH}_summary_desc.csv"
        fi
    done
done

# Resolve the report path: absolute paths are used as-is; relative paths are
# relative to SCRIPT_DIR (consistent with how tune_all_models.py resolves them).
if [[ "$REPORT" = /* ]]; then
    REPORT_ABS="$REPORT"
else
    REPORT_ABS="$SCRIPT_DIR/$REPORT"
fi
check_file "$REPORT_ABS" "$REPORT"

# Verify INI files contain all 6 tuning parameters
hdr "  Verifying INI parameter completeness (all 6 params per section)"
REQUIRED_KEYS=(ROW_BLOCK_SIZE COL_BLOCK_SIZE PARALLEL_SIZE bm bk bmm)
for MODEL in $MODELS; do
    for ARCH in tl1 tl2; do
        INI="$SCRIPT_DIR/$OUTPUT_DIR/$MODEL/kernel_config_$ARCH.ini"
        if [[ ! -f "$INI" ]]; then
            continue
        fi
        ALL_KEYS_PRESENT=true
        for KEY in "${REQUIRED_KEYS[@]}"; do
            if ! grep -qi "^${KEY}\s*=" "$INI"; then
                err "$MODEL/$ARCH: missing key '$KEY' in $INI"
                ALL_KEYS_PRESENT=false
                FAIL=$(( FAIL + 1 ))
            fi
        done
        if [[ "$ALL_KEYS_PRESENT" == true ]]; then
            ok "$MODEL/kernel_config_$ARCH.ini contains all 6 tuning params"
            PASS=$(( PASS + 1 ))
        fi
    done
done

# Verify CSV contains tokens_per_second
if [[ "$BENCHMARK" == true ]]; then
    hdr "  Verifying CSV files contain tokens_per_second column"
    for MODEL in $MODELS; do
        for ARCH in tl1 tl2; do
            CSV="$SCRIPT_DIR/$TUNING_LOGS_DIR/$MODEL/tuning_log_${ARCH}_summary_desc.csv"
            if [[ ! -f "$CSV" ]]; then
                continue
            fi
            if head -1 "$CSV" | grep -q "tokens_per_second"; then
                ok "$MODEL/tuning_log_${ARCH}_summary_desc.csv has tokens_per_second"
                PASS=$(( PASS + 1 ))
            else
                err "$MODEL/tuning_log_${ARCH}_summary_desc.csv missing tokens_per_second"
                FAIL=$(( FAIL + 1 ))
            fi
        done
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
hdr "Summary"
echo ""
echo -e "  Checks passed : ${GREEN}${BOLD}$PASS${RESET}"
echo -e "  Checks failed : $([ $FAIL -eq 0 ] && echo "${GREEN}${BOLD}0${RESET}" || echo "${RED}${BOLD}$FAIL${RESET}")"
echo ""

if [[ $FAIL -eq 0 ]]; then
    ok "All checks passed ✅"
    echo ""
    echo "  Tuning report  : $SCRIPT_DIR/$REPORT"
    echo "  Tuned configs  : $SCRIPT_DIR/$OUTPUT_DIR/"
    if [[ "$BENCHMARK" == true ]]; then
        echo "  Tuning logs    : $SCRIPT_DIR/$TUNING_LOGS_DIR/"
    fi
else
    err "$FAIL check(s) failed. Review the output above for details."
    exit 1
fi
