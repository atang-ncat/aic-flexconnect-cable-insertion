#!/usr/bin/env bash
# collect_multi_config.sh — outer multi-config loop over collect_many.sh
#
# Reads a TSV table of (id, slot, episodes, launch_args) rows and calls
# scripts/collect_many.sh once per row, appending to the same LeRobot
# dataset across configs. Per-config logs and a running summary TSV
# land under $LOG_ROOT so failures can be diagnosed offline.
#
# Quickstart:
#   source ~/lab/ws_aic/setup_dev.sh
#   # Peek at the plan without running anything:
#   scripts/collect_multi_config.sh --group A --dry-run
#   # Actually collect group A:
#   scripts/collect_multi_config.sh --group A
#   # Collect the full 120-episode sweep:
#   scripts/collect_multi_config.sh
#   # Resume from the middle (e.g. after a crash):
#   scripts/collect_multi_config.sh --start-at C.3
#
# Filter options (combine freely):
#   --group X            Keep only rows whose id starts with "X."
#   --ids ID1,ID2,...    Keep only rows with exactly these ids (CSV)
#   --start-at ID        Skip rows before ID (inclusive of ID itself)
#   --stop-at ID         Stop after this ID (inclusive)
#
# Output layout (defaults to /tmp/collect_multi_logs):
#   $LOG_ROOT/
#     run-<timestamp>.log   # full wrapper stdout (tee'd)
#     summary.tsv           # one row per config, APPENDED across runs
#     <CONFIG_ID>/
#       config.txt          # invocation snapshot (args, launch string)
#       gazebo.log          # Gazebo launch log (last ep of this config)
#       ep-NN.log           # per-episode auto_collect stdout
#       collect_many.stdout # stdout of the inner collect_many.sh call
#
# The summary.tsv schema (append-only, human + awk-friendly):
#   ts_iso  config_id  slot  target  saved  missing  would_pass \
#     would_penalize  inner_rc  dur_s  dataset_before  dataset_after
#
# This file is the primary debug artifact — tail it live with:
#   tail -f $LOG_ROOT/summary.tsv

set -u

# ------------------------------------------------------------------
# Defaults (override via CLI flags; anything not exposed can be set
# directly via env before invoking the script)
# ------------------------------------------------------------------
# Resolve script directory so defaults Just Work whether the user is
# running from the workspace root, from scripts/, or via a symlink.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_FILE="${CONFIGS_FILE:-$_SCRIPT_DIR/sfp_auto_configs.tsv}"
COLLECT_MANY="${COLLECT_MANY:-$_SCRIPT_DIR/collect_many.sh}"
DATASET="${DATASET:-/run/host/scratch2/atang/ws_aic/teleop-automated-dataset/sfp}"
REPO_ID="${REPO_ID:-atang/aic_sfp_auto}"
LOG_ROOT="${LOG_ROOT:-/tmp/collect_multi_logs}"
PIXI_DIR="${PIXI_DIR:-/run/host/scratch2/atang/ws_aic/src/aic}"
PLUG_FRAME="${PLUG_FRAME:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"
EPISODES_OVERRIDE=""

GROUP=""
IDS_CSV=""
START_AT=""
STOP_AT=""
DRY_RUN=0
CONTINUE_ON_CONFIG_FAIL=1   # 1 = keep going if a single config comes up short

usage() {
    cat <<'EOF'
Usage: collect_multi_config.sh [options]

Loops over scripts/sfp_auto_configs.tsv and runs collect_many.sh
once per config, appending to the same LeRobot dataset.

Filter options:
  --group X             Only rows whose id starts with "X." (e.g. A, G)
  --ids ID1,ID2,...     Only these exact ids (CSV, no spaces)
  --start-at ID         Skip rows before ID (inclusive)
  --stop-at ID          Stop after this ID (inclusive)

Overrides (otherwise read from env or sensible defaults):
  --configs-file PATH   TSV file of (id, slot, episodes, launch_args)
  --dataset-root DIR    LeRobot dataset root
  --repo-id REPO        LeRobot repo id
  --log-root DIR        Where to write per-config logs + summary.tsv
  --max-attempts N      Forwarded to collect_many.sh
  --ready-timeout SEC   Forwarded to collect_many.sh
  --pixi-dir DIR        Forwarded to collect_many.sh
  --plug-frame FRAME    Forwarded to collect_many.sh (if set)
  --episodes N          Override per-config episode count (debug only;
                        normally the TSV's episodes column wins)

Modes:
  --dry-run             Print the planned invocations and exit
  --stop-on-fail        Abort the sweep as soon as one config misses
                        its target count (default: keep going)
  -h, --help            Show this message

Output:
  $LOG_ROOT/summary.tsv is the primary audit trail; tail -f it during
  long sweeps. Per-config subdirs under $LOG_ROOT preserve Gazebo
  logs + per-episode auto_collect logs for postmortem debugging.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --group)          GROUP="$2"; shift 2 ;;
        --ids)            IDS_CSV="$2"; shift 2 ;;
        --start-at)       START_AT="$2"; shift 2 ;;
        --stop-at)        STOP_AT="$2"; shift 2 ;;
        --configs-file)   CONFIGS_FILE="$2"; shift 2 ;;
        --dataset-root)   DATASET="$2"; shift 2 ;;
        --repo-id)        REPO_ID="$2"; shift 2 ;;
        --log-root)       LOG_ROOT="$2"; shift 2 ;;
        --max-attempts)   MAX_ATTEMPTS="$2"; shift 2 ;;
        --ready-timeout)  READY_TIMEOUT="$2"; shift 2 ;;
        --pixi-dir)       PIXI_DIR="$2"; shift 2 ;;
        --plug-frame)     PLUG_FRAME="$2"; shift 2 ;;
        --episodes)       EPISODES_OVERRIDE="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --stop-on-fail)   CONTINUE_ON_CONFIG_FAIL=0; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "[multi] Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

# ------------------------------------------------------------------
# Environment sanity checks
# ------------------------------------------------------------------
if [[ ! -f "$CONFIGS_FILE" ]]; then
    echo "[multi] Configs file not found: $CONFIGS_FILE" >&2
    exit 3
fi
if [[ ! -x "$COLLECT_MANY" ]]; then
    echo "[multi] collect_many.sh not executable at $COLLECT_MANY" >&2
    exit 3
fi
if (( DRY_RUN == 0 )); then
    if ! command -v ros2 >/dev/null; then
        echo "[multi] ros2 not on PATH. Did you source setup_dev.sh?" >&2
        exit 3
    fi
    if ! command -v pixi >/dev/null; then
        echo "[multi] pixi not on PATH." >&2
        exit 3
    fi
fi

mkdir -p "$LOG_ROOT"
RUN_TS="$(date +%Y%m%d-%H%M%S)"
RUN_LOG="$LOG_ROOT/run-$RUN_TS.log"
SUMMARY_TSV="$LOG_ROOT/summary.tsv"

# Initialize summary.tsv header once (append-only afterwards).
if [[ ! -s "$SUMMARY_TSV" ]]; then
    {
        printf "ts_iso\tconfig_id\tslot\ttarget\tsaved\tmissing"
        printf "\twould_pass\twould_penalize\tinner_rc\tdur_s"
        printf "\tdataset_before\tdataset_after\n"
    } > "$SUMMARY_TSV"
fi

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
ts_iso() { date -Iseconds; }

log() {
    # Print to both console and $RUN_LOG. Avoid `tee` in a pipeline so we
    # don't mess with exit codes elsewhere.
    local msg="$*"
    printf '%s %s\n' "$(ts_iso)" "$msg"
    printf '%s %s\n' "$(ts_iso)" "$msg" >> "$RUN_LOG"
}

count_saved_episodes() {
    # total_episodes from the LeRobot dataset's meta/info.json, or 0.
    local info="$DATASET/meta/info.json"
    if [[ ! -f "$info" ]]; then
        echo 0
        return
    fi
    python3 - "$info" <<'PY' 2>/dev/null || echo 0
import json, sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("total_episodes", 0))
PY
}

# Return 0 if id matches all active filters, 1 otherwise.
id_matches_filter() {
    local id="$1"
    if [[ -n "$GROUP" ]]; then
        # id must start with "$GROUP."
        [[ "$id" == "${GROUP}."* ]] || return 1
    fi
    if [[ -n "$IDS_CSV" ]]; then
        # Match on a comma-delimited list with $id as a complete token.
        case ",$IDS_CSV," in
            *",$id,"*) : ;;
            *) return 1 ;;
        esac
    fi
    return 0
}

# ------------------------------------------------------------------
# Parse TSV into parallel arrays (pre-filter)
# ------------------------------------------------------------------
ALL_IDS=()
ALL_SLOTS=()
ALL_EPISODES=()
ALL_ARGS=()

while IFS=$'\t' read -r col1 col2 col3 col4; do
    # Skip blank lines and comments.
    [[ -z "${col1:-}" ]] && continue
    [[ "${col1:0:1}" == "#" ]] && continue
    if [[ -z "${col4:-}" ]]; then
        echo "[multi] WARN: malformed row (< 4 tab-separated columns): '$col1' -- skipped" >&2
        continue
    fi
    ALL_IDS+=("$col1")
    ALL_SLOTS+=("$col2")
    ALL_EPISODES+=("$col3")
    ALL_ARGS+=("$col4")
done < "$CONFIGS_FILE"

if (( ${#ALL_IDS[@]} == 0 )); then
    echo "[multi] No configs parsed from $CONFIGS_FILE" >&2
    exit 4
fi

# ------------------------------------------------------------------
# Apply filters (--group/--ids/--start-at/--stop-at)
# ------------------------------------------------------------------
SEL_IDS=()
SEL_SLOTS=()
SEL_EPISODES=()
SEL_ARGS=()

in_range=1
if [[ -n "$START_AT" ]]; then
    in_range=0
fi

for i in "${!ALL_IDS[@]}"; do
    id="${ALL_IDS[$i]}"

    if [[ -n "$START_AT" && "$id" == "$START_AT" ]]; then
        in_range=1
    fi
    (( in_range == 1 )) || continue

    if id_matches_filter "$id"; then
        SEL_IDS+=("$id")
        SEL_SLOTS+=("${ALL_SLOTS[$i]}")
        SEL_EPISODES+=("${ALL_EPISODES[$i]}")
        SEL_ARGS+=("${ALL_ARGS[$i]}")
    fi

    if [[ -n "$STOP_AT" && "$id" == "$STOP_AT" ]]; then
        break
    fi
done

TOTAL_CONFIGS=${#SEL_IDS[@]}
if (( TOTAL_CONFIGS == 0 )); then
    echo "[multi] No configs matched the active filters." >&2
    echo "[multi]   group=$GROUP ids=$IDS_CSV start=$START_AT stop=$STOP_AT" >&2
    exit 5
fi

# Total episodes target (for ETA math).
TOTAL_EPISODES=0
for n in "${SEL_EPISODES[@]}"; do
    # Optional per-run override
    if [[ -n "$EPISODES_OVERRIDE" ]]; then n="$EPISODES_OVERRIDE"; fi
    TOTAL_EPISODES=$((TOTAL_EPISODES + n))
done

# ------------------------------------------------------------------
# Pretty print the plan (and optionally exit if --dry-run)
# ------------------------------------------------------------------
{
    echo ""
    echo "=================================================================="
    echo "[multi] === collect_multi_config.sh ==="
    echo "=================================================================="
    echo "[multi] run timestamp:  $RUN_TS"
    echo "[multi] configs file:   $CONFIGS_FILE"
    echo "[multi] dataset root:   $DATASET"
    echo "[multi] repo id:        $REPO_ID"
    echo "[multi] log root:       $LOG_ROOT"
    echo "[multi] summary tsv:    $SUMMARY_TSV"
    echo "[multi] collect_many:   $COLLECT_MANY"
    echo "[multi] max attempts:   $MAX_ATTEMPTS"
    echo "[multi] ready timeout:  $READY_TIMEOUT s"
    [[ -n "$PLUG_FRAME" ]] && echo "[multi] plug frame:     $PLUG_FRAME"
    [[ -n "$EPISODES_OVERRIDE" ]] && echo "[multi] episodes ovr:   $EPISODES_OVERRIDE (per config)"
    echo "[multi] filters:        group='$GROUP' ids='$IDS_CSV' start='$START_AT' stop='$STOP_AT'"
    echo "[multi] plan:           $TOTAL_CONFIGS config(s), $TOTAL_EPISODES episode(s)"
    if (( CONTINUE_ON_CONFIG_FAIL == 0 )); then
        echo "[multi] on short config: STOP"
    else
        echo "[multi] on short config: continue"
    fi
    echo "------------------------------------------------------------------"
    printf "[multi]   %-5s %-5s %-5s %s\n" "IDX" "ID" "SLOT" "EPS / LAUNCH_ARGS"
    for i in "${!SEL_IDS[@]}"; do
        n="${SEL_EPISODES[$i]}"
        [[ -n "$EPISODES_OVERRIDE" ]] && n="$EPISODES_OVERRIDE"
        printf "[multi]   %-5d %-5s %-5s %s / %s\n" \
            $((i+1)) "${SEL_IDS[$i]}" "${SEL_SLOTS[$i]}" \
            "$n" "${SEL_ARGS[$i]}"
    done
    echo "=================================================================="
} | tee -a "$RUN_LOG"

if (( DRY_RUN == 1 )); then
    echo "[multi] --dry-run: exiting without launching anything." | tee -a "$RUN_LOG"
    exit 0
fi

# ------------------------------------------------------------------
# Signal handling: print resume hint so a Ctrl-C leaves a paper trail.
# ------------------------------------------------------------------
CURRENT_ID=""

print_resume_hint() {
    if [[ -z "$CURRENT_ID" ]]; then return; fi
    echo ""
    echo "[multi] To resume from where this run left off, use:"
    echo "[multi]   scripts/collect_multi_config.sh --start-at $CURRENT_ID \\"
    [[ -n "$GROUP" ]]       && echo "[multi]       --group $GROUP \\"
    [[ -n "$IDS_CSV" ]]     && echo "[multi]       --ids $IDS_CSV \\"
    [[ -n "$STOP_AT" ]]     && echo "[multi]       --stop-at $STOP_AT \\"
    echo "[multi]       --dataset-root $DATASET --repo-id $REPO_ID"
}

on_interrupt_multi() {
    echo ""
    echo "[multi] Caught signal; propagating to inner collect_many.sh and exiting."
    # collect_many.sh traps INT/TERM itself and cleans Gazebo up.
    print_resume_hint
    exit 130
}
trap on_interrupt_multi INT TERM

# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
sweep_start=$SECONDS
dataset_initial=$(count_saved_episodes)

configs_done=0
configs_ok=0
configs_short=0
configs_fail=0
eps_saved_total=0
eps_target_total=0
would_pass_total=0
would_penalize_total=0

log "[multi] Initial dataset size: $dataset_initial episode(s)"

for i in "${!SEL_IDS[@]}"; do
    id="${SEL_IDS[$i]}"
    slot="${SEL_SLOTS[$i]}"
    n="${SEL_EPISODES[$i]}"
    [[ -n "$EPISODES_OVERRIDE" ]] && n="$EPISODES_OVERRIDE"
    args="${SEL_ARGS[$i]}"
    CURRENT_ID="$id"

    port_frame="task_board/nic_card_mount_${slot}/sfp_port_0_link"
    ep_log_dir="$LOG_ROOT/$id"
    mkdir -p "$ep_log_dir"

    # Snapshot the invocation for postmortem debugging.
    {
        echo "# config_id: $id"
        echo "# run_ts:    $RUN_TS"
        echo "# start_ts:  $(ts_iso)"
        echo "# slot:      $slot"
        echo "# port_frame: $port_frame"
        echo "# episodes:  $n"
        echo "# launch_args: $args"
        echo "# dataset:   $DATASET"
        echo "# max_attempts: $MAX_ATTEMPTS"
        [[ -n "$PLUG_FRAME" ]] && echo "# plug_frame: $PLUG_FRAME"
    } > "$ep_log_dir/config.txt"

    dataset_before=$(count_saved_episodes)
    eps_target_total=$((eps_target_total + n))

    echo "" | tee -a "$RUN_LOG"
    log "=================================================================="
    log "[multi] CONFIG $((i+1)) / $TOTAL_CONFIGS  —  $id  (slot $slot, target $n ep)"
    log "[multi]   port_frame:  $port_frame"
    log "[multi]   launch_args: $args"
    log "[multi]   log dir:     $ep_log_dir"
    log "[multi]   dataset before: $dataset_before"
    log "=================================================================="

    # Build the collect_many.sh invocation. We pass EP_LOG_DIR via env so
    # per-episode logs land under $ep_log_dir instead of the shared
    # /tmp/collect_many_logs (and get overwritten by the next config).
    plug_flag=()
    if [[ -n "$PLUG_FRAME" ]]; then
        plug_flag=(--plug-frame "$PLUG_FRAME")
    fi

    config_start=$SECONDS
    # NB: do not `set -e` here; we want to capture the rc and keep going.
    EP_LOG_DIR="$ep_log_dir" "$COLLECT_MANY" \
        --episodes "$n" \
        --config "$args" \
        --port-frame "$port_frame" \
        --dataset-root "$DATASET" \
        --repo-id "$REPO_ID" \
        --max-attempts "$MAX_ATTEMPTS" \
        --ready-timeout "$READY_TIMEOUT" \
        --gazebo-log "$ep_log_dir/gazebo.log" \
        --pixi-dir "$PIXI_DIR" \
        "${plug_flag[@]}" \
        2>&1 | tee "$ep_log_dir/collect_many.stdout"
    rc=${PIPESTATUS[0]}
    config_dur=$((SECONDS - config_start))

    dataset_after=$(count_saved_episodes)
    saved=$((dataset_after - dataset_before))
    missing=$((n - saved))
    (( missing < 0 )) && missing=0

    # Parse the collect_many summary block for the would-pass / would-fail
    # tally it already computes. Lines look like:
    #   [wrapper]   would pass scoring: 2
    #   [wrapper]   would be penalized: 1
    pass_line=$(grep -E '^\[wrapper\]   would pass scoring:' \
                "$ep_log_dir/collect_many.stdout" | tail -n 1 || true)
    fail_line=$(grep -E '^\[wrapper\]   would be penalized:' \
                "$ep_log_dir/collect_many.stdout" | tail -n 1 || true)
    wp=$(echo "$pass_line" | sed -nE 's/.*: *([0-9]+).*/\1/p')
    wf=$(echo "$fail_line" | sed -nE 's/.*: *([0-9]+).*/\1/p')
    [[ -z "$wp" ]] && wp=0
    [[ -z "$wf" ]] && wf=0

    # Append a row to summary.tsv.
    printf '%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n' \
        "$(ts_iso)" "$id" "$slot" \
        "$n" "$saved" "$missing" \
        "$wp" "$wf" "$rc" "$config_dur" \
        "$dataset_before" "$dataset_after" \
        >> "$SUMMARY_TSV"

    configs_done=$((configs_done + 1))
    eps_saved_total=$((eps_saved_total + saved))
    would_pass_total=$((would_pass_total + wp))
    would_penalize_total=$((would_penalize_total + wf))

    if (( saved == n )); then
        configs_ok=$((configs_ok + 1))
        status="OK"
    elif (( saved > 0 )); then
        configs_short=$((configs_short + 1))
        status="SHORT ($saved/$n)"
    else
        configs_fail=$((configs_fail + 1))
        status="FAILED (0/$n)"
    fi

    # ETA: average per-episode wall time so far, projected over remaining.
    elapsed_now=$((SECONDS - sweep_start))
    remaining_eps=$((eps_target_total - eps_saved_total))
    # Better estimate: use wall time / configs_done * (total_configs - done)
    remaining_configs=$((TOTAL_CONFIGS - configs_done))
    if (( configs_done > 0 && remaining_configs > 0 )); then
        eta_s=$(( elapsed_now * remaining_configs / configs_done ))
        eta_fmt=$(printf '%dm %02ds' $((eta_s/60)) $((eta_s%60)))
    else
        eta_fmt="—"
    fi

    log "[multi] CONFIG $id done in ${config_dur}s — status=$status, inner_rc=$rc"
    log "[multi]   saved=$saved  would_pass=$wp  would_penalize=$wf"
    log "[multi]   dataset: $dataset_before -> $dataset_after"
    log "[multi]   progress: $configs_done/$TOTAL_CONFIGS configs, $eps_saved_total/$eps_target_total episodes, ETA $eta_fmt"

    if (( saved < n )) && (( CONTINUE_ON_CONFIG_FAIL == 0 )); then
        log "[multi] --stop-on-fail: aborting sweep after short config $id"
        print_resume_hint
        break
    fi
done

# Leaving the loop cleanly, clear CURRENT_ID so resume hint isn't spammy.
CURRENT_ID=""

# ------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------
dataset_final=$(count_saved_episodes)
sweep_dur=$((SECONDS - sweep_start))

{
    echo ""
    echo "=================================================================="
    echo "[multi] FINAL SUMMARY"
    echo "=================================================================="
    echo "[multi]   configs total:     $TOTAL_CONFIGS"
    echo "[multi]   configs done:      $configs_done"
    echo "[multi]     ok (full count): $configs_ok"
    echo "[multi]     short:           $configs_short"
    echo "[multi]     failed:          $configs_fail"
    echo "[multi]   episodes saved:    $eps_saved_total / $eps_target_total"
    echo "[multi]   dataset:           $dataset_initial -> $dataset_final"
    echo "[multi]   would pass:        $would_pass_total / $eps_saved_total"
    echo "[multi]   would penalize:    $would_penalize_total / $eps_saved_total"
    printf "[multi]   wall time:         %dm %02ds\n" $((sweep_dur/60)) $((sweep_dur%60))
    echo "[multi]   run log:           $RUN_LOG"
    echo "[multi]   summary tsv:       $SUMMARY_TSV"
    echo "[multi]   per-config logs:   $LOG_ROOT/<CONFIG_ID>/"
    echo "=================================================================="

    # Per-config result table (last TOTAL_CONFIGS rows of summary.tsv
    # belong to this run, even if the file spans multiple runs).
    echo "[multi] Per-config results (this run):"
    printf "[multi]   %-6s %-5s %-5s %-7s %-8s %-12s %-8s\n" \
        "ID" "SLOT" "TGT" "SAVED" "RC" "WOULD_PASS" "DUR_S"
    tail -n "$configs_done" "$SUMMARY_TSV" \
    | awk -F'\t' '{
        printf "[multi]   %-6s %-5s %-5s %-7s %-8s %-12s %-8s\n",
            $2, $3, $4, $5, $9, $7"/"$5, $10
      }'
    echo "=================================================================="
} | tee -a "$RUN_LOG"

if (( configs_fail > 0 || configs_short > 0 )); then
    exit 1
fi
exit 0
