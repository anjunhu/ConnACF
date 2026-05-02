#!/usr/bin/env bash
# tmux_status.sh — report status of all tmux sessions
# Status: RUNNING (turn N), FINISHED (turn N), or ERROR (last error line)

sessions=$(tmux list-sessions -F '#{session_name}' 2>/dev/null)
if [[ -z "$sessions" ]]; then
    echo "No tmux sessions found."
    exit 0
fi

printf "%-30s %-12s %s\n" "SESSION" "STATUS" "DETAIL"
printf "%-30s %-12s %s\n" "-------" "------" "------"

while IFS= read -r session; do
    # Grab last 200 lines of pane output
    log=$(tmux capture-pane -t "$session" -p -S -200 2>/dev/null)

    # Check for error indicators
    error_line=$(echo "$log" | grep -iE "error|traceback|exception|critical" | grep -viE "successfully|no error|0 error" | tail -1)

    # Look for turn indicators — matches "Turn N/M", "turn N", "--- Turn N ---"
    last_turn=$(echo "$log" | grep -oiE "(turn|epoch)[[:space:]]+[0-9]+(/[0-9]+)?" | tail -1)

    # Check if finished
    finished=$(echo "$log" | grep -iE "experiment (complete|done|finished)|all turns (complete|done)|test result:|attack complete" | tail -1)

    if [[ -n "$error_line" && -z "$finished" ]]; then
        # Truncate error to 80 chars
        detail=$(echo "$error_line" | sed 's/^[[:space:]]*//' | cut -c1-80)
        printf "%-30s %-12s %s\n" "$session" "ERROR" "$detail"
    elif [[ -n "$finished" ]]; then
        detail=$(echo "$last_turn" | tr -s ' ')
        [[ -z "$detail" ]] && detail="(no turn info)"
        printf "%-30s %-12s %s\n" "$session" "FINISHED" "$detail"
    elif [[ -n "$last_turn" ]]; then
        printf "%-30s %-12s %s\n" "$session" "RUNNING" "$last_turn"
    else
        # No recognisable output — show last non-empty line
        last_line=$(echo "$log" | grep -v '^[[:space:]]*$' | tail -1 | cut -c1-80)
        printf "%-30s %-12s %s\n" "$session" "UNKNOWN" "$last_line"
    fi
done <<< "$sessions"
