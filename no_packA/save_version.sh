#!/bin/bash
# save_version.sh
NAME="$1"
MSG="$2"
TS=$(date +%Y%m%d_%H%M%S)
NEW_FILE="${NAME%.cpp}_${TS}.cpp"

cp "$NAME" "$NEW_FILE"
echo "[$TS] $MSG" >> VERSIONS.log
echo "Saved: $NEW_FILE"
