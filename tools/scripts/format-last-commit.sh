#!/usr/bin/env bash
RELPATH=$(dirname "$(realpath "$0")")
$RELPATH/format-diff.sh HEAD~1 "$@"
