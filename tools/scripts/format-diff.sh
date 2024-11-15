#!/usr/bin/env bash
# USAGE:
#
# 1) format-diff.sh
# 2) format-diff.sh <commit>
#
# 1) formats current unstaged changes and 2) formats changes since the
# commit or changes between commit range (XREF..YREF).

GITROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ $? -ne 0 ]; then
  >&2 echo "must be run in git repo"
  exit 1
fi

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

# cd to root directory of the git repo
cd "${GITROOT}" || exit 1

if [ $# -ne 1 ]; then
    if git status --porcelain=1 -uno | grep '^.[MTDRC] '; then
	>&2 echo "There are unstaged changes - aborting."
	exit 1
    fi
fi

PATCHY=$(mktemp /tmp/pocl.XXXXXXXX.patch)
trap 'rm -f $PATCHY' EXIT

git diff "$@" -U0 --no-color >"$PATCHY"

if [ -z "${CLANG_FORMAT_BIN}" ]; then
    CLANG_FORMAT_BIN=clang-format
    # clang-format v15 is the lowest version that supports the custom
    # style files ahead.
    for i in {25..15}; do
	cand_bin=$(command -v "clang-format-$i") || continue
	CLANG_FORMAT_BIN=$cand_bin
	break;
    done
fi

echo "Using: $CLANG_FORMAT_BIN"

"$RELPATH"/clang-format-diff.py -v -binary "$CLANG_FORMAT_BIN" \
	  -regex '.*(\.h$|\.c$|\.cl$)' -i -p1 \
	  -style=file:"$RELPATH/style.GNU" <"$PATCHY"

# We need to recreate the diff since the old patch is stale.
git diff "$@" -U0 --no-color >"$PATCHY"

"$RELPATH"/clang-format-diff.py -v -binary "$CLANG_FORMAT_BIN" \
	  -regex '(.*(\.hpp$|\.hh$|\.cc$|\.cpp$|lib/llvmopencl/.*$|/lib/CL/devices/tce/.*$))' \
	  -i -p1 -style=file:"$RELPATH/style.CPP" <"$PATCHY"

