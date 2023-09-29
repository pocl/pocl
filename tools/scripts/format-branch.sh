#!/usr/bin/env bash

if [ ! -e .git ]; then
  echo "must be run in git repo"
  exit 1
fi

case "$(git describe --always --dirty=-DIRTY)" in
  *-DIRTY)
    echo "There are uncommitted changes - aborting."
    exit 1
esac

PATCHY=$(mktemp /tmp/pocl.XXXXXXXX.patch)
trap "rm -f $PATCHY" EXIT

git diff main -U0 --no-color >$PATCHY

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

$RELPATH/clang-format-diff.py -regex '.*(\.h$|\.c$|\.cl$)' -i -p1 -style GNU <$PATCHY
$RELPATH/clang-format-diff.py -regex '(.*(\.hpp$|\.hh$|\.cc$|\.cpp$))|(lib/llvmopencl/.*)|(lib/CL/devices/tce/.*)' -i -p1 -style LLVM <$PATCHY
