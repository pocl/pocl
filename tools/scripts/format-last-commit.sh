#!/bin/bash

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

git show -U0 --no-color >$PATCHY

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

$RELPATH/clang-format-diff.py -regex '(.*(\.hpp$|\.cc$|\.cpp$))|(lib/llvmopencl/.*)' -i -p1 -style LLVM <$PATCHY
$RELPATH/clang-format-diff.py -regex '(.*(\.hh$|\.cc$))|(lib/llvmopencl/.*)|(lib/CL/devices/tce/.*)' -i -p1 -style LLVM <$PATCHY

if [ -z "$(git diff)" ]; then
  echo "No changes."
  exit 0
fi
