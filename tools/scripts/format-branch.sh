#!/bin/bash

if [ ! -d .git ]; then
  echo "must be run in git repo"
  exit 1
fi

PATCHY=/tmp/p.patch

rm -f $PATCHY
git diff master -U0 --no-color >$PATCHY

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

$RELPATH/clang-format-diff.py -regex '.*(\.h$|\.c$|\.cl$)' -i -p1 -style GNU <$PATCHY
$RELPATH/clang-format-diff.py -regex '.*(\.hh$|\.cc$)' -i -p1 -style LLVM <$PATCHY
