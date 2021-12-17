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

$RELPATH/clang-format-diff.py -regex '.*(\.h$|\.c$|\.cl$)' -i -p1 -style GNU <$PATCHY
$RELPATH/clang-format-diff.py -regex '(.*(\.hh$|\.cc$))|(lib/llvmopencl/.*\.h)' -i -p1 -style LLVM <$PATCHY

if [ -z "$(git diff)" ]; then
  echo "No changes."
  exit 0
fi

git diff

echo "ACCEPT CHANGES ?"

read REPLY

if [ "$REPLY" == "y" ]; then

  git add -u

  git commit --amend

  if [ -d .git/rebase-merge ]; then

    git rebase --continue

  fi

else

  git add -p

fi
