#!/bin/bash

if [ ! -d .git ]; then
  echo "must be run in git repo"
  exit 1
fi

PATCHY=/tmp/p.patch

rm -f $PATCHY
git show -U0 --no-color >$PATCHY

SCRIPTPATH=$( realpath "$0"  )
RELPATH=$(dirname "$SCRIPTPATH")

$RELPATH/clang-format-diff.py -regex '.*(\.h$|\.c$|\.cl$)' -i -p1 -style GNU <$PATCHY
$RELPATH/clang-format-diff.py -regex '(.*(\.hh$|\.cc$))|(lib/llvmopencl/.*\.h)' -i -p1 -style LLVM <$PATCHY

git diff

echo "ACCEPT CHANGES ?"

read REPLY

if [ "$REPLY" == "y" ]; then

  git add -u

  git commit --amend

  if [ -e .git/ORIG_HEAD ]; then

    git rebase --continue

  fi

else

  git add -p

fi
