# Usage of this script needs git-clang-format.bat to be added to the
# PATH.
#
# Usage:
#
#   format-diff.ps1
#   format-diff.ps1 REF
#
# The former formats staged changes and the latter formats changes since
# the REF commit.
#
# Note that unstaged changes will not be formatted.

git-clang-format.bat `
  --extensions h,c,cl --style "file:$PSScriptRoot\style.GNU" @args
git-clang-format.bat `
  --extensions hh,hpp,cc,cpp --style "file:$PSScriptRoot\style.CPP" @args
