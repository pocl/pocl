# USAGE:
#
# 1) format-diff.ps1
# 2) format-diff.ps1 <commit>
#
# 1) formats current unstaged changes and 2) formats changes since the
# commit or changes between commit range (XREF..YREF).

# Set CWD to PoCL's repository root directory.
Push-Location "$PSScriptRoot\..\.."

git diff $args -U0 --no-color | py $PSScriptRoot\clang-format-diff.py `
  -v -regex '.*(\.h$|\.c$|\.cl$)' -i -p1 `
  -style=file:"$PSScriptRoot\style.GNU"

git diff $args -U0 --no-color | py $PSScriptRoot\clang-format-diff.py `
  -v -regex '(.*(\.hpp$|\.hh$|\.cc$|\.cpp$|lib\\llvmopencl\\(?!CMakeLists).*$|\\lib\\CL\\devices\\tce\\.*$))' `
  -i -p1 -style=file:"$PSScriptRoot\style.CPP"

Pop-Location
