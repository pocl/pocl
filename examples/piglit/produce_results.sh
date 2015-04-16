#!/bin/sh
# The below test cases seem to pass indeterministically. Filter them out
# _for now_ to make regression checking feasible.
piglit/piglit-run.py -v piglit/tests/cl.tests piglit/results/all 2>&1 | egrep "^pass:" | \
egrep -v "program@execute@builtin@builtin-float-asinh-1.0.generated|\
program@execute@store@store-uint2-global" > result
sed -i "s/[ \t]*$//" result
LC_ALL=C sort result -o sorted_result
