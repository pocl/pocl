#!/bin/sh
# The test case seems to pass sometimes indeterministically. Filter it out
# to make regression checking feasible.
piglit/piglit-run.py -v piglit/tests/cl.tests piglit/results/all 2>&1 | egrep "^pass:" | \
egrep -v "program@execute@builtin@builtin-float-asinh-1.0.generated" > result
sed -i "s/[ \t]*$//" result
LC_ALL=C sort result -o sorted_result
