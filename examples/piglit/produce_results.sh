#!/bin/sh
piglit/piglit-run.py -v piglit/tests/cl.tests piglit/results/all 2>&1 | egrep "^pass:" > result
sed -i "s/[ \t]*$//" result
LC_ALL=C sort result -o sorted_result
