#!/bin/sh
#
# Executes all the examples in a row.
#
TESTS="example1 example2 barriers forloops trig"

for dname in ${TESTS}; 
do 
    echo "### Running $dname..."
	cd $dname; ./$1/$dname; 
	cd ..; 
    echo
done
