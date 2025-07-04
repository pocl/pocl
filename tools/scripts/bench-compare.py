#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# Copyright (c) 2020-2023, Zheming Jin
# Copyright (c) 2023,2025 Michal Babej / Intel Finland Oy
# All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Script to compare to result files

import sys
import argparse
import csv
import statistics

import math

def geomean(xs):
        return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))

def main():
    parser = argparse.ArgumentParser(prog='bench-compare.py',
                                     description='Compare benchmark results via geometric mean')

    parser.add_argument('input', nargs=2,
                        help='Two benchmark result files to compute speedup between (OLD NEW)')
    parser.add_argument('--max', type=float, default=1.2, help='max geometric mean difference between old & new (1.2)')
    parser.add_argument('--min', type=float, default=0.8, help='min geometric mean difference between old & new (0.8)')


    args = parser.parse_args()

    data = dict()
    for inputfile in args.input:
        with open(inputfile, 'r') as f:
            print("Reading file: ", inputfile)
            c = csv.reader(f, delimiter=',')
            # fields: 0=name, 1=min, 2=avg, 3=stddev, 4=variance
            data[inputfile] = {}
            for line in c:
                # skip benchmarks with variance >5%
                if float(line[4]) < 0.05:
                    data[inputfile][line[0]] = float(line[1])
                else:
                    print("Skipping because of variance too high: ", line[0])

    old = data[args.input[0]]
    new = data[args.input[1]]

    new_to_olds = []
    print("|Benchmark|new_to_old")
    print("|--|--")
    for k, v in old.items():
        if not k in new:
            continue

        new_to_old = new[k] / old[k]
        new_to_olds.append(new_to_old)

        print("|{}|{:.2f}".format(k, new_to_old))
    print("|--|--")
    if len(new_to_olds) <= 1:
        print("Not enough tests to calculate geomean")
        sys.exit(1)

    g = geomean(new_to_olds)
    print("|Geomean|{}".format(g))
    if g > args.max:
        print("Geomean too large: expected <= %1.2f but got: %1.2f" % (args.max, g))
        sys.exit(1)
    if g < args.min:
        print("Geomean too small: expected >= %1.2f but got: %1.2f" % (args.min, g))
        sys.exit(1)

if __name__ == "__main__":
    main()
