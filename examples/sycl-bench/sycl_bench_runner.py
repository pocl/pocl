#!/usr/bin/python3

# takes input from sycl-bench benchmark and converts it to JSON suitable for github-action-benchmark

import os
import sys
import subprocess
import argparse
#import traceback

# remove characters
REMOVEMAP = { ord('*'): None }

def print_line(f, benchmark, run_time_mean, run_time_stddev, niters, run_time_min):
    f.write('{ "name": "' + benchmark + '", "unit": "Seconds", "value": ' + str(run_time_mean) + ', "range": ' + str(run_time_stddev) + ', "extra": "niters = ' + str(niters) + ', min = ' + str(run_time_min) + '" },\n')

def filter_content(content, append_path, niters):
    f = None
    fd = 0
    # if the output JSON file does not exist, write the opening "[" for the JSON array
    try:
        fd = os.open(append_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_APPEND, 0o644)
        f = os.fdopen(fd, 'a')
        f.write('[\n')
    except OSError:
        #traceback.print_exc()
        fd = None
        f = open(append_path, 'a')

    content = content.decode('UTF-8')
    #print("Content : \n", content)
    benchmark = "NONE"
    run_time_min = 0
    run_time_avg = 0
    run_time_stddev = 0

    for line in content.splitlines():
        #print("LINE: ", repr(line))

        offset = line.find("******** Results for ");
        if offset > 0:
            if run_time_min > 0 and run_time_mean > 0 and benchmark:
                #coefvar = run_time_stddev / run_time_mean
                print_line(f, benchmark, run_time_mean, run_time_stddev, niters, run_time_min)
                benchmark = None
                run_time_min = 0
                run_time_mean  = 0
                run_time_stddev = 0

            offset += 21
            benchmark = line[offset:-1]
            benchmark = benchmark.translate(REMOVEMAP)
            #print("Benchmark: ", benchmark)

        offset = line.find("run-time-min:");
        if offset >= 0:
            offset += 14
            run_time_min = float(line[offset:-4])
            #print("run_time_min: ", run_time_min)

        offset = line.find("run-time-mean:");
        if offset >= 0:
            offset += 15
            run_time_mean = float(line[offset:-4])
            #print("run_time_mean: ", run_time_mean)

        offset = line.find("run-time-stddev:");
        if offset >= 0:
            offset += 17
            run_time_stddev = float(line[offset:-4])
            #print("run_time_stddev: ", run_time_stddev)

    # write the remaining after the processing
    if run_time_min > 0 and run_time_mean > 0 and benchmark:
        print_line(f, benchmark, run_time_mean, run_time_stddev, niters, run_time_min)

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='sycl_bench_runner',
                        description='Runs a benchmark from sycl-bench and converts its output to a JSON format')
    parser.add_argument('-o', '--output', required=True, help='Path to output JSON file')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    #print("output: ", args.output)
    rest = args.rest
    if rest[0] == '--':
        rest = rest[1:]

    niters = 1
    for arg in rest:
        if arg.startswith('--num-runs='):
            niters = arg[11:]
    print("niters: ", niters)

    print("Running: ", rest)
    proc = subprocess.run(rest, capture_output=True)
    if proc.returncode != 0:
        print("child exit status nonzero:\n")
        print("STDOUT: ", proc.stdout, "\n")
        print("STDERR: ", proc.stderr, "\n")
        sys.exit(proc.returncode)

    filter_content(proc.stdout, args.output, niters)
