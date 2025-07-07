#!/usr/bin/python3

import os
import sys
import subprocess
import argparse


# remove characters
REMOVEMAP = { ord('*'): None }

if len(sys.argv) < 2:
    print("USage: SCRIPT logfile")
    exit(1)

def filter_content(content, append_path, niters):
    f = open(append_path, 'a')
    content = content.decode('UTF-8')
    print("Content : \n", content)
    benchmark = "NONE"
    run_time_min = 0
    run_time_avg = 0
    run_time_stddev = 0

    for line in content.splitlines():
        #print("LINE: ", repr(line))

        offset = line.find("******** Results for ");
        if offset > 0:
            if run_time_min > 0 and run_time_mean > 0:
                #coefvar = run_time_stddev / run_time_mean
                total_time = str(run_time_mean * int(niters))
                print("Writing Results for: ", benchmark)
                f.write('{ "name": "' + benchmark + '", "iterations": ' + niters + ', "real_time": ' + total_time + ', "cpu_time": ' + total_time + " },\n")
                benchmark = None
                run_time_min = 0
                run_time_mean  = 0

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
        total_time = str(run_time_mean * int(niters))
        f.write('{ "name": "' + benchmark + '", "iterations": ' + niters + ', "real_time": ' + total_time + ', "cpu_time": ' + total_time + " },\n")

    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='sycl_bench_runner_json',
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
