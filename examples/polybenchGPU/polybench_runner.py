#!/usr/bin/python3

# runs a single Polybench benchmark, captures output, and converts it to JSON suitable for github-action-benchmark

import os
import sys
import subprocess
import argparse
#import traceback

def filter_content(content, append_path, benchmark_name):
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
    print("Content : \n", content)
    run_time = None
    next_line_contains_result = False

    for line in content.splitlines():

        if next_line_contains_result:
            run_time = line
            next_line_contains_result = False
            break

        if line.startswith("GPU Time in seconds"):
            next_line_contains_result = True
            continue

    # write after the processing, if result exists.
    # some of the test self-skip
    if run_time:
        f.write('{ "name": "' + benchmark_name + '", "unit": "Seconds", "value": ' + run_time.strip() + ' },\n')
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='polybench_runner_json',
                        description='Runs a benchmark from polybench and converts its output to a JSON format')
    parser.add_argument('-o', '--output', required=True, help='path to output JSON file')
    parser.add_argument('rest', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    #print("output: ", args.output)
    rest = args.rest
    if rest[0] == '--':
        rest = rest[1:]

    benchmark_path = rest[0]
    benchmark_name = os.path.basename(benchmark_path)
    #print("benchmark name: ", benchmark_name)

    #print("Running: ", rest)
    proc = subprocess.run(rest, capture_output=True)
    if proc.returncode != 0:
        print("child exit status nonzero:\n")
        print("STDOUT: ", proc.stdout, "\n")
        print("STDERR: ", proc.stderr, "\n")
        sys.exit(proc.returncode)

    filter_content(proc.stdout, args.output, benchmark_name)
