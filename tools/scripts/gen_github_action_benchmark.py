#!/usr/bin/python3

# generates a JSON file with results

# input: a file with one JSON hash per line,
# output: JSON header + input file + closing braces

import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog='gen_google_benchmark_json',
                        description='generates a google benchmark JSON file with input JSON data')
    parser.add_argument('-i', '--input', required=True, help='Path to input JSON file')
    parser.add_argument('-o', '--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    content = ""
    with open(args.input) as my_file:
        content = my_file.read()

    input = content.strip()
    if input[-1] == ',':
        input = input[0:-1]

    f = open(args.output, 'w')

    header = "["
    footer = "]"
    f.write(header)
    f.write(input)
    f.write(footer)
