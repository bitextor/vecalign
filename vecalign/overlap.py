#!/usr/bin/env python3

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import base64
import argparse

from vecalign.dp_utils import yield_overlaps


def overlap(output_file, input_files, num_overlaps, paragraphs=False):
    output = set()
    is_stdin = input_files[0] == "-"

    for fin in (["aux"] if is_stdin else ([input_files] if not isinstance(input_files, list) else input_files)):
        generator = sys.stdin if is_stdin else open(fin, 'rt', encoding="utf-8").readlines()

        for lines in generator:
            lines = lines.strip()
            lines = base64.b64decode(lines).decode("utf-8").split("\n")
            lines = list(map(lambda l: l.split('\t')[0], lines)) if paragraphs else lines # Remove paragraphs
            lines = list(filter(lambda l: len(l) != 0, map(lambda ll: ll.strip(), lines)))

            for out_line in yield_overlaps(lines, num_overlaps):
                output.add(out_line)

    # for reproducibility
    output = list(output)
    output.sort()

    if output_file is None:
        pass
    elif output_file == "-":
        for line in output:
            print(line)
    else:
        with open(output_file, 'wt', encoding="utf-8") as fout:
            for line in output:
                fout.write(line + '\n')

    return output


def _main():
    parser = argparse.ArgumentParser('Create text file containing overlapping sentences.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--inputs', type=str, nargs='+', required=True,
                        help='input text file(s). If "-" is provided, stdin will be used (entries format: doc_base64)')

    parser.add_argument('-o', '--output', type=str, default=None,
                        help='output text file containing overlapping sentences. If "-" is provided, stdout will be used')

    parser.add_argument('-n', '--num_overlaps', type=int, default=4,
                        help='Maximum number of allowed overlaps.')

    parser.add_argument('--paragraph_identification', action='store_true',
                        help='Enable paragraph identification. Identifier is expected to be in the 2nd column of the provided documents')

    args = parser.parse_args()

    if args.inputs[0] == "-":
        # Remove extra args
        args.inputs = args.inputs[:1]

    overlap(output_file=args.output,
            num_overlaps=args.num_overlaps,
            input_files=args.inputs,
            paragraphs=args.paragraph_identification)


if __name__ == '__main__':
    _main()
