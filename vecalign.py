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

import argparse
import logging
import pickle
from math import ceil
from random import seed as seed
import os
import sys
import base64

import numpy as np

logger = logging.getLogger('vecalign')
logger.setLevel(logging.WARNING)
logFormatter = logging.Formatter("%(asctime)s  %(levelname)-5.5s  %(message)s")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

from dp_utils import make_alignment_types, print_alignments, read_alignments, \
    read_in_embeddings, make_doc_embedding, vecalign
from score import score_multiple, log_final_scores
import overlap
import embeddings

def generate_overlapping_and_embedding_files(overlapping_file, embedding_file, label, list_of_doc_paths, num_overlaps,
                                             model_st="LaBSE", gpu_batch_size=32, embeddings_storage_input=None, embeddings_storage_path=None,
                                             embeddings_storage_input_base64=False, dim=768):
    if (os.path.isfile(embedding_file) and not os.path.isfile(overlapping_file)):
        logger.warning('%s embedding file does exist but %s overlapping file does not: only overlapping file will be '
                       'generated, and likely the embedding file will not be compatible (this might lead to wrong results)',
                       label, label)

    # Generate overlapping files?
    if not os.path.isfile(overlapping_file):
        if not os.path.isfile(overlapping_file):
            # overlapping file does not exist -> generate
            logger.info("Generating %s overlapping file", label)

            overlap.overlap(overlapping_file, list_of_doc_paths, num_overlaps)

    # Generate embeddings?
    if not os.path.isfile(embedding_file):
        if not os.path.isfile(embedding_file):
            # embeddings do not exist -> generate
            logger.info("Generating %s embeddings", label)

            embeddings.generate_embeddings(overlapping_file, embedding_file, gpu_batch_size=gpu_batch_size, model_st=model_st,
                                           storage_input_file=embeddings_storage_input, storage_embedding_file=embeddings_storage_path,
                                           storage_input_file_base64=embeddings_storage_input_base64, dim=dim)

def process_docs_and_urls_files(src, tgt, src_urls, tgt_urls):
    src_urls_lines = None
    tgt_urls_lines = None

    if (src[0] == "-" and src_urls[0] == "-"):
        for idx, line in enumerate(sys.stdin):
            line = line.strip().split("\t")

            if len(line) != 4:
                raise Exception('unexpected format when reading from stdin: expected format is src_doc_base64<tab>tgt_doc_base64<tab>src_url<tab>tgt_url')

            src_lines = base64.b64decode(line[0]).decode("utf-8").split("\n")
            src_lines = list(filter(lambda l: len(l) != 0, map(lambda ll: ll.strip(), src_lines)))
            tgt_lines = base64.b64decode(line[1]).decode("utf-8").split("\n")
            tgt_lines = list(filter(lambda l: len(l) != 0, map(lambda ll: ll.strip(), tgt_lines)))

            src_urls_lines = [line[2].strip()] * len(src_lines)
            tgt_urls_lines = [line[3].strip()] * len(tgt_lines)

            yield src_lines, tgt_lines, src_urls_lines, tgt_urls_lines
    elif src[0] == "-":
        for idx, line in enumerate(sys.stdin):
            line = line.strip().split("\t")

            if len(line) != 2:
                raise Exception('unexpected format when reading from stdin: expected format is src_doc_base64<tab>tgt_doc_base64')

            src_lines = base64.b64decode(line[0]).decode("utf-8").split("\n")
            src_lines = list(filter(lambda l: len(l) != 0, map(lambda ll: ll.strip(), src_lines)))
            tgt_lines = base64.b64decode(line[1]).decode("utf-8").split("\n")
            tgt_lines = list(filter(lambda l: len(l) != 0, map(lambda ll: ll.strip(), tgt_lines)))

            src_urls_lines = list(map(lambda line: line.strip()[:10000], open(src_urls[idx], encoding="utf-8").readlines()))
            tgt_urls_lines = list(map(lambda line: line.strip()[:10000], open(tgt_urls[idx], encoding="utf-8").readlines()))

            yield src_lines, tgt_lines, src_urls_lines, tgt_urls_lines
    elif src_urls[0] == "-":
        for idx, line in enumerate(sys.stdin):
            line = line.strip().split("\t")

            if len(line) != 2:
                raise Exception('unexpected format when reading from stdin: expected format is src_url<tab>tgt_url')

            src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
            tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
            src_urls_lines = [line[0].strip()] * len(src_lines)
            tgt_urls_lines = [line[1].strip()] * len(tgt_lines)

            yield src_lines, tgt_lines, src_urls_lines, tgt_urls_lines
    else:
        for src_file, tgt_file, src_urls_file, tgt_urls_file in zip(src, tgt, src_urls, tgt_urls):
            src_lines = open(src_file, 'rt', encoding="utf-8").readlines()
            tgt_lines = open(tgt_file, 'rt', encoding="utf-8").readlines()
            src_urls_lines = list(map(lambda line: line.strip()[:10000], open(src_urls_file, encoding="utf-8").readlines()))
            tgt_urls_lines = list(map(lambda line: line.strip()[:10000], open(tgt_urls_file, encoding="utf-8").readlines()))

            yield src_lines, tgt_lines, src_urls_lines, tgt_urls_lines

def _main():
    # make runs consistent
    seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser('Sentence alignment using sentence embeddings and FastDTW',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', '--src', type=str, nargs='+', required=True,
                        help='preprocessed source file to align. If "-" is provided, stdin will be used, also for --tgt (entries format: src_doc_base64<tab>tgt_doc_base64[<tab>src_url<tab>tgt_url])')
    parser.add_argument('-t', '--tgt', type=str, nargs='+', required=True,
                        help='preprocessed target file to align. If "-" is provided, stdin will be used, also for --src (entries format: src_doc_base64<tab>tgt_doc_base64[<tab>src_url<tab>tgt_url])')
    parser.add_argument('-g', '--gold_alignment', type=str, nargs='+', required=False,
                        help='preprocessed target file to align')
    parser.add_argument('--src_embed', type=str, nargs=2, required=True,
                        help='Source embeddings. Requires two arguments: first is a text file, sencond is a binary embeddings file. ')
    parser.add_argument('--tgt_embed', type=str, nargs=2, required=True,
                        help='Target embeddings. Requires two arguments: first is a text file, sencond is a binary embeddings file. ')
    parser.add_argument('-a', '--alignment_max_size', type=int, default=4,
                        help='Searches for alignments up to size N-M, where N+M <= this value. Note that the the embeddings must support the requested number of overlaps')
    parser.add_argument('-d', '--del_percentile_frac', type=float, default=0.2,
                        help='Deletion penalty is set to this percentile (as a fraction) of the cost matrix distribution. Should be between 0 and 1.')
    parser.add_argument('-v', '--verbose', help='sets console to logging.INFO instead of logging.WARNING',
                        action='store_true')
    parser.add_argument('-vv', '--more_verbose', help='sets console to logging.DEBUG instead of logging.WARNING',
                        action='store_true')
    parser.add_argument('--max_size_full_dp', type=int, default=300,
                        help='Maximum size N for which is is acceptable to run full N^2 dynamic programming.')
    parser.add_argument('--costs_sample_size', type=int, default=20000,
                        help='Sample size to estimate costs distribution, used to set deletion penalty in conjunction with deletion_percentile.')
    parser.add_argument('--num_samps_for_norm', type=int, default=100,
                        help='Number of samples used for normalizing embeddings')
    parser.add_argument('--search_buffer_size', type=int, default=5,
                        help='Width (one side) of search buffer. Larger values makes search more likely to recover from errors but increases runtime.')
    parser.add_argument('--debug_save_stack', type=str,
                        help='Write stack to pickle file for debug purposes')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold which will be applied to the obtained scores of the alignment. All matches whose score is lower than the provided threshold will be discarded. The threshold is only applied to the printed results (it is not applied to the evaluation)')
    parser.add_argument('--urls_format', action='store_true',
                        help='URLs will be used for the results: src_URLs<tab>tgt_URLs<tab>src_sentences<tab>tgt_sentences[<tab>score]')
    parser.add_argument('--src_urls', type=str, nargs='+',
                        help='Source file of urls to print the results. If "-" is provided, stdin will be used, also for --tgt-urls (entries format: [src_doc_base64<tab>tgt_doc_base64<tab>]src_url<tab>tgt_url)')
    parser.add_argument('--tgt_urls', type=str, nargs='+',
                        help='Target file of urls to print the results. If "-" is provided, stdin will be used, also for --src-urls (entries format: [src_doc_base64<tab>tgt_doc_base64<tab>]src_url<tab>tgt_url)')

    # Embeddings
    parser.add_argument('--embeddings_dim', type=int, default=768,
                        help='Dimension of the embeddings. The default value is 768')
    parser.add_argument('--embeddings_batch_size', type=int, default=32,
                        help='Batch size for GPU when generating embeddings. The default value is 32')
    parser.add_argument('--embeddings_model', type=str, default="LaBSE",
                        help='Model which will be used to generate the embeddings with sentence_transformers. The default value is LaBSE')
    parser.add_argument('--embeddings_src_storage_input', type=str,
                        help='Path to the src storage file which contains sentences. You will need to provide --embeddings_storage_path as well')
    parser.add_argument('--embeddings_src_storage_input_base64', action='store_true',
                        help='Sentences provided via --embeddings_src_storage_input are base64 encoded')
    parser.add_argument('--embeddings_src_storage_path', type=str,
                        help='Path to the src storage file which contains embeddings. You will need to provide --embeddings_storage_input as well')
    parser.add_argument('--embeddings_tgt_storage_input', type=str,
                        help='Path to the tgt storage file which contains sentences. You will need to provide --embeddings_storage_path as well')
    parser.add_argument('--embeddings_tgt_storage_input_base64', action='store_true',
                        help='Sentences provided via --embeddings_tgt_storage_input are base64 encoded')
    parser.add_argument('--embeddings_tgt_storage_path', type=str,
                        help='Path to the tgt storage file which contains embeddings. You will need to provide --embeddings_storage_input as well')

    args = parser.parse_args()

    docs_provided_via_stdin = False
    urls_provided_via_stdin = False

    for embeddings_storage_input, embeddings_storage_path, label in zip([args.embeddings_src_storage_input, args.embeddings_tgt_storage_input],
                                                                        [args.embeddings_src_storage_path,  args.embeddings_tgt_storage_path],
                                                                        ["src",                             "tgt"]):
        if ((embeddings_storage_input is None) ^ (embeddings_storage_path is None)):
            logger.warning("--embeddings_%s_storage_input and --embeddings_%s_storage_path both need to be provided to take effect", label, label)

            if label == "src":
                args.embeddings_src_storage_input = None
                args.embeddings_src_storage_path = None
            else:
                args.embeddings_tgt_storage_input = None
                args.embeddings_tgt_storage_path = None
    if (args.src[0] == "-" and args.tgt[0] == "-"):
        # Docs are going to ve provided via stdin
        logger.info('reading documents from stdin')

        docs_provided_via_stdin = True

        # Remove extra args
        args.src = args.src[:1]
        args.tgt = args.tgt[:1]

        if (not os.path.isfile(args.src_embed[0]) or not os.path.isfile(args.tgt_embed[0])):
            raise Exception('if --src and --tgt are going to be provided via stdin, src and tgt overlapping files must exist')
    if (args.src_urls is not None and args.tgt_urls is not None and args.src_urls[0] == "-" and args.tgt_urls[0] == "-"):
        # URLs are going to ve provided via stdin
        logger.info('Reading URLs from stdin')

        urls_provided_via_stdin = True

        # Remove extra args
        args.src_urls = args.src_urls[:1]
        args.tgt_urls = args.tgt_urls[:1]

    if len(args.src) != len(args.tgt):
        raise Exception('number of source files must match number of target files')

    if args.gold_alignment is not None:
        if (not docs_provided_via_stdin and len(args.gold_alignment) != len(args.src)):
            raise Exception('number of gold alignment files, if provided, must match number of source and target files')

    # Checks about the URLs format
    if (args.urls_format and (args.src_urls is None or args.tgt_urls is None)):
        raise Exception('if you use --urls_format, you need to provide --src_urls and --tgt_urls')
    if args.urls_format:
        if (not docs_provided_via_stdin and not urls_provided_via_stdin and (len(args.src) != len(args.src_urls) or len(args.tgt) != len(args.tgt_urls))):
            raise Exception('number of files must match number of URLs files')

        if not urls_provided_via_stdin:
            for src_url, tgt_url in zip(args.src_urls, args.tgt_urls):
                if (not os.path.isfile(src_url) or not os.path.isfile(tgt_url)):
                    raise Exception('--src_urls and --tgt_urls must exist')
    else:
        args.src_urls, args.tgt_urls = None, None

    if args.more_verbose:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)

    if args.alignment_max_size < 2:
        logger.warning('alignment_max_size < 2: increasing to 2 so that 1-1 alignments will be considered')
        args.alignment_max_size = 2

    # Generate overlapping files and/or embeddings?
    generate_overlapping_and_embedding_files(args.src_embed[0], args.src_embed[1], "src", args.src, args.alignment_max_size,
                                             model_st=args.embeddings_model, gpu_batch_size=args.embeddings_batch_size,
                                             embeddings_storage_input=args.embeddings_src_storage_input,
                                             embeddings_storage_path=args.embeddings_src_storage_path,
                                             embeddings_storage_input_base64=args.embeddings_src_storage_input_base64,
                                             dim=args.embeddings_dim)
    generate_overlapping_and_embedding_files(args.tgt_embed[0], args.tgt_embed[1], "tgt", args.tgt, args.alignment_max_size,
                                             model_st=args.embeddings_model, gpu_batch_size=args.embeddings_batch_size,
                                             embeddings_storage_input=args.embeddings_tgt_storage_input,
                                             embeddings_storage_path=args.embeddings_tgt_storage_path,
                                             embeddings_storage_input_base64=args.embeddings_tgt_storage_input_base64,
                                             dim=args.embeddings_dim)


    # Load embeddings
    src_sent2line, src_line_embeddings = read_in_embeddings(args.src_embed[0], args.src_embed[1], dim=args.embeddings_dim)
    tgt_sent2line, tgt_line_embeddings = read_in_embeddings(args.tgt_embed[0], args.tgt_embed[1], dim=args.embeddings_dim)

    width_over2 = ceil(args.alignment_max_size / 2.0) + args.search_buffer_size

    test_alignments = []
    stack_list = []

    # Process every pair of documents and, optionally, URLs
    for idx, (src_lines, tgt_lines, src_urls_lines, tgt_urls_lines) in enumerate(process_docs_and_urls_files(args.src, args.tgt, args.src_urls, args.tgt_urls)):
        if docs_provided_via_stdin:
            logger.info('Aligning documents pair #%d', idx)
        else:
            logger.info('Aligning src="%s" to tgt="%s"', args.src[idx], args.tgt[idx])

        vecs0 = make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, args.alignment_max_size)
        vecs1 = make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, args.alignment_max_size)

        final_alignment_types = make_alignment_types(args.alignment_max_size)
        logger.debug('Considering alignment types %s', final_alignment_types)

        # Sentence alignment
        stack = vecalign(vecs0=vecs0,
                         vecs1=vecs1,
                         final_alignment_types=final_alignment_types,
                         del_percentile_frac=args.del_percentile_frac,
                         width_over2=width_over2,
                         max_size_full_dp=args.max_size_full_dp,
                         costs_sample_size=args.costs_sample_size,
                         num_samps_for_norm=args.num_samps_for_norm)

        # URLs format
        if args.urls_format:
            if (docs_provided_via_stdin and not urls_provided_via_stdin):
                if (idx >= len(args.src_urls) or idx >= len(args.tgt_urls)):
                    raise Exception('number of files must match number of URLs files')

            # check that we have the expected number of lines
            if len(src_urls_lines) != len(src_lines):
                raise Exception(f'different number of lines in src lines and urls: {len(src_lines)} vs {len(src_urls_lines)} (idx {idx})')
            if len(tgt_urls_lines) != len(tgt_lines):
                raise Exception(f'different number of lines in tgt lines and urls: {len(tgt_lines)} vs {len(tgt_urls_lines)} (idx {idx})')

        # write final alignments to stdout
        print_alignments(stack[0]['final_alignments'], stack[0]['alignment_scores'], threshold=args.threshold,
                         urls_format=args.urls_format, src_lines=src_lines, tgt_lines=tgt_lines,
                         src_urls=src_urls_lines, tgt_urls=tgt_urls_lines, doc_idx=idx)

        test_alignments.append(stack[0]['final_alignments'])
        stack_list.append(stack)

    if args.gold_alignment is not None:
        if (docs_provided_via_stdin and idx + 1 != len(args.gold_alignment)):
            raise Exception('number of gold alignment files, if provided, must match number of source and target files')

        gold_list = [read_alignments(x) for x in args.gold_alignment]
        res = score_multiple(gold_list=gold_list, test_list=test_alignments)
        log_final_scores(res)

    if args.debug_save_stack:
        pickle.dump(stack_list, open(args.debug_save_stack, 'wb'))


if __name__ == '__main__':
    _main()
