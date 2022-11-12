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

from vecalign.dp_utils import make_alignment_types, print_alignments, read_alignments, \
    read_in_embeddings, make_doc_embedding, vecalign
from vecalign.score import score_multiple, log_final_scores
import vecalign.overlap as overlap
import vecalign.embeddings as embeddings

def generate_overlapping_and_embedding_files(overlapping_file, embedding_file, label, list_of_doc_paths, num_overlaps,
                                             model_st="LaBSE", gpu_batch_size=32, embeddings_storage_input=None, embeddings_storage_path=None,
                                             embeddings_storage_input_base64=False, storage_embeddings_are_uniq=True, dim=768,
                                             optimization_strategy=0, storage_embeddings_optimization_strategy=0):
    if (os.path.isfile(embedding_file) and not os.path.isfile(overlapping_file)):
        logging.warning('%s embedding file does exist but %s overlapping file does not: only overlapping file will be '
                       'generated, and likely the embedding file will not be compatible (this might lead to wrong results)',
                       label, label)

    # Generate overlapping files?
    if not os.path.isfile(overlapping_file):
        if not os.path.isfile(overlapping_file):
            # overlapping file does not exist -> generate
            logging.info("Generating %s overlapping file", label)

            overlap.overlap(overlapping_file, list_of_doc_paths, num_overlaps)

    # Generate embeddings?
    if not os.path.isfile(embedding_file):
        if not os.path.isfile(embedding_file):
            # embeddings do not exist -> generate
            logging.info("Generating %s embeddings", label)

            kwargs = {
                "gpu_batch_size": gpu_batch_size,
                "model_st": model_st,
                "storage_input_file": embeddings_storage_input,
                "storage_embedding_file": embeddings_storage_path,
                "storage_input_file_base64": embeddings_storage_input_base64,
                "dim": dim,
                "storage_embeddings_are_uniq": storage_embeddings_are_uniq,
                "optimization_strategy": optimization_strategy,
                "storage_embeddings_optimization_strategy": storage_embeddings_optimization_strategy,
                }

            embeddings.generate_embeddings(overlapping_file, embedding_file, **kwargs)

def process_docs_and_urls_files(src, tgt, src_urls=None, tgt_urls=None, src_metadata=None, tgt_metadata=None, read_from_stdin=False,
                                urls_format=False, metadata_format=False):
    src_urls_lines = None
    tgt_urls_lines = None
    generator = zip(
        open(src, 'rt', encoding="utf-8").readlines(),
        open(tgt, 'rt', encoding="utf-8").readlines(),
        open(src_urls, 'rt', encoding="utf-8").readlines(),
        open(tgt_urls, 'rt', encoding="utf-8").readlines(),
        open(src_metadata, 'rt', encoding="utf-8").readlines(),
        open(tgt_metadata, 'rt', encoding="utf-8").readlines(),
    ) if not read_from_stdin else sys.stdin

    for line in generator:
        if read_from_stdin:
            line = line.strip().split("\t")

            if urls_format and metadata_format:
                if len(line) != 6:
                    raise Exception('unexpected format when reading from stdin: expected format: src_doc_base64<tab>tgt_doc_base64<tab>src_url<tab>tgt_url<tab>src_metadata<tab>tgt_metadata')
            elif urls_format:
                if len(line) != 4:
                    raise Exception('unexpected format when reading from stdin: expected format: src_doc_base64<tab>tgt_doc_base64<tab>src_url<tab>tgt_url')
            elif metadata_format:
                if len(line) != 4:
                    raise Exception('unexpected format when reading from stdin: expected format: src_doc_base64<tab>tgt_doc_base64<tab>src_metadata<tab>tgt_metadata')
            elif len(line) != 2:
                raise Exception('unexpected format when reading from stdin: expected format: src_doc_base64<tab>tgt_doc_base64')

        src_lines = line[0]
        tgt_lines = line[1]

        # Decode src and tgt doc
        src_lines = base64.b64decode(src_lines).decode("utf-8").split("\n")
        tgt_lines = base64.b64decode(tgt_lines).decode("utf-8").split("\n")

        # Remove empty docs
        src_lines = list(filter(lambda l: len(l) != 0, map(lambda ll: ll.strip(), src_lines)))
        tgt_lines = list(filter(lambda l: len(l) != 0, map(lambda ll: ll.strip(), tgt_lines)))

        # Get URLs and metadata
        src_urls_lines = [line[2].strip()[:10000]] * len(src_lines) if urls_format else []
        tgt_urls_lines = [line[3].strip()[:10000]] * len(tgt_lines) if urls_format else []
        src_metadata_lines = list(map(lambda l: l.split('\t'), line[4 if urls_format else 2])) if metadata_format else []
        tgt_metadata_lines = list(map(lambda l: l.split('\t'), line[4 if urls_format else 2])) if metadata_format else []

        yield src_lines, tgt_lines, src_urls_lines, tgt_urls_lines, src_metadata_lines, tgt_metadata_lines

def _main():
    # make runs consistent
    seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser('Sentence alignment using sentence embeddings and FastDTW',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', '--src', type=str, nargs='*',
                        help='preprocessed source file to align. If "-" is provided, stdin will be used, also for --tgt (entries format: src_doc_base64<tab>tgt_doc_base64[<tab>src_url<tab>tgt_url][<tab>src_metadata<tab>tgt_metadata])')
    parser.add_argument('-t', '--tgt', type=str, nargs='*',
                        help='preprocessed target file to align. If "-" is provided, stdin will be used, also for --src (entries format: src_doc_base64<tab>tgt_doc_base64[<tab>src_url<tab>tgt_url][<tab>src_metadata<tab>tgt_metadata])')
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
                        help='URLs will be used for the results: src_URLs<tab>tgt_URLs<tab>src_sentences<tab>tgt_sentences[<tab>score][<tab>src_metadata_field1<tab>tgt_metadata_field1...]')
    parser.add_argument('--src_urls', type=str, nargs='*',
                        help='Source file of urls to print the results. If "-" is provided, stdin will be used, also for --tgt_urls (entries format: [src_doc_base64<tab>tgt_doc_base64<tab>]src_url<tab>tgt_url)')
    parser.add_argument('--tgt_urls', type=str, nargs='*',
                        help='Target file of urls to print the results. If "-" is provided, stdin will be used, also for --src_urls (entries format: [src_doc_base64<tab>tgt_doc_base64<tab>]src_url<tab>tgt_url)')
    parser.add_argument('--metadata_header_fields', type=str,
                        help='Provide language agnostic comma separated header fields if metadata is provided in the input. If provided, metadata will be processed')
    parser.add_argument('--src_metadata', type=str, nargs='*',
                        help='Source file of metadata to print the results. If "-" is provided, stdin will be used, also for --tgt_metadata (entries format: [src_doc_base64<tab>tgt_doc_base64<tab>][src_url<tab>tgt_url<tab>][src_metadata<tab>tgt_metadata])')
    parser.add_argument('--tgt_metadata', type=str, nargs='*',
                        help='Target file of metadata to print the results. If "-" is provided, stdin will be used, also for --src_metadata (entries format: [src_doc_base64<tab>tgt_doc_base64<tab>][src_url<tab>tgt_url<tab>][src_metadata<tab>tgt_metadata])')
    parser.add_argument('--read_from_stdin', action='store_true',
                        help='Read all provided input files from stdin: documents, URLs and metadata')

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
    parser.add_argument('--embeddings_src_storage_are_not_uniq', action='store_true',
                        help='Expected src storage embeddings are monotonic and uniq (i.e. embeddings from previous sentences are not expected to be provided). If the provided embeddings are 1-1 with the sentences, this flag must be set')
    parser.add_argument('--embeddings_tgt_storage_are_not_uniq', action='store_true',
                        help='Expected tgt storage embeddings are monotonic and uniq (i.e. embeddings from previous sentences are not expected to be provided). If the provided embeddings are 1-1 with the sentences, this flag must be set')
    parser.add_argument('--src_embeddings_optimization_strategy', type=int, default=0, choices=[0, 1, 2],
                        help='Optimization strategy applied to the embeddings when being generated. The generated embeddings will be stored applying the same strategy')
    parser.add_argument('--src_storage_embeddings_optimization_strategy', type=int, default=0, choices=[0, 1, 2],
                        help='Optimization strategy applied to the storage embeddings when being loaded')
    parser.add_argument('--tgt_embeddings_optimization_strategy', type=int, default=0, choices=[0, 1, 2],
                        help='Optimization strategy applied to the embeddings when being generated. The generated embeddings will be stored applying the same strategy')
    parser.add_argument('--tgt_storage_embeddings_optimization_strategy', type=int, default=0, choices=[0, 1, 2],
                        help='Optimization strategy applied to the storage embeddings when being loaded')

    args = parser.parse_args()

    # Logging
    logging_level = logging.WARNING

    if args.more_verbose:
        logging_level = logging.DEBUG
    elif args.verbose:
        logging_level = logging.INFO

    logging.basicConfig(handlers=[logging.StreamHandler()], level=logging_level,
                        format="%(asctime)s  %(levelname)-5.5s  %(message)s")

    # Main
    read_from_stdin = args.read_from_stdin
    urls_format = args.urls_format
    metadata_format = True if args.metadata_header_fields and urls_format else False

    if read_from_stdin:
        args.src, args.tgt = [], []
        args.src_urls, args.tgt_urls = [], []
        args.src_metadata, args.tgt_metadata = [], []
    elif not args.src or not args.tgt:
        raise Exception("You need to either provide src and tgt files or read input files from stdin")

    for embeddings_storage_input, embeddings_storage_path, label in zip([args.embeddings_src_storage_input, args.embeddings_tgt_storage_input],
                                                                        [args.embeddings_src_storage_path,  args.embeddings_tgt_storage_path],
                                                                        ["src",                             "tgt"]):
        if ((embeddings_storage_input is None) ^ (embeddings_storage_path is None)):
            logging.warning("--embeddings_%s_storage_input and --embeddings_%s_storage_path both need to be provided to take effect", label, label)

            if label == "src":
                args.embeddings_src_storage_input = None
                args.embeddings_src_storage_path = None
            else:
                args.embeddings_tgt_storage_input = None
                args.embeddings_tgt_storage_path = None

    if len(args.src) != len(args.tgt):
        raise Exception('number of source files must match number of target files')

    if args.gold_alignment is not None:
        if (not read_from_stdin and len(args.gold_alignment) != len(args.src)):
            raise Exception('number of gold alignment files, if provided, must match number of source and target files')

    # Checks about the URLs format
    if (not read_from_stdin and urls_format and (not args.src_urls or not args.tgt_urls)):
        raise Exception('if you use --urls_format, you need to provide --src_urls and --tgt_urls')
    if urls_format:
        if not read_from_stdin:
            if len(args.src) != len(args.src_urls) or len(args.tgt) != len(args.tgt_urls):
                raise Exception('number of files must match number of URLs files')

            for src_url, tgt_url in zip(args.src_urls, args.tgt_urls):
                if (not os.path.isfile(src_url) or not os.path.isfile(tgt_url)):
                    raise Exception('--src_urls and --tgt_urls must exist')
    # Checks about the metadata format
    if (not read_from_stdin and metadata_format and (args.src_metadata is None or args.tgt_metadata is None)):
        raise Exception('if you use --metadata_header_fields, you need to provide --src_metadata and --tgt_metadata')
    if metadata_format:
        if not read_from_stdin:
            if len(args.src) != len(args.src_metadata) or len(args.tgt) != len(args.tgt_metadata):
                raise Exception('number of files must match number of metadata files')

            for src_metadata, tgt_metadata in zip(args.src_metadata, args.tgt_metadata):
                if (not os.path.isfile(src_metadata) or not os.path.isfile(tgt_metadata)):
                    raise Exception('--src_metadata and --tgt_metadata must exist')

    if args.alignment_max_size < 2:
        logging.warning('alignment_max_size < 2: increasing to 2 so that 1-1 alignments will be considered')
        args.alignment_max_size = 2

    # Generate overlapping files and/or embeddings?
    src_gen_kwargs = {
        "model_st": args.embeddings_model,
        "gpu_batch_size": args.embeddings_batch_size,
        "embeddings_storage_input": args.embeddings_src_storage_input,
        "embeddings_storage_path": args.embeddings_src_storage_path,
        "embeddings_storage_input_base64": args.embeddings_src_storage_input_base64,
        "storage_embeddings_are_uniq": not args.embeddings_src_storage_are_not_uniq,
        "dim": args.embeddings_dim,
        "optimization_strategy": args.src_embeddings_optimization_strategy,
        "storage_embeddings_optimization_strategy": args.src_storage_embeddings_optimization_strategy,
        }
    tgt_gen_kwargs = {
        "model_st": args.embeddings_model,
        "gpu_batch_size": args.embeddings_batch_size,
        "embeddings_storage_input": args.embeddings_tgt_storage_input,
        "embeddings_storage_path": args.embeddings_tgt_storage_path,
        "embeddings_storage_input_base64": args.embeddings_tgt_storage_input_base64,
        "storage_embeddings_are_uniq": not args.embeddings_tgt_storage_are_not_uniq,
        "dim": args.embeddings_dim,
        "optimization_strategy": args.tgt_embeddings_optimization_strategy,
        "storage_embeddings_optimization_strategy": args.tgt_storage_embeddings_optimization_strategy,
        }

    generate_overlapping_and_embedding_files(args.src_embed[0], args.src_embed[1], "src", args.src, args.alignment_max_size, **src_gen_kwargs)
    generate_overlapping_and_embedding_files(args.tgt_embed[0], args.tgt_embed[1], "tgt", args.tgt, args.alignment_max_size, **tgt_gen_kwargs)

    del src_gen_kwargs, tgt_gen_kwargs

    # Load embeddings
    src_sent2line, src_line_embeddings = read_in_embeddings(args.src_embed[0], args.src_embed[1], dim=args.embeddings_dim, to_float32=False)
    tgt_sent2line, tgt_line_embeddings = read_in_embeddings(args.tgt_embed[0], args.tgt_embed[1], dim=args.embeddings_dim, to_float32=False)

    if args.src_embeddings_optimization_strategy:
        src_line_embeddings = embeddings.get_original_embedding_from_optimized(src_line_embeddings, strategy=args.src_embeddings_optimization_strategy)
    if args.tgt_embeddings_optimization_strategy:
        tgt_line_embeddings = embeddings.get_original_embedding_from_optimized(tgt_line_embeddings, strategy=args.tgt_embeddings_optimization_strategy)

    width_over2 = ceil(args.alignment_max_size / 2.0) + args.search_buffer_size

    test_alignments = []
    stack_list = []
    print_header = True

    # Process every pair of documents and, optionally, URLs
    for idx, (src_lines, tgt_lines, src_urls_lines, tgt_urls_lines, src_meta_lines, tgt_meta_lines) \
            in enumerate(process_docs_and_urls_files(args.src, args.tgt,
                                                     src_urls=args.src_urls, tgt_urls=args.tgt_urls,
                                                     src_metadata=args.src_metadata, tgt_metadata=args.tgt_metadata,
                                                     read_from_stdin=read_from_stdin,
                                                     urls_format=urls_format, metadata_format=metadata_format)):
        if read_from_stdin:
            logging.info('Aligning documents pair #%d', idx)
        else:
            logging.info('Aligning src="%s" to tgt="%s"', args.src[idx], args.tgt[idx])

        vecs0 = make_doc_embedding(src_sent2line, src_line_embeddings, src_lines, args.alignment_max_size)
        vecs1 = make_doc_embedding(tgt_sent2line, tgt_line_embeddings, tgt_lines, args.alignment_max_size)

        final_alignment_types = make_alignment_types(args.alignment_max_size)
        logging.debug('Considering alignment types %s', final_alignment_types)

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
        if urls_format:
            # check that we have the expected number of lines
            if len(src_urls_lines) != len(src_lines):
                raise Exception(f'different number of lines in src lines and urls: {len(src_lines)} vs {len(src_urls_lines)} (idx {idx})')
            if len(tgt_urls_lines) != len(tgt_lines):
                raise Exception(f'different number of lines in tgt lines and urls: {len(tgt_lines)} vs {len(tgt_urls_lines)} (idx {idx})')
        if metadata_format:
            # check that we have the expected number of lines
            if len(src_meta_lines) != len(src_lines):
                raise Exception(f'different number of lines in src lines and metadata: {len(src_lines)} vs {len(src_meta_lines)} (idx {idx})')
            if len(tgt_meta_lines) != len(tgt_lines):
                raise Exception(f'different number of lines in tgt lines and metadata: {len(tgt_lines)} vs {len(tgt_meta_lines)} (idx {idx})')

        # write final alignments to stdout
        print_alignments(stack[0]['final_alignments'], stack[0]['alignment_scores'], threshold=args.threshold,
                         urls_format=urls_format, src_lines=src_lines, tgt_lines=tgt_lines,
                         src_urls=src_urls_lines, tgt_urls=tgt_urls_lines, doc_idx=idx,
                         src_metadata=src_meta_lines, tgt_metadata=src_meta_lines,
                         metadata_header_fields=args.metadata_header_fields, print_header=print_header)

        print_header = False

        test_alignments.append(stack[0]['final_alignments'])
        stack_list.append(stack)

    if args.gold_alignment is not None:
        if (read_from_stdin and idx + 1 != len(args.gold_alignment)):
            raise Exception('number of gold alignment files, if provided, must match number of source and target files')

        gold_list = [read_alignments(x) for x in args.gold_alignment]
        res = score_multiple(gold_list=gold_list, test_list=test_alignments)
        log_final_scores(res)

    if args.debug_save_stack:
        pickle.dump(stack_list, open(args.debug_save_stack, 'wb'))


if __name__ == '__main__':
    _main()
