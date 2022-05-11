#!/usr/bin/env python3

import os
import sys
import logging
import argparse

from sentence_transformers import SentenceTransformer, util
import numpy as np

from vecalign.dp_utils import read_in_embeddings

_STRATEGY_2_BITS = 8
_STRATEGY_2_BINS = (np.array(range(2 ** _STRATEGY_2_BITS - 1), dtype=np.float32) - (2 ** _STRATEGY_2_BITS - 1) // 2) / ((2 ** _STRATEGY_2_BITS) / 2)
_STRATEGY_2_BINS_RECOVER = (np.array(range(2 ** _STRATEGY_2_BITS), dtype=np.float32) - (2 ** _STRATEGY_2_BITS - 1) // 2) / ((2 ** _STRATEGY_2_BITS) / 2)

_OPTIMIZATION_STRATEGY_DTYPE = {
    None: np.float32,
    0: np.float32,
    1: np.float16,
    2: np.uint8,
}

_OPTIMIZATION_STRATEGIES = (1, 2)

def get_optimized_embedding(embedding, strategy=1):
    if strategy not in _OPTIMIZATION_STRATEGIES:
        logging.warning(f"Unknown optimization strategy ({strategy}): returning embedding without any optimization strategy applied")

        return embedding

    x = embedding.copy()

    # Apply strategy to optimize
    if strategy == 1:
        x = x.astype(np.float16, copy=False)
    elif strategy == 2:
        # linear quantization (range [-1., 1.])
        x = np.digitize(x, _STRATEGY_2_BINS).astype(np.uint8)

    return x

def get_original_embedding_from_optimized(embedding, strategy=1, to_float32=True):
    if strategy not in _OPTIMIZATION_STRATEGIES:
        logging.warning(f"Unknown optimization strategy ({strategy}): returning original embedding")

        return embedding

    x = embedding.copy()

    # Strategies have to work with shape length 1 and 2
    if strategy == 1:
        x = x
    elif strategy == 2:
        # vector quantization (range [-1., 1.])
        x = _STRATEGY_2_BINS_RECOVER[x]

    if to_float32:
        x = x.astype(np.float32)

    return x

def generate_embeddings(input_file, output_file, gpu_batch_size=32, batch_size=None, model_st="LaBSE",
                        storage_input_file=None, storage_embedding_file=None, storage_input_file_base64=False,
                        storage_embeddings_are_uniq=True, dim=768, optimization_strategy=0,
                        storage_embeddings_optimization_strategy=0):
    input_fd = open(input_file)
    output_fd = open(output_file, "wb")
    batch_size = max(1, batch_size) if batch_size is not None else None
    model = SentenceTransformer(model_st)
    storage_sent2line = {}
    storage_embeddings = []
    lines_already_processed = set()

    if (optimization_strategy != 0 and optimization_strategy not in _OPTIMIZATION_STRATEGIES):
        logging.warning("Unknown optimization strategy: %s -> 0", optimization_strategy)

        optimization_strategy = 0
    elif optimization_strategy in _OPTIMIZATION_STRATEGIES:
        logging.info("Optimization strategy: %d", optimization_strategy)
    if (storage_embeddings_optimization_strategy != 0 and storage_embeddings_optimization_strategy not in _OPTIMIZATION_STRATEGIES):
        logging.warning("Unknown optimization strategy (storage): %s -> 0", storage_embeddings_optimization_strategy)

        storage_embeddings_optimization_strategy = 0
    elif storage_embeddings_optimization_strategy in _OPTIMIZATION_STRATEGIES:
        logging.info("Optimization strategy (storage): %d", storage_embeddings_optimization_strategy)

    if (storage_input_file is not None and storage_embedding_file is not None):
        # Load lines and embeddings from storage
        storage_sent2line, storage_embeddings = read_in_embeddings(storage_input_file, storage_embedding_file,
                                                                   dim=dim, exception_when_dup=False, to_float32=False,
                                                                   decode_text_base64=storage_input_file_base64,
                                                                   embed_file_uniq=storage_embeddings_are_uniq)

    # Embeddings optimization (storage)
    if (storage_embeddings_optimization_strategy and len(storage_embeddings) > 0):
        if optimization_strategy != storage_embeddings_optimization_strategy:
            storage_embeddings = get_original_embedding_from_optimized(storage_embeddings, strategy=storage_embeddings_optimization_strategy)

            # Are storage embeddings compatibles with the new embeddings?
            if optimization_strategy:
                # Make the storage embeddings to be compatible with the new embeddings
                storage_embeddings = get_optimized_embedding(storage_embeddings, strategy=optimization_strategy)

    while not input_fd.closed:
        stop = False
        lines = []
        storage_idxs = []
        skipped_lines = 0

        # Read sentences (batch)
        for idx, line in enumerate(input_fd):
            line = line.strip()
            line_hash = hash(line)

            if line_hash in lines_already_processed:
                skipped_lines += 1
                logging.warning("line '%s' had been already processed", line)

                continue
            else:
                lines_already_processed.add(line_hash)

            if line in storage_sent2line.keys():
                storage_idxs.append((idx, storage_sent2line[line]))
            else:
                lines.append(line)

            if (batch_size is not None and idx >= batch_size + len(storage_idxs) + skipped_lines):
                stop = True
                break

        logging.debug("%d lines and %d embeddings from storage", len(lines), len(storage_idxs))

        embeddings = np.zeros((0, dim)).astype(_OPTIMIZATION_STRATEGY_DTYPE[optimization_strategy])

        # Process embeddings
        if len(lines) != 0:
            embeddings = model.encode(lines, batch_size=gpu_batch_size, show_progress_bar=False)

            # Embeddings optimization
            if optimization_strategy:
                embeddings = get_optimized_embedding(embeddings, strategy=optimization_strategy)

        if len(storage_idxs) != 0:
            # Add embeddings from the storage
            for position, line_idx in storage_idxs:
                embedding = storage_embeddings[line_idx]
                embeddings = np.insert(embeddings, position, embedding, axis=0).astype(_OPTIMIZATION_STRATEGY_DTYPE[optimization_strategy])

#                logging.debug("embedding loaded from storage: position %d and idx %d", position, line_idx)

        if embeddings.shape[0] > 0:
            np.save(output_fd, embeddings)

        # Have we finished?
        if not stop:
            input_fd.close()

    output_fd.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate embeddings',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_file', type=str,
                        help='Input file')
    parser.add_argument('output_file', type=str,
                        help='Output file')

    parser.add_argument('--dim', type=int, default=768,
                        help='Dim. of embeddings')
    parser.add_argument('--gpu-batch-size', type=int, default=32,
                        help='GPU batch size which will be used by SentenceTransformers')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size which will be used by SentenceTransformers')
    parser.add_argument('--model', type=str, default="LaBSE",
                        help='Model to use by SentenceTransformers')
    parser.add_argument('--storage-input-file', type=str,
                        help='Path to storage with sentences')
    parser.add_argument('--storage-input-file-base64', action='store_true',
                        help='Storage with sentences are base64 encoded')
    parser.add_argument('--storage-embedding-file', type=str,
                        help='Path to storage with embeddings')
    parser.add_argument('--storage-embedding-are-uniq', action='store_true',
                        help='Storage embeddings are expected to be monotonic (with input_file) and unique. If your embeddings are not unique but there is an embedding per line, this option must be set')
    parser.add_argument('--embedding-optimization', type=int, default=0, choices=[0, 1, 2],
                        help='Embedding optimization')
    parser.add_argument('--storage-embedding-optimization', type=int, default=0, choices=[0, 1, 2],
                        help='Embedding optimization (storage)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if not os.path.isfile(args.input_file):
        logging.error("Input file does not exist")

        sys.exit(1)
    if os.path.isfile(args.output_file):
        logging.error("Output file already exists")

        sys.exit(1)

    kwargs = {"gpu_batch_size": args.gpu_batch_size,
              "batch_size": args.batch_size,
              "model_st": args.model,
              "storage_input_file": args.storage_input_file,
              "storage_embedding_file": args.storage_embedding_file,
              "storage_input_file_base64": args.storage_input_file_base64,
              "storage_embeddings_are_uniq": args.storage_embedding_are_uniq,
              "dim": args.dim,
              "optimization_strategy": args.embedding_optimization,
              "storage_embeddings_optimization_strategy": args.storage_embedding_optimization,}

    generate_embeddings(args.input_file, args.output_file, **kwargs)
