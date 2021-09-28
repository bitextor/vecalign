#!/usr/bin/env python3

import os
import sys
import logging
import argparse

from sentence_transformers import SentenceTransformer, util
import numpy as np

from dp_utils import read_in_embeddings

if __name__ != "__main__":
    logging = logging.getLogger('vecalign')

def generate_embeddings(input_file, output_file, gpu_batch_size=32, batch_size=None, model_st="LaBSE",
                        storage_input_file=None, storage_embedding_file=None, dim=768):
    input_fd = open(input_file)
    output_fd = open(output_file, "wb")
    batch_size = max(1, batch_size) if batch_size is not None else None
    model = SentenceTransformer(model_st)
    storage_sent2line = {}
    storage_embeddings = []
    lines_already_processed = set()

    if (storage_input_file is not None and storage_embedding_file is not None):
        # Load lines and embeddings from storage
        storage_sent2line, storage_embeddings = read_in_embeddings(storage_input_file, storage_embedding_file,
                                                                   dim=dim, exception_when_dup=False)

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

        embeddings = np.zeros((0, dim)).astype(np.float32)

        # Process embeddings
        if len(lines) != 0:
            embeddings = model.encode(lines, batch_size=gpu_batch_size, show_progress_bar=False)

        if len(storage_idxs) != 0:
            # Add embeddings from the storage
            for position, line_idx in storage_idxs:
                embedding = storage_embeddings[line_idx]
                embeddings = np.insert(embeddings, position, embedding, axis=0).astype(np.float32)

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
    parser.add_argument('--storage-embedding-file', type=str,
                        help='Path to storage with embeddings')

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
              "dim": args.dim,}

    generate_embeddings(args.input_file, args.output_file, **kwargs)
