
import logging

from sentence_transformers import SentenceTransformer, util
import numpy as np

logger = logging.getLogger('vecalign')  # set up in vecalign.py

def generate_embeddings(input_file, output_file, gpu_batch_size=32, batch_size=None, model_st="LaBSE"):
    input_fd = open(input_file)
    output_fd = open(output_file, "wb")
    batch_size = min(1, batch_size) if batch_size is not None else None
    model = SentenceTransformer(model_st)

    while not input_fd.closed:
        stop = False
        lines = []

        # Read sentences (batch)
        for idx, line in enumerate(input_fd):
            line = line.strip()[:10000]
            lines.append(line)

            if (batch_size is not None and idx >= batch_size):
                stop = True
                break

        # Process embeddings
        if len(lines) != 0:
            embeddings = model.encode(lines, batch_size=gpu_batch_size, show_progress_bar=False)

            np.save(output_fd, embeddings)
        # Have we finished?
        if not stop:
            input_fd.close()

    output_fd.close()
