#!/usr/bin/python3
from collections import defaultdict
import sys

import h5py
import numcodecs
import numpy as np
import tqdm
import zarr


from encoder.replay_encoder import CHUNK_SIZE, CHUNK_SHAPES, DEFAULT_ENCODER, ENCODERS

numcodecs.blosc.set_nthreads(5)
BATCH_SIZE = 2048


def main(input_paths, out_path):
    in_files = [h5py.File(in_path) for in_path in input_paths]
    in_file_sizes = [0] * len(in_files)
    out_file = zarr.open(out_path, mode='a')

    datasets = list(in_files[0].keys())
    datasets_per_file = defaultdict(dict)
    for in_path, in_file in enumerate(in_files):
        in_file_sizes[in_path] = len(in_file[datasets[0]])
        for dataset in datasets:
            datasets_per_file[in_path][dataset] = in_file[dataset]

    for dataset in datasets:
        current_infile = datasets_per_file[0][dataset]
        args = dict(
            name=dataset,
            shape=current_infile.shape,
            chunks=CHUNK_SHAPES.get(dataset, (CHUNK_SIZE,) + current_infile.shape[1:]),
            dtype=current_infile.dtype,
        )
        args.update(ENCODERS.get(dataset, DEFAULT_ENCODER))
        print("Dataset %s: %s" % (dataset, args))
        out_set = out_file.create_dataset(**args)

        for in_path in range(len(input_paths)):
            print("In-file %s..." % input_paths[in_path])
            current_infile = datasets_per_file[in_path][dataset]

            num_batches = int(np.ceil(len(current_infile) / BATCH_SIZE))
            for index_batch in tqdm.tqdm(range(num_batches)):
                start = index_batch * BATCH_SIZE
                end = min(len(current_infile), (index_batch + 1) * BATCH_SIZE)

                out_set[start:end] = current_infile[start:end]
    out_file.store.close()
    return


if __name__ == "__main__":
    in_files = sys.argv[1:]
    main(in_files[:-1], in_files[-1])