from collections import defaultdict
import sys

import h5py
import numpy as np
import tqdm

from encoder.replay_encoder import BatchExporter, ENCODER_DTYPE


BATCH_SIZE = 1024


def main(input_paths, out_path):
    in_files = [h5py.File(in_path) for in_path in input_paths]
    in_file_sizes = [0] * len(in_files)
    out_file = h5py.File(out_path, mode='w')

    datasets = list(in_files[0].keys())
    datasets_per_file = defaultdict(dict)
    for in_path, in_file in enumerate(in_files):
        in_file_sizes[in_path] = len(in_file[datasets[0]])
        for dataset in datasets:
            datasets_per_file[in_path][dataset] = in_file[dataset]

    for dataset in datasets:
        print("Dataset %s..." % dataset)
        current_infile = datasets_per_file[0][dataset]
        out_set = out_file.create_dataset(
            dataset,
            shape=(0,) + current_infile.shape[1:],
            maxshape=(None,) + current_infile.shape[1:],
            chunks=(BATCH_SIZE,) + current_infile.shape[1:],
            dtype=current_infile.dtype,
            compression='gzip',
            compression_opts=9,
            shuffle=True
        )

        for in_path in range(len(input_paths)):
            print("In-file %s..." % input_paths[in_path])
            current_infile = datasets_per_file[in_path][dataset]

            num_batches = int(np.ceil(len(current_infile) / BATCH_SIZE))
            for index_batch in tqdm.tqdm(range(num_batches)):
                start = index_batch * BATCH_SIZE
                end = min(len(current_infile), (index_batch + 1) * BATCH_SIZE)

                batch = current_infile[start:end]
                out_set.resize(out_set.shape[0] + len(batch), axis=0)
                out_set[-len(batch):] = batch

    out_file.close()
    return


if __name__ == "__main__":
    in_files = sys.argv[1:]
    main(in_files[:-1], in_files[-1])