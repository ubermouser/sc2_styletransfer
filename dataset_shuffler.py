from collections import defaultdict
import sys

import h5py
import numpy as np
import tqdm

from encoder.replay_encoder import BatchExporter, ENCODER_DTYPE


BATCH_SIZE = 128
def create_shuffle_index(in_file_sizes):
    dataset_size = sum(in_file_sizes)
    shuffle_index = np.zeros((dataset_size, 2), dtype=np.uint64)

    start = 0
    for in_file_idx, fs_size in enumerate(in_file_sizes):
        end = start+fs_size
        shuffle_index[start:end, 1] = in_file_idx
        shuffle_index[start:end, 0] = np.arange(end - start)
        start = end

    np.random.shuffle(shuffle_index)
    return shuffle_index


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

    shuffle_index = create_shuffle_index(in_file_sizes)
    for dataset in datasets:
        current_inset = datasets_per_file[0][dataset]
        out_file.create_dataset(
            dataset,
            shape=(len(shuffle_index),) + current_inset.shape[1:],
            dtype=current_inset.dtype,
            compression='gzip',
            compression_opts=9,
            shuffle=True)

    num_batches = int(np.ceil(len(shuffle_index) / BATCH_SIZE))
    for batch in tqdm.tqdm(range(num_batches)):
        start = batch * BATCH_SIZE
        end = min((batch + 1) * BATCH_SIZE, len(shuffle_index))
        shuffle_index_batch = shuffle_index[start:end]

        row_outs = defaultdict(list)
        for row_idx, dataset_idx in shuffle_index_batch:
            current_infile = datasets_per_file[dataset_idx]
            for dataset in datasets:
                row_outs[dataset].append(current_infile[dataset][row_idx])

        for dataset in datasets:
            out_file[dataset][start:end] = row_outs[dataset]

    return

if __name__ == "__main__":
    in_files = sys.argv[1:]
    main(in_files[:-1], in_files[-1])