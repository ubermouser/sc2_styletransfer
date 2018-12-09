from collections import defaultdict
import sys

import h5py
import numpy as np
import tqdm

from encoder.replay_encoder import BatchExporter, ENCODER_DTYPE


BATCH_SIZE = 512
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
    out_file = BatchExporter(out_path, batch_size=BATCH_SIZE, dtypes=ENCODER_DTYPE)

    datasets = list(in_files[0].keys())
    datasets_per_file = defaultdict(dict)
    for in_path, in_file in enumerate(in_files):
        in_file_sizes[in_path] = len(in_file[datasets[0]])
        for dataset in datasets:
            datasets_per_file[in_path][dataset] = in_file[dataset]

    shuffle_index = create_shuffle_index(in_file_sizes)

    for row_idx, dataset_idx in tqdm.tqdm(shuffle_index):
        current_infile = datasets_per_file[dataset_idx]
        export = {dataset: current_infile[dataset][row_idx] for dataset in datasets}
        out_file.export(**export)

    out_file.close()
    return


if __name__ == "__main__":
    in_files = sys.argv[1:]
    main(in_files[:-1], in_files[-1])