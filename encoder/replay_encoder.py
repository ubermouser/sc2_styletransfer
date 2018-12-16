from collections import defaultdict
from functools import lru_cache
from glob import glob
import os

import six
import mpyq
import json
import zarr
import numcodecs
import numpy as np
import tqdm

from pysc2.lib import features
from pysc2.run_configs import lib as run_configs_lib

from s2clientprotocol import common_pb2 as sc_common
import sc2reader

numcodecs.blosc.set_nthreads(2)
CHUNK_SIZE = 2048
CHUNK_SHAPES = {'feature_minimap': (CHUNK_SIZE, 1, 64, 64), 'feature_screen': (CHUNK_SIZE, 1, 84, 84)}
DEFAULT_ENCODER = {
    'compressor': numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.blosc.BITSHUFFLE)
}
ENCODER_DTYPE = {
    'replay_path': str,
    'name': str,
    'race': 'S7'
}
ENCODERS = {
    'name': {'dtype': str},
    'replay_path': {'dtype': str},
    'race': {'dtype': 'S7'}
}

class NullBatchExporter(object):
    def __init__(self, *nargs, **kwargs):
        pass

    def export(self, **kwargs):
        pass

    def close(self):
        pass

    def flush(self):
        pass

class BatchExporter(object):
    def __init__(self, output_path, mode='w', batch_size=CHUNK_SIZE, dtypes={}):
        self._batch_size = batch_size
        self._lock = zarr.ProcessSynchronizer(os.path.join(os.path.dirname(output_path), "zarr.sync"))
        self._out = zarr.open(output_path, mode=mode, synchronizer=self._lock)
        self._cache = defaultdict(list)
        self._size = 0
        self._dtypes = dtypes

    def export(self, **kwargs):
        for dataset, output in kwargs.items():
            self._cache[dataset].append(output)
        self._size += 1
        if self._size < self._batch_size:
            return

        self.flush()

    def close(self):
        try:
            self.flush()
        finally:
            pass
            #self._out.close()

    def flush(self):
        for dataset, output in self._cache.items():
            assert len(output) == self._size
            if dataset not in self._dtypes:
                batch = np.asarray(output)
                dtype = batch.dtype
                shape = batch.shape
            else:
                dtype = self._dtypes[dataset]
                batch = np.array(output, dtype=dtype)
                shape = batch.shape

            if dataset not in self._out:
                args = dict(
                    name=dataset,
                    shape=(0,) + shape[1:],
                    chunks=CHUNK_SHAPES.get(dataset, (CHUNK_SIZE,) + shape[1:]),
                    dtype=dtype)
                args.update(ENCODERS.get(dataset, DEFAULT_ENCODER))
                out_set = self._out.create_dataset(**args)
            else:
                out_set = self._out[dataset]

            out_set.append(batch)
            out_set.shape
            output.clear()

        #self._out.flush()
        self._size = 0


def get_replay_version(replay_path):
    with open(replay_path, "rb") as f:
        replay_data = f.read()

    replay_io = six.BytesIO()
    replay_io.write(replay_data)
    replay_io.seek(0)
    archive = mpyq.MPQArchive(replay_io).extract()
    metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
    return run_configs_lib.Version(
        game_version=".".join(metadata["GameVersion"].split(".")[:-1]),
        build_version=int(metadata["BaseBuild"][4:]),
        data_version=metadata.get("DataVersion"),  # Only in replays version 4.1+.
        binary=None)


def replay_paths(replay_dir):
    """A generator yielding the full path to the replays under `replay_dir`."""
    for dirpath, dirnames, filenames in os.walk(replay_dir):
        for file in filenames:
            if os.path.splitext(file)[1].lower() == ".sc2replay":
                yield os.path.join(dirpath, file)


@lru_cache()
def player_names(replay_path):
    replay = sc2reader.load_replay(replay_path, load_level=2)
    return {id: p.name for id, p in replay.player.items()}


def encode(exporter, obs, player_id, replay_path, replay_info, step_index):
    info_base = replay_info.player_info[player_id - 1]
    player_info = replay_info.player_info[player_id - 1].player_info
    player_name = player_names(replay_path)[player_id]
    player_race = sc_common.Race.Name(player_info.race_actual)
    player_mmr = info_base.player_mmr

    exporter.export(
        feature_minimap=obs['feature_minimap'],
        feature_screen=obs['feature_screen'],
        score_cumulative=obs['score_cumulative'],
        player_stats=obs['player'],
        replay_path=replay_path,
        mmr=player_mmr,
        name=player_name,
        race=player_race,
        step=step_index
    )


class ReplayDumper(object):
    def __init__(self, exporter, step_mul):
        self._exporter = exporter
        self._step_mul = step_mul

    def process_replay(self, controller, **kwargs):
        f = features.features_from_game_info(game_info=controller.game_info())
        steps = 0
        with tqdm.tqdm() as pbar:
            while True:
                pbar.update()
                steps += 1

                o = controller.observe()
                if steps % self._step_mul == 0:
                    obs = f.transform_obs(o)
                    self._exporter.export(
                        feature_minimap=obs['feature_minimap'],
                        feature_screen=obs['feature_screen'],
                        score_cumulative=obs['score_cumulative'],
                        player=obs['player'],
                        #available_actions=obs['available_actions'],
                        **kwargs
                    )

                if o.player_result:  # end of game
                    break

                controller.step()
            self._exporter.flush()
        return