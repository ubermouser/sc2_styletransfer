from collections import defaultdict
from functools import lru_cache
from glob import glob
import os

import six
import mpyq
import json
import h5py
import numpy as np
import tqdm

from pysc2.lib import features
from pysc2.run_configs import lib as run_configs_lib

from s2clientprotocol import common_pb2 as sc_common
import sc2reader


class BatchExporter(object):
    def __init__(self, output_path, mode='w', batch_size=125, dtypes={}):
        self._batch_size = batch_size
        self._out = h5py.File(output_path, mode)
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
            self._out.close()

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
                out_set = self._out.create_dataset(
                    dataset,
                    shape=shape,
                    maxshape=(None,) + shape[1:],
                    chunks=(self._batch_size,) + shape[1:],
                    dtype=dtype,
                    compression='gzip',
                    compression_opts=9,
                    shuffle=True)
            else:
                out_set = self._out[dataset]
                out_set.resize(out_set.shape[0] + shape[0], axis=0)

            out_set[-shape[0]:] = batch
            output.clear()

        self._out.flush()
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


ENCODER_DTYPE = {
    'replay_path': h5py.special_dtype(vlen=str),
    'name': h5py.special_dtype(vlen=str),
    'race': 'S7'
}


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