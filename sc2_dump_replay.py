#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tqdm

from absl import app
from absl import flags

from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import point_flag

from s2clientprotocol import sc2api_pb2 as sc_pb

from encoder.replay_encoder import BatchExporter, encode, ENCODER_DTYPE


FLAGS = flags.FLAGS
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,
                  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_integer("step_mul", 8, "Game steps per observation read.")


#REPLAY_PATH = "C:\Program Files (x86)\StarCraft II\Replays"
REPLAY_PATH = "/media/sf_B_DRIVE/downloads/SC2.4.1.2.60604_2018_05_16/StarCraftII/Replays/"
#REPLAY_PATH = "B:\downloads\SC2.4.1.2.60604_2018_05_16\StarCraftII\Replays"
OUT_PATH = "/media/sf_B_DRIVE/downloads"


class GameController(object):
    """Wrapper class for interacting with the game in play/replay mode."""

    def __init__(self):
        """Constructs the game controller object.

        Args:
          config: Interface configuration options.
        """
        agent_interface_format = features.parse_agent_interface_format(
            feature_screen=FLAGS.feature_screen_size,
            feature_minimap=FLAGS.feature_minimap_size,
            rgb_screen=FLAGS.rgb_screen_size,
            rgb_minimap=FLAGS.rgb_minimap_size,
            action_space=FLAGS.action_space,
            use_feature_units=FLAGS.use_feature_units
        )
        self._interface = sc2_env.SC2Env._get_interface(
            agent_interface_format, require_raw=False)
        self._player_id = 1
        self._sc2_proc = None
        self._controller = None
        self._replay_data = None

        self._initialize()

    def _initialize(self):
        """Initialize play/replay connection."""
        self._run_config = run_configs.get()
        self._sc2_proc = self._run_config.start()
        self._controller = self._sc2_proc.controller

    def start_replay(self, replay_path):
        replay_data = self._run_config.replay_data(replay_path)
        replay_info = self._controller.replay_info(replay_data)
        map_data = self._run_config.map_data(replay_info.local_map_path)

        start_replay = sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=self._interface,
            disable_fog=False,
            observed_player_id=self._player_id)
        self._controller.start_replay(start_replay)

        return replay_info

    @property
    def controller(self):
        return self._controller

    def close(self):
        """Close the controller connection."""
        if self._controller:
            self._controller.quit()
            self._controller = None
        if self._sc2_proc:
            self._sc2_proc.close()
            self._sc2_proc = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class ReplayDumper(object):
    def __init__(self, exporter):
        self._exporter = exporter
        self._step_mul = FLAGS.step_mul

    def process_replay(self, controller, replay_info, **kwargs):
        f = features.features_from_game_info(game_info=controller.game_info())
        steps = 0
        with tqdm.tqdm() as pbar:
            while True:
                pbar.update()
                steps += 1

                o = controller.observe()
                if steps % self._step_mul == 0:
                    obs = f.transform_obs(o)

                    encode(self._exporter, obs, 1, replay_info, steps)

                if o.player_result:  # end of game
                    break

                controller.step()
            self._exporter.flush()
        return


def replay_observations(replay_path, output_path):
    exporter = BatchExporter(output_path, dtypes=ENCODER_DTYPE)
    dumper = ReplayDumper(exporter)
    with GameController() as game_controller:
        replay_info = game_controller.start_replay(replay_path)
        dumper.process_replay(game_controller.controller, replay_info, class_label=0)


def main(argv):
    replay_observations(
        os.path.join(REPLAY_PATH, "ZergAgent/Simple64_2018-12-08-21-43-41.SC2Replay"),
        os.path.join(OUT_PATH, "/tmp/output.hdf5"))


if __name__ == '__main__':
    app.run(main)