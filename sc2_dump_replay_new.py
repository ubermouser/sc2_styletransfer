#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import multiprocessing
import os
import signal
import sys
import threading
import time

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin
import queue
import six

from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import point_flag
from pysc2.lib import protocol
from pysc2.lib import remote_controller

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb

from encoder.replay_encoder import get_replay_version
from encoder.replay_encoder import BatchExporter, encode, ENCODER_DTYPE, replay_paths


if os.name == 'nt':
    REPLAY_PATH = os.path.join("B:", "downloads", "SC2.4.1.2.60604_2018_05_16", "StarCraftII", "Replays")
else:
    REPLAY_PATH = os.path.join("/media", "sf_B_DRIVE", "downloads", "SC2.4.1.2.60604_2018_05_16", "StarCraftII", "Replays", "ZergAgent")

FLAGS = flags.FLAGS
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", "256,192",
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", "128",
                        "Resolution for rendered minimap.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 8, "How many game steps per observation.")
flags.DEFINE_string("replays", REPLAY_PATH, "Path to a directory of replays.")
flags.DEFINE_string("out_path", "/tmp/output_%d.hdf5", "Path to an output hdf5.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,
                  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
#flags.mark_flag_as_required("replays")
FLAGS(sys.argv)

def sorted_dict_str(d):
    return "{%s}" % ", ".join("%s: %s" % (k, d[k])
                              for k in sorted(d, key=d.get, reverse=True))


class ReplayStats(object):
    """Summary stats of the replays seen so far."""

    def __init__(self):
        self.replays = 0
        self.steps = 0
        self.invalid_states = 0
        self.maps = collections.defaultdict(int)
        self.races = collections.defaultdict(int)
        self.crashing_replays = set()
        self.invalid_replays = set()

    def merge(self, other):
        """Merge another ReplayStats into this one."""

        def merge_dict(a, b):
            for k, v in six.iteritems(b):
                a[k] += v

        self.invalid_states += other.invalid_states
        self.replays += other.replays
        self.steps += other.steps
        merge_dict(self.maps, other.maps)
        merge_dict(self.races, other.races)
        self.crashing_replays |= other.crashing_replays
        self.invalid_replays |= other.invalid_replays

    def __str__(self):
        len_sorted_dict = lambda s: (len(s), sorted_dict_str(s))
        len_sorted_list = lambda s: (len(s), sorted(s))
        return "\n\n".join((
            "Replays: %s, Steps total: %s, Invalid: %s" % (self.replays, self.steps, self.invalid_states),
            "Maps: %s\n%s" % len_sorted_dict(self.maps),
            "Races: %s\n%s" % len_sorted_dict(self.races),
            "Crashing replays: %s\n%s" % len_sorted_list(self.crashing_replays),
            "Invalid replays: %s\n%s" % len_sorted_list(self.invalid_replays),
        ))


class ProcessStats(object):
    """Stats for a worker process."""

    def __init__(self, proc_id):
        self.proc_id = proc_id
        self.time = time.time()
        self.stage = ""
        self.replay = ""
        self.replay_stats = ReplayStats()

    def update(self, stage):
        self.time = time.time()
        self.stage = stage

    def __str__(self):
        return ("[%2d] replay: %10s, replays: %5d, steps: %7d, game loops: %7s, "
                "last: %12s, %3d s ago" % (
                    self.proc_id, self.replay, self.replay_stats.replays,
                    self.replay_stats.steps,
                    self.replay_stats.steps * FLAGS.step_mul, self.stage,
                    time.time() - self.time))


def valid_replay(info, ping):
    """Make sure the replay isn't corrupt, and is worth looking at."""
    if info.HasField("error"):
        # Probably corrupt
        return "Replay info parse failure"
    elif info.base_build != ping.base_build:
        # different game version
        return "Game Version Mismatch"
    elif info.game_duration_loops < 1000:
        return "Duration too short"
    elif len(info.player_info) != 2:
        return "Incorrect number of players"
    for p in info.player_info:
        if p.player_apm < 10:
            # Low APM = player just standing around.
            # Low MMR = corrupt replay or player who is weak.
            return "low APM"
        #elif p.player_mmr < 1000:
        #    return "low MMR"
    return None


class ReplayProcessor(multiprocessing.Process):
    """A Process that pulls replays and processes them."""

    def __init__(
            self,
            proc_id,
            run_config,
            replay_queue,
            stats_queue,
            game_version):
        super(ReplayProcessor, self).__init__()
        self.stats = ProcessStats(proc_id)
        self.run_config = run_config
        self.replay_queue = replay_queue
        self.stats_queue = stats_queue
        self.step_multplier = FLAGS.step_mul
        self.out_path = FLAGS.out_path % proc_id
        self.sc2_version = game_version

    def interface(self):
        agent_interface_format = features.parse_agent_interface_format(
            feature_screen=FLAGS.feature_screen_size,
            feature_minimap=FLAGS.feature_minimap_size,
            rgb_screen=FLAGS.rgb_screen_size,
            rgb_minimap=FLAGS.rgb_minimap_size,
            action_space=FLAGS.action_space,
            use_feature_units=FLAGS.use_feature_units
        )
        return sc2_env.SC2Env._get_interface(agent_interface_format, require_raw=False)

    def run(self):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
        self._update_stage("spawn")
        replay_name = "none"
        while True:
            self._print("Starting up a new SC2 instance.")
            self._update_stage("launch")
            exporter = BatchExporter(self.out_path, mode='a', dtypes=ENCODER_DTYPE)
            try:
                with self.run_config.start(version=self.sc2_version) as controller:
                    self._print("SC2 Started successfully.")
                    ping = controller.ping()
                    for _ in range(300):
                        try:
                            replay_path = self.replay_queue.get()
                        except queue.Empty:
                            self._update_stage("done")
                            self._print("Empty queue, returning")
                            return
                        try:
                            replay_name = os.path.basename(replay_path)[:30]
                            self.stats.replay = replay_name
                            self._print("Got replay: %s" % replay_path)
                            self._update_stage("open replay file")
                            replay_data = self.run_config.replay_data(replay_path)
                            self._update_stage("replay_info")
                            info = controller.replay_info(replay_data)
                            self._print((" Replay Info %s " % replay_name).center(80, "-"))
                            self._print(info)
                            self._print("-" * 80)
                            replay_validity = valid_replay(info, ping)
                            if replay_validity is None:
                                self.stats.replay_stats.maps[info.map_name] += 1
                                for player_info in info.player_info:
                                    race_name = sc_common.Race.Name(
                                        player_info.player_info.race_actual)
                                    self.stats.replay_stats.races[race_name] += 1
                                map_data = None
                                if info.local_map_path:
                                    self._update_stage("open map file")
                                    map_data = self.run_config.map_data(info.local_map_path)
                                for player_id in [1, 2]:
                                    self._print("Starting %s from player %s's perspective" % (
                                        replay_name, player_id))
                                    self.process_replay(exporter, controller, replay_data, map_data,
                                                        replay_path, info, player_id)
                            else:
                                self._print("Replay is invalid: %s" % replay_validity)
                                self.stats.replay_stats.invalid_replays.add(replay_name)
                        finally:
                            self.replay_queue.task_done()
                    self._update_stage("shutdown")
            except (protocol.ConnectionError, protocol.ProtocolError,
                    remote_controller.RequestError):
                self.stats.replay_stats.crashing_replays.add(replay_name)
            except KeyboardInterrupt:
                return
            finally:
                exporter.close()

    def join(self, timeout=None):
        #self.exporter.close()
        super(ReplayProcessor, self).join(timeout)

    def _print(self, s):
        for line in str(s).strip().splitlines():
            print("[%s] %s" % (self.stats.proc_id, line))

    def _update_stage(self, stage):
        self.stats.update(stage)
        self.stats_queue.put(self.stats)

    def process_replay(self, exporter, controller, replay_data, map_data, replay_path, replay_info, player_id):
        """Process a single replay, updating the stats."""
        self._update_stage("start_replay")
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=self.interface(),
            observed_player_id=player_id))

        feat = features.features_from_game_info(
            controller.game_info(), use_feature_units=False,
            action_space=actions.ActionSpace[FLAGS.action_space.upper()])

        steps = 0
        self.stats.replay_stats.replays += 1
        self._update_stage("step")
        controller.step()
        while True:
            self.stats.replay_stats.steps += 1
            steps += 1

            self._update_stage("observe")
            o = controller.observe()
            try:
                obs = feat.transform_obs(o)
                encode(exporter, obs, player_id, replay_path, replay_info, steps)
            except ValueError:
                self.stats.replay_stats.invalid_states += 1

            if o.player_result:
                break

            self._update_stage("step")
            controller.step(self.step_multplier)


def stats_printer(stats_queue):
    """A thread that consumes stats_queue and prints them every 10 seconds."""
    proc_stats = [ProcessStats(i) for i in range(max(FLAGS.parallel, 1))]
    print_time = start_time = time.time()
    width = 107

    running = True
    while running:
        print_time += 10

        while time.time() < print_time:
            try:
                s = stats_queue.get(True, print_time - time.time())
                if s is None:  # Signal to print and exit NOW!
                    running = False
                    break
                proc_stats[s.proc_id] = s
            except queue.Empty:
                pass

        replay_stats = ReplayStats()
        for s in proc_stats:
            replay_stats.merge(s.replay_stats)

        print((" Summary %0d secs " % (print_time - start_time)).center(width, "="))
        print(replay_stats)
        print(" Process stats ".center(width, "-"))
        print("\n".join(str(s) for s in proc_stats))
        print("=" * width)


def replay_queue_filler(replay_queue, replay_list):
    """A thread that fills the replay_queue with replay filenames."""
    for replay_path in replay_list:
        replay_queue.put(replay_path)


def main(unused_argv):
    """Dump stats about all the actions that are in use in a set of replays."""
    run_config = run_configs.get()

    stats_queue = multiprocessing.Queue()
    stats_thread = threading.Thread(target=stats_printer, args=(stats_queue,))
    stats_thread.start()
    try:
        # For some reason buffering everything into a JoinableQueue makes the
        # program not exit, so save it into a list then slowly fill it into the
        # queue in a separate thread. Grab the list synchronously so we know there
        # is work in the queue before the SC2 processes actually run, otherwise
        # The replay_queue.join below succeeds without doing any work, and exits.
        print("Getting replay list:", FLAGS.replays)
        replay_list = sorted(replay_paths(FLAGS.replays))
        print(len(replay_list), "replays found.\n")

        if len(replay_list) > 0:
            game_version = get_replay_version(replay_list[0])
        else:
            game_version = None

        replay_queue = multiprocessing.JoinableQueue(FLAGS.parallel * 10)
        replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                               args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        if FLAGS.parallel > 0:
            for i in range(FLAGS.parallel):
                p = ReplayProcessor(i, run_config, replay_queue, stats_queue, game_version)
                p.daemon = True
                p.start()
                time.sleep(1)  # Stagger startups, otherwise they seem to conflict somehow
        else:
            ReplayProcessor(0, run_config, replay_queue, stats_queue, game_version).run()

        replay_queue.join()  # Wait for the queue to empty.
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")
    finally:
        stats_queue.put(None)  # Tell the stats_thread to print and exit.
        stats_thread.join()


if __name__ == "__main__":
    FLAGS(sys.argv)
    app.run(main)
