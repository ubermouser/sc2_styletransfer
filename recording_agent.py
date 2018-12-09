import gzip
import numpy as np

from pysc2.agents import base_agent
from pysc2.agents import random_agent
from pysc2.lib import actions
from pysc2.lib import features


class RecordingAgent(base_agent.BaseAgent):
    def __init__(self):
        super(RecordingAgent, self).__init__()

        self._feature_minimap = []
        self._export_path = "."
        self._num_outputs = 0

    def step(self, obs):
        self._feature_minimap.append(np.asarray(obs.observation['feature_minimap']))

        if obs.last():
            data = np.asarray(self._feature_minimap)
            np.save(gzip.open("output_%d.npy.gz" % self._num_outputs, 'w'), data)
            self._feature_minimap = []
            self._num_outputs += 1

        return super(RecordingAgent, self).step(obs)


class RecordedRandomAgent(random_agent.RandomAgent, RecordingAgent):
    pass