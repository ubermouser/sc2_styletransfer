from collections import deque
import gzip
import numpy as np

import keras as k

from pysc2.agents import base_agent
from pysc2.agents import random_agent
from pysc2.lib import actions
from pysc2.lib import features

from discriminator import preprocess_input, SAMPLING_RATE

PREDICTION_RATE = 51

class PredictorAgent(base_agent.BaseAgent):
    def __init__(self):
        super(PredictorAgent, self).__init__()

        self._minimap_buffer = deque(maxlen=SAMPLING_RATE)
        self._model = k.models.load_model("trained_model.keras")
        self._i_obs = 0

    def step(self, obs):
        self._minimap_buffer.append(np.asarray(obs.observation['feature_minimap']))
        self._i_obs += 1

        if self._i_obs % PREDICTION_RATE == 0 and len(self._minimap_buffer) == SAMPLING_RATE:
            input = np.asarray([self._minimap_buffer[-1], self._minimap_buffer[0]])
            input = input.reshape((1,) + input.shape)
            input = preprocess_input(input)
            prediction = self._model.predict(input)

            print("Agent Prediction: %s" % np.argmax(prediction))

        if obs.last():
            self._minimap_buffer.clear()
            self._i_obs = 0

        return super(PredictorAgent, self).step(obs)


class PredictorRandomAgent(random_agent.RandomAgent, PredictorAgent):
    pass