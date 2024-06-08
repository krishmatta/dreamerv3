import embodied
import numpy as np
from pathlib import Path
from gym import spaces

def get_dummy_spaces(action_size=(23, 11), img_size=(64, 64)):
    obs = {
        "image": embodied.Space(dtype=np.uint8, shape=(img_size[0], img_size[1], 3), low=0, high=255),
        "reward": embodied.Space(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
        "is_first": embodied.Space(dtype=bool, shape=(), low=False, high=True),
        "is_last": embodied.Space(dtype=bool, shape=(), low=False, high=True),
        "is_terminal": embodied.Space(dtype=bool, shape=(), low=False, high=True),
    }

    act = {
        "action": embodied.Space(dtype=np.float32, shape=(action_size[0] * action_size[1] - 1,), low=0, high=1),
        "reset": embodied.Space(dtype=bool, shape=(), low=False, high=True),
    }

    return obs, act

class AndroidSimple(embodied.Env):
    def __init__(self,
                 task, # We'll case on this for setup
                 action_size=(23, 11),
                 img_size=(64, 64)):
        from android_env.simple.gym_environment import AndroidGymEnvironment
        from android_env.simple.gym_wrappers import DiscreteWrapper
        from . import from_gym

        self.action_size = action_size

        reward_terminate = None
        reset = []
        app = None
        if task == "clock_reset_timer":
            def reward_terminate(logs):
                ret = 0
                for x in logs:
                    if "PAUSED to RESET" in x:
                        ret += 1
                return ret, bool(ret)
            reset = ["shell pm clear com.google.android.deskclock"]
            app = "com.google.android.deskclock"

        self._env = DiscreteWrapper(AndroidGymEnvironment("emulator-5554", reward_terminate, reset, app=app), action_size[0], action_size[1])

        self.observation_space = self._env.observation_space
        self.action_space = spaces.Dict({
            "action": spaces.Box(low=0, high=(action_size[0] * action_size[1]) - 1, shape=(), dtype=np.int32)
        })

        self.wrappers = [from_gym.FromGym,
                         lambda e: embodied.core.wrappers.ResizeImage(e, img_size)]

    def reset(self):
        return self._env.reset()

    def step(self, action):
        action = dict(action)
        action["pos"] = (action["action"] // self.action_size[1], action["action"] % self.action_size[1])
        del action["action"]

        obs, reward, done, info = self._env.step(action)

        return obs, reward, done, info

    def render(self):
        return self._env.render()
