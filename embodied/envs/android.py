import embodied
from pathlib import Path
from gym import spaces

# We'll assume the accessibility forwarder APK since it's the most flexible.
class Android(embodied.Env):
    def __init__(self,
                 task, # We'll case on this to find the textproto file (task_path).
                 size=(64, 64),
                 avd_name='my_avd',
                 android_avd_home='/Users/krishxmatta/.android/avd',
                 android_sdk_root='/Users/krishxmatta/Library/Android/sdk',
                 emulator_path='/Users/krishxmatta/Library/Android/sdk/emulator/emulator',
                 adb_path='/Users/krishxmatta/Library/Android/sdk/platform-tools/adb'):
        from android_env import loader
        from android_env.components import config_classes
        from android_env.wrappers.gym_wrapper import GymInterfaceWrapper
        from android_env.wrappers.discrete_action_wrapper import DiscreteActionWrapper
        from . import from_gym
        import pygame

        pygame.init() # TODO: This is a hack to prevent android environment from blocking during load, not sure why this works.

        task_path = None
        current_file_path = Path(__file__)
        if task == "clock_set_timer":
            task_path = current_file_path.parent / "android/accessibility_forwarder_clock_set_timer.textproto"

        config = config_classes.AndroidEnvConfig(
            task=config_classes.FilesystemTaskConfig(path=task_path),
            simulator=config_classes.EmulatorConfig(
                emulator_launcher=config_classes.EmulatorLauncherConfig(
                    emulator_path=emulator_path,
                    android_sdk_root=android_sdk_root,
                    android_avd_home=android_avd_home,
                    avd_name=avd_name,
                    run_headless=False,
                ),
                adb_controller=config_classes.AdbControllerConfig(
                    adb_path=adb_path
                ),
            ),
        )

        self._env = GymInterfaceWrapper(DiscreteActionWrapper(loader.load(config), action_grid=(10, 8)))

        print(spaces.Dict(self._env.observation_space))

        # Modify observation space so "pixels" is renamed to "image" per dreamer spec
        observation_space = dict(self._env.observation_space)
        observation_space["image"] = observation_space["pixels"]
        del observation_space["pixels"]

        # TODO: temporarily delete timedelta and orientation
        del observation_space["timedelta"]
        del observation_space["orientation"]

        self.observation_space = spaces.Dict(observation_space)

        # Modify action space so "action_id" is renamed to "action" per dreamer spec.
        action_space = dict(self._env.action_space)
        action_space["action"] = action_space["action_id"]
        del action_space["action_id"]

        self.action_space = spaces.Dict(action_space)

        self.wrappers = [from_gym.FromGym,
                         lambda e: embodied.core.wrappers.ResizeImage(e, size)]

    def reset(self):
        return self._env.reset()

    def step(self, action):
        action = dict(action)
        action["action_id"] = action["action"]
        del action["action"]

        obs, reward, done, info = self._env.step(action)

        del obs["timedelta"] # TODO: temporarily remove timedelta and orientation
        del observation_space["orientation"]

        obs = dict(obs)
        obs["image"] = obs["pixels"]
        del obs["pixels"]

        return obs, reward, done, info

    def render(self):
        return self._env.render()
