import importlib
import os
import gym
import numpy as np

import nle

from nle import nethack

from rl_baseline.obs_wrappers import RenderCharImagesWithNumpyWrapper, MessageWrapper, BlstatsWrapper, ModifierWrapper
from sample_factory.envs.env_registry import global_env_registry
from utils.forked_pdb import ForkedPdb


def load_player_class(policy_name, player_class):
    try:
        # Dynamically construct the module name
        module_name = f"rl_baseline.{policy_name}.execution"
        # Import the module
        module = importlib.import_module(module_name)
        # Get the Player class from the module
        PlayerClass = getattr(module, player_class)
        return PlayerClass
    except ModuleNotFoundError:
        print(f"Module for {policy_name} not found.")
        return None
    except AttributeError:
        print(f"Player class not found in {policy_name}.code.")
        return None


class RootNLEWrapper(gym.Wrapper):
    """Some root-level additions to the NLE environment"""
    def __init__(self, env, num_skills=1, env_id=-1):
        super().__init__(env)
        
        bl_shape = 30 if 'Monk' in env.env.spec.name else 27
        bl_shape += num_skills
        task_space = {
            "tty_chars": gym.spaces.Box(0, 255, shape=(24, 80), dtype=np.uint8),
            "tty_colors": gym.spaces.Box(0, 31, shape=(24, 80), dtype=np.int8),
            "tty_cursor": gym.spaces.Box(0, 255, shape=(2,), dtype=np.uint8),
            "message": gym.spaces.Box(0, 255, shape=(256,), dtype=np.uint8),
            "glyphs": gym.spaces.Box(0, 5976, (21, 79), dtype=np.int16),
            "blstats": gym.spaces.Box(-2147483648, 2147483647, shape=(bl_shape,), dtype=np.int32),

            # インベントリ観測を追加
            "inv_strs": gym.spaces.Box(0, 255, shape=(55, 80), dtype=np.uint8),
            "inv_letters": gym.spaces.Box(0, 255, shape=(55,), dtype=np.uint8),
            "inv_oclasses": gym.spaces.Box(0, 255, shape=(55,), dtype=np.uint8),
            "inv_glyphs": gym.spaces.Box(0, 5976, shape=(55,), dtype=np.int16), 
        }

        self.env_id = np.array([env_id]).astype(np.int16)
        self.task_space = list(task_space.keys())

        self.observation_space = gym.spaces.Dict(task_space)

    def seed(self, *args):
        # Nethack does not allow seeding, so monkey-patch disable it here
        return

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = {k: obs[k] for k in obs if k in self.task_space}
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = {k: obs[k] for k in obs if k in self.task_space}
        return obs


def make_custom_env_func(full_env_name, cfg=None, env_config=None):
    root_env = cfg.root_env
    if env_config is not None and 'root_env' in env_config.keys():
        root_env = env_config['root_env']

    if env_config is not None and 'env_id' in env_config.keys():
        env_id = env_config['env_id']
    else:
        env_id = -1

    llm_reward = cfg.llm_reward if 'llm_reward' in cfg.keys() else 0.0
    penalty_mode = cfg.penalty_mode
    msg_max_keys = cfg.msg_max_keys

    base_actions = (
        [nethack.MiscAction.MORE] +
        list(nethack.CompassDirection) +
        list(nethack.CompassDirectionLonger) +
        list(nethack.MiscDirection) +
        [nethack.Command.KICK, nethack.Command.EAT, nethack.Command.SEARCH]
    )

    actions = base_actions + [
            nethack.Command.CAST,
            nethack.Command.ENHANCE,
            nethack.Command.PRAY,
            nethack.Command.QUAFF,
            nethack.Command.DROP,
            nethack.Command.READ,
            nethack.Command.ZAP,
        ]

    ttyrec = cfg.ttyrec if 'ttyrec' in cfg.keys() else cfg.save_ttyrec_every
    cfg.save_dir = f'{cfg.train_dir}/{cfg.experiment}/ttyrecs/A'

    env = RootNLEWrapper(gym.make(
            root_env,
            save_ttyrec_every=ttyrec,
            savedir=cfg.save_dir,
            observation_keys=[
                "tty_chars",
                "tty_colors", 
                "tty_cursor", 
                "blstats", 
                "message", 
                "glyphs", 
                "inv_strs", 
                "inv_letters", 
                "inv_oclasses",
                "inv_glyphs"
                ],
            actions=tuple(actions),
            max_episode_steps=cfg.max_episode_steps,
            penalty_mode=penalty_mode,
            evaluation='none' not in cfg.eval_target,
            eval_target=cfg.eval_target,
            ),
            num_skills=cfg.num_skills,
            env_id=env_id,
        )

    env = RenderCharImagesWithNumpyWrapper(env, font_size=9, crop_size=cfg.crop_size, rescale_font_size=(6, 6))

    env = MessageWrapper(env, llm_reward=llm_reward, msg_max_keys=msg_max_keys)

    exploration_str = 'exploration'

    if cfg.eval_target != 'none':
        exp_folder = cfg.eval_target
    else:
        exp_folder = exploration_str
        if cfg.meta_policy_name:
            exp_folder = cfg.meta_policy_name

    seed = cfg.code_seed if cfg.code_seed != -1 else cfg.seed

    # If pure RL mode, do not load meta policy class unless explicitly forced
    if llm_reward > 0.0 and (cfg.eval_target != 'none' or cfg.meta_policy):
        policy_name = f"meta_policies.{exp_folder}.seed{seed}"
        player_class = "NetHackPlayer"
        MetaPolicyClass = load_player_class(policy_name, player_class)
    else:
        MetaPolicyClass = None

    env = ModifierWrapper(env, llm_reward=llm_reward, experiment=cfg.experiment, 
                          num_skills=cfg.num_skills, meta_policy_class=MetaPolicyClass)

    env = BlstatsWrapper(env, experiment=cfg.experiment, diff_h=cfg.diff_h, diffstats_size=cfg.diffstats_size)

    return env


global_env_registry().register_env(
    env_name_prefix='nle_',
    make_env_func=make_custom_env_func,
)
