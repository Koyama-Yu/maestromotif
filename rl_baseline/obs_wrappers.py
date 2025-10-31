import copy
import os
import re
import sys
from collections import defaultdict, deque, OrderedDict

import cv2
import gym
import nle
import numpy as np
import pickle
from numba import njit
from PIL import Image, ImageDraw, ImageFont
from nle.nethack.actions import Command, MiscDirection, WizardCommand

from rl_baseline.price_id import is_item_identified
from rl_baseline.item_tracker import ItemTracker
from sample_factory.utils.utils import log
from utils.forked_pdb import ForkedPdb

# Mapping of 0-15 colors used.
# Taken from bottom image here. It seems about right https://i.stack.imgur.com/UQVe5.png
COLORS = [
    "#000000",
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",  # - flipped these ones around
    "#C0C0C0",  # | the gray-out dull stuff
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#0000FF",
    "#FF00FF",
    "#00FFFF",
    "#FFFFFF",
]


class BlstatsWrapper(gym.Wrapper):
    """Create normalized version of the baseline stats from NetHack"""

    # Hand-chosen scaling values for each blstat entry. Aims to limit them in [0, 1] range.
    BLSTAT_NORMALIZATION_STATS = np.array(
        [
            1.0 / 79,  # hero col 0
            1.0 / 21,  # hero row 1

            # Probably useless
            0.0,  # strength pct 2
            0.0 / 10,  # strength 3
            0.0 / 10,  # dexterity 4
            0.0 / 10,  # constitution 5
            0.0 / 10,  # intelligence 6
            0.0 / 10,  # wisdom 7
            0.0 / 10,  # charisma 8
            0.0,  # score 9

            # Super useful
            1.0 / 10,  # hitpoints 10
            1.0 / 10,  # max hitpoints 11
            0.0 / 10,  # depth 12
            0.0,  # gold 13
            1.0 / 10,  # energy 14
            1.0 / 10,  # max energy 15
            1.0 / 10,  # armor class 16

            # Probably useless
            0.0,  # monster level 17

            # Useful
            1.0 / 10,  # experience level 18
            0.0,  # experience points 19
            1.0 / 1000,  # time 20
            1.0,  # hunger_state 21

            # Probably useless
            0.0,  # carrying capacity 22 
            1.0,  # dungeon number 23
            0.0,  # level number 24
            1.0,  # condition bits 25
            0.0,  # alignment 26

            # Super useful
            1.0, # spell category 27
            1.0, # potions or no potions 28
            1.0, # comestibles or no comestibles 29
        ]
    )

    # Make sure we do not spook the network
    BLSTAT_CLIP_RANGE = (-5, 5)

    def __init__(self, env, experiment, diff_h=50, diffstats_size=1):
        super().__init__(env)
        self.experiment = experiment

        self.skill_multiplier = [1.0] * self.env.num_skills
        self.bl_norm = np.concatenate((BlstatsWrapper.BLSTAT_NORMALIZATION_STATS, self.skill_multiplier))

        ## 追記 ##

        # monk_tasks_nle.pyでの追加要素を考慮して動的に長さを決定
        # 基本blstats(24) + skill_feature(1) + potions(1) + comestibles(1) + skill_vector(num_skills)
        base_blstats_size = 24  # NetHackの基本blstats
        additional_monk_features = 3  # skill_feature + potions + comestibles
        expected_blstats_size = base_blstats_size + additional_monk_features + self.env.num_skills
        
        # 正規化配列も動的に構築
        monk_features_norm = [1.0, 1.0, 1.0]  # skill_feature, potions, comestibles用
        self.bl_norm = np.concatenate((
            BlstatsWrapper.BLSTAT_NORMALIZATION_STATS,  # 元の24要素
            monk_features_norm,  # monk追加の3要素
            self.skill_multiplier  # skill vector
        ))

        # 実際の環境から正確なサイズを取得
        dummy_obs = env.reset()
        actual_blstats_size = dummy_obs['blstats'].shape[0] if 'blstats' in dummy_obs else expected_blstats_size

        ## 追記 ##

        self.num_items = min(
                self.bl_norm.shape[0],
                env.observation_space['blstats'].shape[0]
            )
        
        ## 追記 ##
        # 正規化配列のサイズを実際のblstatsに合わせる
        if self.bl_norm.shape[0] != actual_blstats_size:
            # 足りない場合は1.0で埋める、多い場合は切り詰める
            if self.bl_norm.shape[0] < actual_blstats_size:
                padding = np.ones(actual_blstats_size - self.bl_norm.shape[0])
                self.bl_norm = np.concatenate([self.bl_norm, padding])
            else:
                self.bl_norm = self.bl_norm[:actual_blstats_size]
        
        #log.info(f"BlstatsWrapper: actual_blstats_size={actual_blstats_size}, bl_norm_size={self.bl_norm.shape[0]}")
        ## 追記 ##

        self.diff_h = diff_h
        self.diffstats_dict = {'dlvl': 12, 'gold': 13, 'hp': 10, 'xp': 18, 'hunger': 21}
        self.prev_stats = {}

        obs_spaces = {
            "norm_blstats": gym.spaces.Box(
                low=BlstatsWrapper.BLSTAT_CLIP_RANGE[0],
                high=BlstatsWrapper.BLSTAT_CLIP_RANGE[1],
                shape=(self.num_items,),
                dtype=np.float32,
            ),
            "diffstats": gym.spaces.Box(
                low=BlstatsWrapper.BLSTAT_CLIP_RANGE[0],
                high=BlstatsWrapper.BLSTAT_CLIP_RANGE[1],
                shape=(diffstats_size,),
                dtype=np.float32,
            )
        }
        # Update observation_space to restrict to only the used spaces by SF
        obs_spaces.update(
            [
                (k, self.env.observation_space[k])
                for k in self.env.observation_space
                if k != "blstats"
            ]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _adjust_blstats(self, obs):
        obs['diffstats'] = []
        for i, (key, ind) in enumerate(self.diffstats_dict.items()):
            stat = obs['blstats'][ind]
            prev_stat = self.prev_stats[key][0]
            diff = np.clip(stat - prev_stat, -1, 1)
            obs['diffstats'].append(diff)
        obs['diffstats'] = np.array(obs['diffstats'], dtype=np.float32)
        obs['diffstats'][1:] *= 0 # Only keep some stats for BT model

        # 動的サイズ対応
        blstats_size = min(obs["blstats"].shape[0], self.bl_norm.shape[0])
        #norm_blstats = (obs["blstats"] * self.bl_norm[:self.num_items])
        norm_blstats = (obs["blstats"][:blstats_size] * self.bl_norm[:blstats_size])
        norm_blstats = norm_blstats.astype(np.float32)
        obs["norm_blstats"] = norm_blstats
        return obs

    def reset_prev_stats(self, obs):
        for key, ind in self.diffstats_dict.items():
            self.prev_stats[key] = deque(maxlen=self.diff_h)
            self.prev_stats[key].append(obs['blstats'][ind])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.prev_stats['dlvl'].append(obs['blstats'][12])
        # record stats here to align with option selection

        obs = self._adjust_blstats(obs)
        if self.env.skill_end:
            self.prev_stats = {'dlvl': deque(maxlen=self.diff_h)}
            self.prev_stats['dlvl'].append(obs['blstats'][12])
            self.reset_prev_stats(obs)
        _ = obs.pop("blstats")
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()

        self.prev_stats = {'dlvl': deque(maxlen=self.diff_h)}
        self.prev_stats['dlvl'].append(obs['blstats'][12])
        self.reset_prev_stats(obs)

        obs = self._adjust_blstats(obs)
        _ = obs.pop("blstats")
        return obs

@njit
def tile_characters_to_image(
    out_image,
    chars,
    colors,
    output_height_chars,
    output_width_chars,
    char_array,
    offset_h,
    offset_w,
):
    """
    Build an image using cached images of characters in char_array to out_image
    """
    char_height = char_array.shape[3]
    char_width = char_array.shape[4]
    for h in range(output_height_chars):
        h_char = h + offset_h
        # Stuff outside boundaries is not visible, so
        # just leave it black
        if h_char < 0 or h_char >= chars.shape[0]:
            continue
        for w in range(output_width_chars):
            w_char = w + offset_w
            if w_char < 0 or w_char >= chars.shape[1]:
                continue
            char = chars[h_char, w_char]
            color = colors[h_char, w_char]
            h_pixel = h * char_height
            w_pixel = w * char_width
            out_image[
                :, h_pixel : h_pixel + char_height, w_pixel : w_pixel + char_width
            ] = char_array[char, color]


def initialize_char_array(
    font_size, rescale_font_size, font_path="rl_baseline/Hack-Regular.ttf"
):
    """Draw all characters in PIL and cache them in numpy arrays

    if rescale_font_size is given, assume it is (width, height)

    Returns a np array of (num_chars, num_colors, char_height, char_width, 3)
    """
    font = ImageFont.truetype(os.path.abspath(font_path), font_size)
    dummy_text = "".join(
        [(chr(i) if chr(i).isprintable() else " ") for i in range(256)]
    )
    _, _, image_width, image_height = font.getbbox(dummy_text)
    # Above can not be trusted (or its siblings)....
    image_width = int(np.ceil(image_width / 256) * 256)

    if rescale_font_size:
        char_width = rescale_font_size[0]
        char_height = rescale_font_size[1]
    else:
        char_width = image_width // 256
        char_height = image_height

    char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)
    image = Image.new("RGB", (image_width, image_height))
    image_draw = ImageDraw.Draw(image)
    for color_index in range(16):
        image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
        image_draw.text((0, 0), dummy_text, fill=COLORS[color_index], spacing=0)

        arr = np.array(image).copy()
        arrs = np.array_split(arr, 256, axis=1)
        for char_index in range(256):
            char = arrs[char_index]
            if rescale_font_size:
                char = cv2.resize(char, rescale_font_size, interpolation=cv2.INTER_AREA)
            char_array[char_index, color_index] = char
    return char_array


class RenderCharImagesWithNumpyWrapper(gym.Wrapper):
    """
    Render characters as images, using PIL to render characters like we humans see on screen
    but then some caching and numpy stuff to speed up things.

    To speed things up, crop image around the player.
    """

    def __init__(self, env, font_size=9, crop_size=None, rescale_font_size=None):
        super().__init__(env)
        self.char_array = initialize_char_array(font_size, rescale_font_size)
        self.char_height = self.char_array.shape[2]
        self.char_width = self.char_array.shape[3]
        # Transpose for CHW
        self.char_array = self.char_array.transpose(0, 1, 4, 2, 3)

        self.crop_size = crop_size

        if crop_size is None:
            # Render full "obs"
            old_obs_space = self.env.observation_space["obs"]
            self.output_height_chars = old_obs_space.shape[0]
            self.output_width_chars = old_obs_space.shape[1]
        else:
            # Render only crop region
            self.half_crop_size = crop_size // 2
            self.output_height_chars = crop_size
            self.output_width_chars = crop_size
        self.chw_image_shape = (
            3,
            self.output_height_chars * self.char_height,
            self.output_width_chars * self.char_width,
        )

        # sample-factory expects at least one observation named "obs"
        obs_spaces = {
            "obs": gym.spaces.Box(
                low=0, high=255, shape=self.chw_image_shape, dtype=np.uint8
            )
        }

        obs_spaces.update(
            [(k, self.env.observation_space[k]) for k in self.env.observation_space]
        )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _render_text_to_image(self, obs):
        chars = obs["tty_chars"]
        colors = obs["tty_colors"]
        offset_w = 0
        offset_h = 0
        if self.crop_size:
            # Center around player
            center_x, center_y = obs["blstats"][:2]
            offset_h = center_y - self.half_crop_size
            offset_w = center_x - self.half_crop_size

        out_image = np.zeros(self.chw_image_shape, dtype=np.uint8)

        tile_characters_to_image(
            out_image=out_image,
            chars=chars,
            colors=colors,
            output_height_chars=self.output_height_chars,
            output_width_chars=self.output_width_chars,
            char_array=self.char_array,
            offset_h=offset_h,
            offset_w=offset_w,
        )

        obs["obs"] = out_image
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._render_text_to_image(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._render_text_to_image(obs)
        return obs


class LimitedDefaultDict:
    def __init__(self, default_factory, max_keys):
        self.store = defaultdict(default_factory)
        self.order = OrderedDict()
        self.max_keys = max_keys

    def __setitem__(self, key, value):
        self.store[key] = value
        self.order[key] = None
        if key in self.order:
            self.order.move_to_end(key)
        self._maintain_limit()

    def __getitem__(self, key):
        value = self.store[key]
        self.order[key] = None
        self.order.move_to_end(key)
        return value

    def _maintain_limit(self):
        if self.max_keys > 0 and len(self.order) > self.max_keys:
            oldest = next(iter(self.order))
            del self.order[oldest]
            del self.store[oldest]

    def __delitem__(self, key):
        del self.store[key]
        del self.order[key]

    def __repr__(self):
        return repr(self.store)


class MessageWrapper(gym.Wrapper):
    """Keep some statistic about the messages."""

    def __init__(self, env, llm_reward=0.0, msg_max_keys=-1):
        super().__init__(env)
        self.msg_max_keys = msg_max_keys
        self.llm_reward = llm_reward

        self.messages_dict = LimitedDefaultDict(int, max_keys=self.msg_max_keys)
        self.metric_counts = defaultdict(int)

        obs_spaces = {
            "msg_count": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        }
        obs_spaces.update(
            [(k, self.env.observation_space[k]) for k in self.env.observation_space]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # keeping track of messages counts
        obs["msg_count"] = np.array([1.0]).astype(np.float32)
        msg_str = self.env.message[1]

        if self.llm_reward > 0.:
            self.messages_dict[msg_str] += 1
            obs["msg_count"] = np.array([self.messages_dict[msg_str]]).astype(np.float32)

        if done:
            info.update(self.metric_counts)

        return obs, reward, done, info

    def reset(self):
        self.messages_dict = LimitedDefaultDict(int, max_keys=self.msg_max_keys)
        self.metric_counts = defaultdict(int)

        obs = self.env.reset()

        # keeping track of messages counts
        obs["msg_count"] = np.array([1.0]).astype(np.float32)
        msg_str = self.env.message[1]

        if self.llm_reward > 0.:
            self.messages_dict[msg_str] += 1
            obs["msg_count"] = np.array([self.messages_dict[msg_str]]).astype(np.float32)

        return obs


class ModifierWrapper(gym.Wrapper):

    def __init__(self, env, llm_reward, experiment, num_skills, meta_policy_class):
        super().__init__(env)
        self.llm_reward = llm_reward
        self.experiment = experiment

        self.prev_msg = b''
        self.num_skills = num_skills
        self.meta_policy_class = meta_policy_class

        self.skill_vector = np.eye(self.num_skills)
        self.max_level_reached = 1
        self.skill_end = False

        # 統計情報の初期化
        self.num_buc = 0
        self.num_sold = 0
        self.num_sell = 0
        self.num_price_id = 0
        self.altar_seen = False
        self.shop_seen = False
        self.price_id = False

        # アイテム追跡機能の初期化
        #mode = 'eval' if not getattr(env, 'evaluation', True) else 'train'
        mode = 'eval' if getattr(env, 'evaluation', True) else 'train'
        # ひとまず, enable_detailed_loggingはexport名に基づいて決定
        enable_detailed = os.getenv('EXPORT_NAME', 'false').lower() == 'true'
        # worker_idx = getattr(env, 'worker_index', 0)
        # env_idx = getattr(env, 'env_index', 0)
        worker_idx = None
        env_idx = None

        # 方法1: env_config から取得（最優先）
        if hasattr(env, 'env_config'):
            env_config = env.env_config
            worker_idx = getattr(env_config, 'worker_idx', None) or getattr(env_config, 'worker_index', None)
            env_idx = getattr(env_config, 'env_idx', None) or getattr(env_config, 'vector_index', None)
            log.debug(f'ModifierWrapper: Got from env_config: worker={worker_idx}, env={env_idx}')
        
        # 方法2: 直接属性を確認（フォールバック）
        if worker_idx is None:
            if hasattr(env, 'worker_idx'):
                worker_idx = env.worker_idx
            elif hasattr(env, 'worker_index'):
                worker_idx = env.worker_index
        
        if env_idx is None:
            if hasattr(env, 'env_idx'):
                env_idx = env.env_idx
            elif hasattr(env, 'env_index'):
                env_idx = env.env_index
            elif hasattr(env, 'vector_index'):
                env_idx = env.vector_index
        # if worker_idx is None and hasattr(env, 'env_config'):
        #     worker_idx = getattr(env.env_config, 'worker_index', None)
        # if env_idx is None and hasattr(env, 'env_config'):
        #     env_idx = getattr(env.env_config, 'vector_index', None)
        
        # 方法3: ラッパーチェーンを辿って取得（最終フォールバック）
        if worker_idx is None or env_idx is None:
            current = env
            depth = 0
            while current is not None and depth < 20:
                if worker_idx is None:
                    if hasattr(current, 'env_config'):
                        env_config = current.env_config
                        worker_idx = getattr(env_config, 'worker_idx', None) or getattr(env_config, 'worker_index', None)
                    if worker_idx is None:
                        worker_idx = getattr(current, 'worker_idx', None) or getattr(current, 'worker_index', None)
                
                if env_idx is None:
                    if hasattr(current, 'env_config'):
                        env_config = current.env_config
                        env_idx = getattr(env_config, 'env_idx', None) or getattr(env_config, 'vector_index', None)
                    if env_idx is None:
                        env_idx = getattr(current, 'env_idx', None) or getattr(current, 'env_index', None) or getattr(current, 'vector_index', None)
                
                if worker_idx is not None and env_idx is not None:
                    break
                
                current = getattr(current, 'env', None)
                depth += 1
        
        # デフォルト値
        final_worker_idx = worker_idx if worker_idx is not None else 0
        final_env_idx = env_idx if env_idx is not None else 0
        
        log.info(f'ModifierWrapper init: worker={final_worker_idx}, env={final_env_idx}, experiment={experiment}')

        #is_training = getattr(env, 'is_training', True)  # デフォルトは学習モード
        self.item_tracker = ItemTracker(
            experiment_name=experiment, 
            mode=mode,
            #worker_idx=worker_idx,
            worker_idx=final_worker_idx,
            #env_idx=env_idx
            env_idx=final_env_idx,
            enable_detailed_logging=enable_detailed
        )

        # 評価フラグの初期化
        self.evaluation = getattr(env, 'evaluation', False)
        self.eval_target = getattr(env, 'eval_target', None)

        self.skill_to_int = {string: i for i, string 
            in enumerate(['discoverer', 'descender', 'ascender', 'worshipper', 'merchant'])}
        self.int_to_skill = {i: string for i, string 
            in enumerate(['discoverer', 'descender', 'ascender', 'worshipper', 'merchant'])}

        obs_spaces = {
            "option": gym.spaces.Box(0, num_skills+1, shape=(1,), dtype=np.int64),
            "buc": gym.spaces.Box(0, 1, shape=(1,), dtype=np.int64),
        }
        obs_spaces.update(
            [(k, self.env.observation_space[k]) for k in self.env.observation_space]
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        intrinsic_reward = 0.0
        extrinsic_reward = reward

        # Calculate intrinsic reward if llm_reward > 0
        if self.llm_reward > 0.:
            intrinsic_reward = reward * self.llm_reward  # Example calculation, adjust as needed
            extrinsic_reward = reward - intrinsic_reward

        #if reward > 0.:
            #log.info(f"Reward: {reward}, Action: {self.actions[action]}")

        msg_str = self.env.message[1]
        cur_buc = 0
        self.dlvl = dlvl = obs['blstats'][12]
        self.xlvl = xlvl = obs['blstats'][18]
        self.dungeon_number = obs['blstats'][23]
        self.depth = obs['blstats'][12]

        # 基本的な統計情報更新（nethack_playerがNoneでも実行）
        self.max_level_reached = max(self.max_level_reached, dlvl)

        if self.env.env.env.env.env.branch_dlvl != -2:
            if self.nethack_player is not None:
                self.nethack_player.branch_depth = self.env.env.env.env.env.branch_dlvl

        if self.nethack_player is not None:
            # 必要な属性が存在する場合のみskill_preconditionを呼び出す
            if hasattr(self, 'char_ascii_encodings') and hasattr(self, 'char_ascii_colors') and hasattr(self, 'cur_num_items') and hasattr(self, 'color_map'):
                worshipper_precondition, merchant_precondition = self.nethack_player.skill_precondition(
                    self.char_ascii_encodings, 
                    self.char_ascii_colors, 
                    self.cur_num_items, 
                    self.color_map
                )
            else:
                worshipper_precondition, merchant_precondition = False, False
        else:
            worshipper_precondition, merchant_precondition = False, False

        # 基本的なメッセージ解析（nethack_playerがNoneでも実行）
        # BUC判定
        if b'altar' in msg_str:
            # actionsの有無をチェックせずにBUC統計を取得
            if b'cursed' not in msg_str and b'blessed' not in msg_str:
                # Dropアクションかどうかの判定を簡略化
                if hasattr(self, 'actions') and action < len(self.actions) and self.actions[action] == Command.DROP:
                    self.num_buc += 1
                    cur_buc = 1
                # actionsが存在しない場合でも、DROP actionの番号による判定を試す
                elif not hasattr(self, 'actions') and action == 20:  # DROP actionの一般的な番号
                    self.num_buc += 1
                    cur_buc = 1
            self.altar_seen = True

        msg_str = self.env.message[1]
        # アイテム追跡の実行
        try:
            # self.item_tracker.update_from_obs(obs)
            # self.item_tracker.update_usage_from_message(msg_str)
            self.item_tracker.update_from_obs(obs, action, msg_str)
        except Exception as e:
            log.warning(f"ItemTracker update failed: {e}")

        # 売却統計
        if b'sold' in msg_str:
            if self.price_id:
                self.num_price_id += 1
                self.price_id = False
            self.num_sold += 1

        if b'Sell' in msg_str:
            self.price_id = is_item_identified(msg_str.decode('utf-8'))
            self.num_sell += 1
            self.shop_seen = True
        else:
            self.price_id = False

        if self.llm_reward > 0. and self.nethack_player is not None:

            if self.evaluation and 'discoveryhunger' in self.eval_target:
                if obs['blstats'][23] == 2 and info['just_eaten']:
                    self.nethack_player.eaten_food = True
                
                if self.nethack_player.eaten_food:
                    self.env.env.env.env.env.eval_prereq_met = True

            if self.evaluation and 'goldenexit' in self.eval_target:
                self.nethack_player.update_gold(info['gold'])
                if info['kill']:
                    self.nethack_player.defeat_monster()

                if self.nethack_player.monsters_defeated >= 25 and self.nethack_player.gold_pieces >= 20 and self.max_level_reached >= 3:
                    self.env.env.env.env.env.eval_prereq_met = True

            if self.evaluation and 'levelupsell' in self.eval_target:
                self.nethack_player.update_xp_level(self.xlvl)

            preconditions = [worshipper_precondition, merchant_precondition]

            self.skill_time = obs['blstats'][20] - self.skill_start_time

            skill_str = self.int_to_skill[self.skill]
            player_skill_end = self.nethack_player.skill_termination(
                    skill_str,
                    self.skill_time,
                    self.depth,
                    self.previous_depth,
                    preconditions,
            )

            if player_skill_end:
                if self.env.env.env.env.env.branch_dlvl == -2 and self.dungeon_number != 0 and self.skill == 1:
                    self.env.env.env.env.env.branch_dlvl = self.previous_depth
                    self.nethack_player.branch_depth = self.env.env.env.env.env.branch_dlvl

                skill_str = self.int_to_skill[self.skill]
                skill_str = self.nethack_player.perform_task(skill_str, 
                                                            self.depth, 
                                                            self.dungeon_number, 
                                                            merchant_precondition, 
                                                            worshipper_precondition
                                                            )
                if skill_str not in self.skill_to_int:
                    print("obs[blstats]: ", obs['blstats'])
                    print(f"brancH_dlvl: {self.env.env.env.env.env.branch_dlvl}")
                    print(f"branch_number: {self.dungeon_number}")
                
                self.skill = self.skill_to_int[skill_str]

                self.skill_start_time = obs['blstats'][20].copy()
                self.previous_depth = self.depth.copy()

            self.skill_end = player_skill_end

            self.prev_xlvl = self.xlvl
            self.prev_msg = msg_str
        else:
            # nethack_playerがNoneの場合
            self.skill_end = False
            self.prev_xlvl = self.xlvl
            self.prev_msg = msg_str

        obs["blstats"] = np.append(obs["blstats"], self.skill_vector[self.skill])
        obs['option'] = np.array([self.skill]).astype(np.int64)
        obs['buc'] = np.array([cur_buc]).astype(np.int64)
        info['branch_id'] = int(getattr(self, 'branch_dlvl', -2) != -2)
        info['dungeon_number'] = self.dungeon_number
        info['depth'] = self.depth
        info['skill'] = self.skill
        info['skill_end'] = self.skill_end
        info['num_buc'] = self.num_buc
        info['num_sold'] = self.num_sold
        info['num_price_id'] = self.num_price_id
        info['num_sell'] = self.num_sell
        info['altar_seen'] = int(self.altar_seen)
        info['shop_seen'] = int(self.shop_seen)
        info['skill_start_time'] = self.skill_start_time
        info['done'] = int(done)
        info['max_level_reached'] = self.max_level_reached

        info['intrinsic_reward'] = intrinsic_reward
        info['extrinsic_reward'] = extrinsic_reward

        # エピソード終了時の処理
        if done:
            try:
                ep_stats = self.item_tracker.episode_summary()
                for k, v in ep_stats.items():
                    info[f'item_{k}'] = v
                timestep = int(obs.get('blstats', [0]*30)[20]) if 'blstats' in obs else 0
                meta = {
                    'timestep': timestep,
                    'episode_length': timestep,
                    'reward': float(reward),
                }
                #self.item_tracker.on_episode_end(meta=meta)
                self.item_tracker.on_episode_end()
            except Exception as e:
                log.warning(f"ItemTracker episode end failed: {e}")

        # if done:
        #     episode_length = obs.get('blstats', [0]*30)[20] if 'blstats' in obs else 0  # timestep
        #     self.item_tracker.end_episode(
        #         episode_length=episode_length,
        #         episode_reward=extrinsic_reward
        #     )

        # Return extrinsic_reward as reward, intrinsic_reward separately in info
        return obs, extrinsic_reward, done, info

    def reset(self):
        self.prev_msg = b''
        self.num_buc = 0
        self.num_sold = 0
        self.num_sell = 0
        self.num_price_id = 0

        # アイテム追跡機能のリセット
        self.item_tracker.reset_episode()

        # start with dummy values and the initiate all necessary attributes
        if self.meta_policy_class is not None:
            self.nethack_player = self.meta_policy_class(max_depth=-1, branch_depth=-2)
            self.nethack_player.set_initial_values()
            self.skill = self.skill_to_int[self.nethack_player.skill]
        else:
            self.nethack_player = None
            self.skill = 0
        
        self.previous_depth = 1
        self.skill_start_time = 0

        obs = self.env.reset()

        # 共通の統計情報初期化（nethack_playerがNoneでも必要）
        self.altar_seen = False
        self.shop_seen = False
        self.price_id = False
        self.xlvl = obs['blstats'][18]
        self.prev_xlvl = self.xlvl
        self.max_level_reached = 1
        self.skill_end = False

        obs["blstats"] = np.append(obs["blstats"], self.skill_vector[self.skill])
        obs['option'] = np.array([self.skill]).astype(np.int64)
        obs['buc'] = np.array([0]).astype(np.int64)  # 初期値を明示的に設定

        return obs
    
    ## 追記 ##
    def close(self):
        """ワーカー終了時の自動保存"""
        try:
            # 統計が収集されている場合のみ保存
            if hasattr(self, 'item_tracker') and self.item_tracker:
                worker_idx = getattr(self.item_tracker, 'worker_idx', -1)
                env_idx = getattr(self.item_tracker, 'env_idx', -1)
                total_episodes = self.item_tracker.total_episodes
                
                log.info(f'ModifierWrapper close called: worker={worker_idx}, env={env_idx}, episodes={total_episodes}')
                
                # エピソードが1つでも完了していれば保存
                if total_episodes > 0:
                    experiment_name = getattr(self, 'experiment', 'default')
                    save_dir = os.path.join("train_dir", "skill_policy", experiment_name, "item_stats", "final")
                    json_path = self.save_item_statistics(save_dir)
                    log.info(f'✓ Saved item statistics: worker={worker_idx}, env={env_idx}, file={json_path}')
                else:
                    log.debug(f'No episodes completed for worker={worker_idx}, env={env_idx}')
            else:
                log.debug('No item_tracker found in ModifierWrapper')
        except Exception as e:
            log.error(f'Failed to save item statistics: {e}', exc_info=True)
        
        # 親クラスのclose処理
        try:
            if hasattr(super(), 'close'):
                super().close()
        except Exception as e:
            log.warning(f'Parent close() failed: {e}')
        
    ## 追記 ##

    def save_item_statistics(self, save_dir=None):
        """APPOから呼び出されるセッション統計保存"""
        if not hasattr(self, 'item_tracker') or self.item_tracker is None:
            return None, None

        if save_dir is None:
            experiment_name = getattr(self, 'experiment', 'default')
            save_dir = os.path.join("train_dir", "skill_policy", experiment_name, "item_stats")

        try:
            os.makedirs(save_dir, exist_ok=True)
            json_path = self.item_tracker.save_worker_stats(save_dir)  # メソッド名変更
            log.info(f'Item stats saved: {json_path}')
            return json_path, json_path  # 両方ともjson_pathを返す
        except Exception as e:
            log.error(f"Failed to save item statistics: {e}")
            return None, None

    def get_item_summary(self):
        """
        アイテム統計のサマリーを取得
        
        Returns:
            dict: アイテム統計のサマリー辞書
        """
        #return self.item_tracker.get_usage_stats()
        if hasattr(self, 'item_tracker'):
            return self.item_tracker.get_worker_statistics()
        return {}
    
    def get_normalized_item_summary(self):
        """正規化されたアイテム統計のサマリーを取得"""
        if hasattr(self, 'item_tracker'):
            # return self.item_tracker.get_normalized_item_stats()  # 削除
            stats = self.item_tracker.get_worker_statistics()
            return stats.get('base_item_actions', {})  # 修正！
        return {}
    
    def get_glyph_statistics(self):
        """
        グリフベースアイテム統計を取得
        
        Returns:
            dict: グリフベースアイテム統計辞書
        """
        return self.item_tracker.get_usage_stats(glyph_based=True)
    
    def get_detailed_item_statistics(self):
        """
        詳細アイテム統計を取得
        
        Returns:
            dict: 詳細アイテム統計辞書
        """
        return self.item_tracker.get_usage_stats(detailed=True)
    
    def print_current_item_stats(self):
        """現在のアイテム統計をコンソールに表示"""
        if hasattr(self, 'item_tracker'):
            # self.item_tracker.print_summary(show_detailed=True, show_glyph=True)  # 削除
            self.item_tracker.print_action_summary()  # 修正！パラメータも削除
