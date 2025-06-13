
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

        self.num_items = min(
                self.bl_norm.shape[0],
                env.observation_space['blstats'].shape[0]
            )
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

        norm_blstats = (obs["blstats"] * self.bl_norm[:self.num_items])
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
        msg_str = self.env.message[1]
        self.dlvl = dlvl = obs['blstats'][12]
        self.xlvl = xlvl = obs['blstats'][18]
        self.dungeon_number = obs['blstats'][23]
        self.depth = obs['blstats'][12]

        # update the branch dungeon level
        if self.env.env.env.env.env.branch_dlvl != -2:
            self.nethack_player.branch_depth = self.env.env.env.env.env.branch_dlvl

        worshipper_precondition, merchant_precondition = self.nethack_player.skill_precondition(
                self.char_ascii_encodings, 
                self.char_ascii_colors, 
                self.cur_num_items, 
                self.color_map
        )

        if self.llm_reward > 0.:

            if self.evaluation and 'discoveryhunger' in self.eval_target:
                # Verify that the agent has eaten in the Gnomish Mines
                if obs['blstats'][23] == 2 and info['just_eaten']:
                    self.nethack_player.eaten_food = True
                
                # for evaluating performance
                if self.nethack_player.eaten_food:
                    self.env.env.env.env.env.eval_prereq_met = True
                # for evaluating performance

            if self.evaluation and 'goldenexit' in self.eval_target:
                self.nethack_player.update_gold(info['gold'])
                if info['kill']:
                    self.nethack_player.defeat_monster()

                # for evaluating performance
                self.max_level_reached = max(self.max_level_reached, dlvl)
                if self.nethack_player.monsters_defeated >= 25 and self.nethack_player.gold_pieces >= 20 and self.max_level_reached >= 3:
                    self.env.env.env.env.env.eval_prereq_met = True
                # for evaluating performance

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

                # keeping track of branch_dlvl
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

            # some tab-keeping
            if worshipper_precondition:
                self.altar_seen = True
            if merchant_precondition:
                self.shop_seen = True

            cur_buc = 0
            if b'altar' in msg_str:
                if b'cursed' not in msg_str and b'blessed' not in msg_str and self.actions[action] == Command.DROP:
                    self.num_buc += 1
                    cur_buc = 1

            if b'sold' in msg_str:
                if self.price_id:
                    self.num_price_id += 1
                    self.price_id = False
                self.num_sold += 1
                if self.evaluation and 'levelupsell' in self.eval_target:
                    self.env.env.env.env.env.eval_prereq_met = True

            if b'Sell' in msg_str:
                self.price_id = is_item_identified(msg_str.decode('utf-8'))
                self.num_sell += 1
            else:
                self.price_id = False
            # some tab-keeping

            self.prev_xlvl = self.xlvl
            self.prev_msg = msg_str

        obs["blstats"] = np.append(obs["blstats"], self.skill_vector[self.skill])
        obs['option'] = np.array([self.skill]).astype(np.int64)
        obs['buc'] = np.array([cur_buc]).astype(np.int64)
        info['branch_id'] = int(self.branch_dlvl != -2)
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

        return obs, reward, done, info

    def reset(self):
        self.prev_msg = b''
        self.num_buc = 0
        self.num_sold = 0
        self.num_sell = 0
        self.num_price_id = 0

        # start with dummy values and the initiate all necessary attributes
        self.nethack_player = self.meta_policy_class(max_depth=-1, branch_depth=-2)
        self.nethack_player.set_initial_values()
        
        self.skill = self.skill_to_int[self.nethack_player.skill]
        self.previous_depth = 1
        self.skill_start_time = 0

        obs = self.env.reset()

        self.altar_seen = False
        self.shop_seen = False
        self.price_id = False
        self.xlvl = obs['blstats'][18]
        self.prev_xlvl = self.xlvl
        self.max_level_reached = 1

        obs["blstats"] = np.append(obs["blstats"], self.skill_vector[self.skill])
        obs['option'] = np.array([self.skill]).astype(np.int64)

        return obs
