exec
# Copyright (c) Facebook, Inc. and its affiliates.
import enum
import re
import time

import gym
import numpy as np
from gym.envs import registration
from nle import nethack
from nle.env import base
from nle.env.base import ASCII_ESC, ASCII_SPACE, ASCII_y, ASCII_n
from nle.nethack import NETHACKOPTIONS

from sample_factory.utils.utils import log
from utils.forked_pdb import ForkedPdb

TASK_ACTIONS = tuple(
    [nethack.MiscAction.MORE]
    + list(nethack.CompassDirection)
    + list(nethack.CompassDirectionLonger)
    + list(nethack.MiscDirection)
    + [
        nethack.Command.KICK,
        nethack.Command.EAT,
        nethack.Command.SEARCH,
    ]
    + [
        nethack.Command.CAST,
        nethack.Command.ENGRAVE,
        nethack.Command.ENHANCE,
        nethack.Command.QUAFF,
        nethack.Command.PRAY,
    ]
)

CUSTOM_OPTIONS=("@.nethackrc",)

SKIP_EXCEPTIONS = (b"eat", b"attack", b"direction?", b"pray", b"drop", b'drink', b'Sell')


class NetHackScoreMonk(base.NLE):
    def __init__(
        self,
        *args,
        actions=None,
        max_episode_steps=5000,
        penalty_mode="none",
        penalty_step: float = -0.0005,
        penalty_time: float = -0.0,
        experiment='',
        evaluation=False,
        eval_target='none',
        **kwargs,
    ):
        self.penalty_mode = penalty_mode
        self.penalty_step = penalty_step
        self.penalty_time = penalty_time
        self.dungeon_explored = {}
        self.dungeon_explored_custom = {}
        self.last_action = None
        self.enhance = False
        self.timestep = 0
        self.penalty = 0
        self._frozen_steps = 0
        self.dlvl = 1
        self.branch_dlvl = -2
        self.experiment = experiment
        self.extra_timestep_start = 0
        self.color_map = {'@': 15, '_': 7}
        self.done = False

        self.reached = False
        self.evaluation = evaluation
        self.eval_target = eval_target
        # Tab-keeping for evaluating performance
        if self.eval_target in ['discoveryhunger', 'goldenexit', 'levelupsell']:
            self.eval_prereq_met = False 
        else:
            self.eval_prereq_met = True

        if actions is None:
            actions = kwargs.pop("actions", TASK_ACTIONS)

        super().__init__(
            *args, 
            actions=actions, 
            options=CUSTOM_OPTIONS, 
            max_episode_steps=max_episode_steps,
            **kwargs)

    def _get_time_penalty(self, last_observation, observation):
        blstats_old = last_observation[self._blstats_index]
        blstats_new = observation[self._blstats_index]

        old_time = blstats_old[nethack.NLE_BL_TIME]
        new_time = blstats_new[nethack.NLE_BL_TIME]

        if old_time == new_time:
            self._frozen_steps += 1
        else:
            self._frozen_steps = 0

        penalty = 0
        if self.penalty_mode == "constant":
            if self._frozen_steps > 0:
                penalty += self.penalty_step
        elif self.penalty_mode == "exp":
            penalty += 2**self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "square":
            penalty += self._frozen_steps**2 * self.penalty_step
        elif self.penalty_mode == "linear":
            penalty += self._frozen_steps * self.penalty_step
        elif self.penalty_mode == "always":
            penalty += self.penalty_step
        elif self.penalty_mode == 'none':
            penalty = 0
            self.penalty_time = 0.
        else:  # default
            raise ValueError("Unknown penalty_mode '%s'" % self.penalty_mode)
        penalty += (new_time - old_time) * self.penalty_time
        return penalty

    def get_dungeon_scout_score(self, last_observation):
        glyphs = last_observation[self._glyph_index]
        blstats = last_observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        if key in self.dungeon_explored_custom:
            total_explored_diff = explored - self.dungeon_explored_custom[key]
            self.dungeon_explored_custom[key] = explored
        else:
            total_explored_diff = 0
            self.dungeon_explored_custom[key] = explored
        return total_explored_diff

    def get_scout_score(self, last_observation):
        glyphs = last_observation[self._glyph_index]
        blstats = last_observation[self._blstats_index]

        dungeon_num = blstats[nethack.NLE_BL_DNUM]
        dungeon_level = blstats[nethack.NLE_BL_DLEVEL]

        key = (dungeon_num, dungeon_level)
        explored = np.sum(glyphs != nethack.GLYPH_CMAP_OFF)
        self.dungeon_explored[key] = explored
        total_explored = 0
        for key, value in self.dungeon_explored.items():
            total_explored += value
        return total_explored

    def _reward_fn(self, last_observation, action, observation, end_status):
        """Score delta."""
        score_diff = super()._reward_fn(
            last_observation, action, observation, end_status
        )
        return score_diff

    def _perform_known_steps(self, observation, done, action=None, exceptions=True):

        while not done:
            message_str = bytes(observation[self._message_index]).decode("utf-8")

            # Macro-action for casting the spell
            if action == nethack.Command.CAST:
                observation, done = self.nethack.step(ord("a"))
                if self.skill_feature[0] == 0:
                    observation, done = self.nethack.step(ord("."))
            # Enhance skills whenever possible
            elif "You feel more confident in" in message_str:
                self.enhance = True

            if observation[self._internal_index][3]:  # xwaitforspace
                if "You feel more confident in" in message_str:
                    self.enhance = True

                # Make sure to include information about going down the stairs.
                previous_msg = observation[self._message_index].copy()
                previous_msg_str = bytes(previous_msg)
                observation, done = self.nethack.step(ASCII_SPACE)
                action = ASCII_SPACE
                cur_msg_str = bytes(observation[self._message_index])

                # Fix for some messages disappearing
                if b"You descend the stairs." in previous_msg_str:
                    observation = (
                        *observation[: self._message_index],
                        previous_msg,
                        *observation[self._message_index + 1 :],
                    )
                elif b'You climb up the stairs' in previous_msg_str:
                    observation = (
                        *observation[: self._message_index],
                        previous_msg,
                        *observation[self._message_index + 1 :],
                    )
                elif b'You are frozen' in previous_msg_str:
                    observation = (
                        *observation[: self._message_index],
                        previous_msg,
                        *observation[self._message_index + 1 :],
                    )
                continue

            # Other cases
            internal = observation[self._internal_index]
            in_yn_function = internal[1]
            in_getline = internal[2]

            if in_getline:  # Game asking for a line of text. We don't do that.
                observation, done = self.nethack.step(ASCII_ESC)
                action = ASCII_ESC

            if in_yn_function:  # Game asking for a single character.
                # Note: No auto-yes to final questions thanks to the disclose option.

                # This causes an annoying unnecessary copy...
                msg = bytes(observation[self._message_index])

                if exceptions:
                    # Eval only: stop episode when agent attempts to exit dungeon
                    if self.evaluation and 'goldenexit' in self.eval_target:
                        if b'climb' in msg:
                            break

                    # Do not skip some questions to allow agent to select
                    # stuff to eat, attack, and to select directions.
                    # Also do not skip if all allowed or the allowed message appears.
                    if self._allow_all_yn_questions or any(el in msg for el in SKIP_EXCEPTIONS):
                        break

                # Otherwise, auto-decline.
                observation, done = self.nethack.step(ASCII_ESC)
                action = ASCII_ESC

            if self.enhance:
                observation, done = self.nethack.step(nethack.Command.ENHANCE)
                observation, done = self.nethack.step(ord("a"))
                self.enhance = False

            break

        return observation, done

    def calc_branch_dlvl(self):
        # Helper function to detect when the agent is at the fork
        glyphs = self.last_observation[self._glyph_index]
        if (glyphs == 2383).sum() > 1 and self.branch_dlvl == -2:
            self.branch_dlvl = self.last_observation[self._blstats_index][12]

    def encodings_and_colors(self):
        x, y = self.last_observation[self._blstats_index][:2]
        char_ascii_encodings = self.last_observation[0].copy() # tty_chars
        char_ascii_colors = self.last_observation[1].copy() # tty_colors
        char_ascii_colors[y+1, x] = 0

        char_ascii_encodings = char_ascii_encodings[y - 3 : y + 4, x - 3 : x + 4]
        char_ascii_colors = char_ascii_colors[y - 3 : y + 4, x - 3 : x + 4]

        self.char_ascii_encodings = char_ascii_encodings
        self.char_ascii_colors = char_ascii_colors
        self.cur_num_items = len(self.inv_items)

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)

        self.message = (obs['message'].copy(), bytes(obs['message']).replace(b'\x00', b''))

        self.last_action = None
        self.timestep = 0
        self.penalty = 0
        self.inv_items = []
        self.calc_branch_dlvl()
        self.encodings_and_colors()
        self.branch_dlvl = -2
        # for evaluation
        self.reached = False
        if self.eval_target in ['discoveryhunger', 'goldenexit', 'levelupsell']:
            self.eval_prereq_met = False 
        else:
            self.eval_prereq_met = True
        self.extra_timestep_start = 0 # used for levelupsell

        # Identify the spell class (randomly generated)
        inventory_types = obs['inv_oclasses']
        spellbook_idx = np.where(inventory_types == 10)[0]
        spellbook = bytes(obs['inv_strs'][spellbook_idx])
        if b"healing" in spellbook:
            self.skill_feature = np.array([0])
        elif b"protection" in spellbook:
            self.skill_feature = np.array([1])
        elif b"sleep" in spellbook:
            self.skill_feature = np.array([2])

        # Ientify inventory
        non_comestibles = (self.last_observation[self._inv_oclasses_index] != 7)
        self.starting_inv_letters = set(self.last_observation[self._inv_letters_index][non_comestibles].copy())

        # Add these as a features
        obs["blstats"] = np.append(obs["blstats"], self.skill_feature)
        potions = np.array([(obs['inv_oclasses'] == 8).sum()])
        obs["blstats"] = np.append(obs["blstats"], potions)
        comestibles = np.array([(obs['inv_oclasses'] == 7).sum()])
        obs["blstats"] = np.append(obs["blstats"], comestibles)

        self.done = False

        return obs

    def reached_string(self, array, string):
        # Dimensions of the array
        rows = 24
        cols = 80

        row_strings = []
        # Loop through each row
        for i in range(rows):
            # Convert the whole row from ASCII codes to a string
            row_string = ''.join(chr(array[i][j]) for j in range(cols))

            row_strings.append(row_string)

            if string in row_string:
                return True, row_strings

        return False, row_strings

    def get_inv_items(self, last_observation):
        non_comestibles = (last_observation[self._inv_oclasses_index] != 7) * (last_observation[self._inv_oclasses_index] != 12)
        curr_inv_letters = set(last_observation[self._inv_letters_index][non_comestibles]).copy()
        unique_elements = list(curr_inv_letters.difference(self.starting_inv_letters))
        return unique_elements

    def step(self, action: int):
        """Steps the environment.

        Args:
            action (int): action integer as defined by ``self.action_space``.

        Returns:
            (dict, float, bool, dict): a tuple containing
                - (*dict*): an observation of the state; this will contain the keys
                  specified by ``self.observation_space``.
                - (*float*): a reward; see ``self._reward_fn`` to see how it is
                  specified.
                - (*bool*): True if the state is terminal, False otherwise.
                - (*dict*): a dictionary of extra information (such as
                  `end_status`, i.e. a status info -- death, task win, etc. --
                  for the terminal state).
        """
        # Careful: By default we re-use Numpy arrays, so copy before!
        last_observation = tuple(a.copy() for a in self.last_observation)

        last_msg = bytes(last_observation[self._message_index]).decode('utf-8')

        just_eaten = False

        # Macro-action for eating the food
        if 'What do you want to eat' in last_msg:
            pattern = r'\[([a-zA-Z]+)'
            match = re.search(pattern, last_msg)
            if match and self.actions[action] == ord('y'):
                action = ord(match.group(1)[0])
                just_eaten = True
            else:
                action = ASCII_SPACE
        # Macro-action for drinking the healing potions
        elif 'What do you want to drink' in last_msg:
            pattern = r'\[([a-zA-Z]+)'
            match = re.search(pattern, last_msg)
            if match and self.actions[action] == ord('y'):
                action = ord(match.group(1)[0])
            else:
                action = ASCII_SPACE
        else:
            action = self.actions[action]

        # Macro-action for waiting
        if action == nethack.MiscDirection.WAIT:
            self.nethack.step(ord("2"))
            self.nethack.step(ord("5"))
            observation, done = self.nethack.step(ord("."))
        # Macro-action for dropping items
        elif action == nethack.Command.DROP:
            observation, done = self.nethack.step(action)
            inv_items = self.get_inv_items(last_observation)
            if len(inv_items) > 0:
                observation, done = self.nethack.step(inv_items[0])
            else:
                observation, done = self.nethack.step(ASCII_ESC)
        else:
            observation, done = self.nethack.step(action)

        is_game_over = observation[self._program_state_index][0] == 1

        # perform known steps
        if is_game_over or not self._allow_all_modes:
            observation, done = self._perform_known_steps(
                observation, done, action=action, exceptions=True
            )

        # A bunch of code for evaluating performance
        if self.evaluation:

            # Evaluate whether the agent has "reached" the goal
            if (self.eval_prereq_met 
                and not done 
                and 
                (
                 b'You descend the stairs.' in bytes(observation[self._message_index]) or
                 'levelupsell' in self.eval_target or 
                 'goldenexit' in self.eval_target
                )
                ):

                self.nethack.step(ord("#"))
                self.nethack.step(ord("o"))
                self.nethack.step(ord("v"))
                self.nethack.step(ord("e"))
                self.nethack.step(ord("r"))
                self.nethack.step(ord("v"))
                self.nethack.step(ord("i"))
                self.nethack.step(ord("e"))
                self.nethack.step(ord("w"))
                hid_obs, _ = self.nethack.step(ord("\r"))

                if self.eval_target == 'delphi':
                    # average depth of Delphi
                    strings = ['Level 7']
                elif self.eval_target == 'gnomish':
                    strings = ['Gnomish Mines']
                elif self.eval_target == 'minetown':
                    # average depth of Minetown
                    strings = ['The Gnomish Mines: levels 2 to 4', 
                            'The Gnomish Mines: levels 3 to 5',
                            'The Gnomish Mines: levels 4 to 6',]
                elif self.eval_target == 'bigroom':
                    strings = ['Level 11']
                elif 'discoveryhunger' in self.eval_target:
                    strings = ['Level 7']
                elif 'levelupsell' in self.eval_target:
                    strings = []
                elif 'goldenexit' in self.eval_target:
                    strings = ['climb']
                else:
                    strings = []
                    self.reached = True

                for string in strings:
                    reached, _ = self.reached_string(hid_obs[0], string)
                    self.reached |= reached

                if 'levelupsell' in self.eval_target:
                    self.extra_timestep_start += 1
                    # Survive for another 300 steps
                    if self.extra_timestep_start > 300:
                        self.reached = True

                if self.reached: 
                    done = True

                self.nethack.step(ord("\r"))
        # A bunch of code for evaluating performance

        self.calc_branch_dlvl()
        self._steps += 1
        self.timestep += 1
        self.last_observation = observation
        self.last_action = action
        obs = self._get_observation(observation)
        self.message = (obs['message'].copy(),
                        bytes(obs['message']).replace(b'\x00', b''))

        obs["blstats"] = np.append(obs["blstats"], self.skill_feature)

        potions = np.array([(obs['inv_oclasses'] == 8).sum()])
        obs["blstats"] = np.append(obs["blstats"], potions)

        comestibles = np.array([(obs['inv_oclasses'] == 7).sum()])
        obs["blstats"] = np.append(obs["blstats"], comestibles)

        self.inv_items = self.get_inv_items(observation)
        self.encodings_and_colors()

        if self._check_abort(observation):
            end_status = self.StepStatus.ABORTED
        else:
            end_status = self._is_episode_end(observation)
        end_status = self.StepStatus(done or end_status)

        reward = float(
            self._reward_fn(last_observation, action, observation, end_status)
        )

        if end_status and not done:
            # Try to end the game nicely.
            self._quit_game(observation, done)
            done = True

        self.done = done

        info = {}
        info["dlvl"] = last_observation[self._blstats_index][12]
        info["gold"] = last_observation[self._blstats_index][13]
        info["xlvl"] = last_observation[self._blstats_index][18]
        info["timestep"] = last_observation[self._blstats_index][20]
        info["scout"] = self.get_scout_score(last_observation)
        info["dungeon_scout"] = self.get_dungeon_scout_score(observation)
        info["step"] = self.timestep
        info["reached"] = float(self.reached)
        msg_str = bytes(observation[self._message_index])
        info['kill'] = int(b'kill' in msg_str or b'destroy' in msg_str)
        info['just_eaten'] = just_eaten

        self.penalty = self._get_time_penalty(last_observation, observation)

        #if reward > 0.:
            #log.info("NetHackScoreMonk: Reward: %s, Penalty: %s", reward, self.penalty)

        return obs, reward, done, info


registration.register(
    id="NetHackScoreMonk-v1",
    entry_point=NetHackScoreMonk,
)
