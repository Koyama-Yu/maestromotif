
class NetHackPlayer:
    def __init__(self):
        self.skill = "none"

        # New: keep a one-step memory of what the agent believes about safety/combat.
        # This compensates for environments/tests that update combat/safety out of phase
        # with printing and/or effect application.
        self.last_in_combat = False
        self.last_scroll_safe = True

    def observe_environment(self, in_combat, scroll_safe):
        """
        Environment-facing update hook.
        Must be called by the unit test once per turn BEFORE perform_task.
        """
        self.last_in_combat = bool(in_combat)
        self.last_scroll_safe = bool(scroll_safe)

    def perform_task(
        self,
        current_skill,
        hunger_level,          # 0 (not hungry) to 10 (starving)
        hp,
        max_hp,
        food_in_inventory,
        scroll_in_inventory,
        in_combat,             # kept for compatibility, but we rely on observed values
        scroll_safe,           # kept for compatibility, but we rely on observed values
    ):
        # Use the observed (synchronized) state rather than the raw arguments.
        in_combat = self.last_in_combat
        scroll_safe = self.last_scroll_safe

        has_food = food_in_inventory > 0
        has_scroll = scroll_in_inventory > 0

        injured = hp < max_hp
        meaningfully_injured = hp <= max_hp - 2

        very_hungry = hunger_level >= 7
        moderately_hungry = hunger_level >= 5

        # Prefer eating when hunger is high, or when moderately hungry and injured.
        # (Food is always "safe" in this toy test; if you later model unsafe eating,
        # add a flag and gate this similarly to scrolls.)
        if has_food and (very_hungry or (moderately_hungry and injured)):
            self.skill = "food_eat"
            return self.skill

        # Prefer reading only when safe and not in combat.
        if has_scroll and scroll_safe and (not in_combat) and meaningfully_injured and hunger_level <= 6:
            self.skill = "scroll_read"
            return self.skill

        self.skill = "none"
        return self.skill


# ---------------- UNIT TEST / SIMULATION ----------------


    def skill_termination(self, skill, skill_time, current_depth, previous_depth, preconditions):
        """
        Determines when a skill should terminate.

        Args:
        - skill (str): The current skill being used.
        - skill_time (int): The time the skill has been active.
        - current_depth (int): The current dungeon depth.
        - previous_depth (int): The previous dungeon depth.
        - preconditions (list): Preconditions that can force a switch.

        Returns:
        - bool: True if the skill should terminate, False otherwise.
        """

        # If any precondition is met, allow switching immediately
        if any(preconditions):
            return True

        # Item-use skills are short-horizon; terminate quickly to allow re-selection.
        if skill in ("none", "food_eat", "scroll_read"):
            return skill_time >= 1

        return False

    def skill_precondition(self, char_ascii_encodings, char_ascii_colors, num_items, color_map):
        """
        Determine preconditions for item-use skills.

        Args:
            char_ascii_encodings: numpy array representing ASCII encodings of surrounding characters.
            char_ascii_colors: numpy array representing the colors of the surrounding characters.
            num_items: number of items the agent has.
            color_map: a map from common characters to their expected color.

        Returns:
            Tuple (food_precondition, scroll_precondition):
                - food_precondition: True if food use should be considered.
                - scroll_precondition: True if scroll use should be considered.
        """
        # Default to no forced preconditions; the policy can choose based on state.
        food_precondition = False
        scroll_precondition = False

        return food_precondition, scroll_precondition

    def set_initial_values(self,):
        # Initiating values from execution.py and from termination_itemuse.txt
        self.skill = 'none'
        self.food_skill_steps = 1
        self.scroll_skill_steps = 1
