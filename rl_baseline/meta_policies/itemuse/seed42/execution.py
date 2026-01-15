
class NetHackPlayer:
    def __init__(self, max_depth=None, branch_depth=None):
        self.max_depth = max_depth
        self.branch_depth = branch_depth
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
        dungeon_depth,
        branch_number,
        food_precondition,
        scroll_precondition,
    ):
        if food_precondition:
            self.skill = "food_eat"
        elif scroll_precondition:
            self.skill = "scroll_read"
        else:
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
