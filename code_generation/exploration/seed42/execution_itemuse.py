class NetHackPlayer:
    def __init__(self):
        self.skill = "none"
        self.food_skill_steps = 1
        self.scroll_skill_steps = 1

    def select_skill(self, current_skill, food_precondition, scroll_precondition):
        if food_precondition:
            return "food_eat"
        if scroll_precondition:
            return "scroll_read"
        return "none"

    def perform_task(
        self,
        current_skill,
        dungeon_depth,
        branch_number,
        food_precondition,
        scroll_precondition,
    ):
        return self.select_skill(current_skill, food_precondition, scroll_precondition)

    def skill_termination(self, skill, skill_time, current_depth, previous_depth, preconditions):
        if skill in ("none", "food_eat", "scroll_read"):
            return skill_time >= 1
        return False

    def skill_precondition(self, char_ascii_encodings, char_ascii_colors, num_items, color_map):
        return False, False

    def set_initial_values(self):
        self.skill = "none"
        self.food_skill_steps = 1
        self.scroll_skill_steps = 1
