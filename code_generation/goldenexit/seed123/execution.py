
import random

class NetHackPlayer:
    def __init__(self, max_depth, branch_depth):
        self.max_depth = max_depth
        self.branch_depth = branch_depth
        self.explored_levels = set()
        self.direction = 'down'  # Start by going down
        self.gold_collected = 0
        self.monsters_defeated = 0
        self.current_level = 1  # Track the current level in Dungeons of Doom

    def merchant_precondition(self):
        # Placeholder for actual merchant precondition logic
        return False

    def worshipper_precondition(self):
        # Placeholder for actual worshipper precondition logic
        return False

    def select_skill(self, current_skill, dungeon_depth, merchant_precondition, worshipper_precondition):
        if merchant_precondition:
            return 'merchant'
        if worshipper_precondition:
            return 'worshipper'

        # Prioritize discovering until we have enough gold and monsters defeated
        if self.gold_collected < 20 or self.monsters_defeated < 25:
            return 'discoverer'
        
        # If we have enough gold and monsters defeated, we can consider descending
        if dungeon_depth < self.max_depth:
            return 'descender'
        else:
            return 'ascender'  # Ascend if at max depth

    def reach_gnomish_mines(self, current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition):
        if branch_number == 0:
            if dungeon_depth == self.branch_depth:
                return 'descender'
            elif dungeon_depth == self.branch_depth + 1:
                return 'ascender'
        elif branch_number == 2:
            return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)
        return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)

    def reach_dungeons_of_doom(self, current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition):
        if dungeon_depth == self.branch_depth:
            if branch_number == 2:
                return 'ascender'
            else:
                return 'descender'
        elif branch_number == 2 and dungeon_depth == self.branch_depth + 1:
            return 'ascender'
        else:
            return self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)

    def perform_task(self, current_skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition):
        skill = self.select_skill(current_skill, dungeon_depth, merchant_precondition, worshipper_precondition)

        # Simulate collecting gold and defeating monsters
        if skill in ['discoverer', 'descender', 'ascender']:
            self.gold_collected += random.randint(1, 5)  # Collect 1 to 5 gold pieces
            self.monsters_defeated += random.randint(1, 3)  # Defeat 1 to 3 monsters

        # If at depth 1 and using ascender, we quit the game
        if skill == 'ascender' and dungeon_depth == 1:
            print("Quitting the game...")
            return skill  # This indicates quitting

        return skill

# Unit test to simulate the player's actions
max_depth = 3
branch_depth = 2
skill = 'discoverer'
dungeon_depth = 1
branch_number = 0

player = NetHackPlayer(max_depth, branch_depth)

for turn in range(20):
    print(f"Turn {turn + 1}: Skill = {skill}, Dungeon depth = {dungeon_depth}, Branch Number = {branch_number}, Gold = {player.gold_collected}, Monsters Defeated = {player.monsters_defeated}")

    if player.gold_collected >= 20 and player.monsters_defeated >= 25:
        # Attempt to reach Gnomish Mines after meeting conditions
        if dungeon_depth == 1:
            skill = 'ascender'  # Try to quit
        else:
            skill = 'descender'  # Continue exploring if not at depth 1
        print("Task was completed!")
        break

    merchant_precondition = player.merchant_precondition()
    worshipper_precondition = player.worshipper_precondition()

    skill = player.perform_task(skill, dungeon_depth, branch_number, merchant_precondition, worshipper_precondition)

    # Update dungeon depth based on skill
    if skill == 'descender':
        dungeon_depth += 1
    elif skill == 'ascender':
        dungeon_depth -= 1

    # Update branch number based on dungeon depth
    if skill == "descender" and dungeon_depth == branch_depth + 1:
        branch_number = random.choice([0, 2])
    elif skill == "ascender" and dungeon_depth == branch_depth and branch_number == 2:
        branch_number = 0

    def skill_termination(self, skill, skill_time, current_depth, previous_depth, preconditions):
        """
        Determines when a skill should terminate.

        Args:
        - skill (str): The current skill being used.
        - skill_time (int): The time the skill has been active.
        - current_depth (int): The current dungeon depth.
        - previous_depth (int): The previous dungeon depth.
        - preconditions (list): A list of preconditions for the Merchant and Worshipper skills.

        Returns:
        - bool: True if the skill should terminate, False otherwise.
        """

        # Check if any preconditions for Merchant or Worshipper are true
        if any(preconditions):
            return True  # Terminate the skill if any preconditions are met

        # Skill-specific termination conditions
        if skill == "discoverer":
            # Terminate when the dungeon is fully explored
            # For simplicity, let's assume the dungeon is fully explored after a certain time
            if skill_time >= self.discoverer_skill_steps:  # Adjust this value as needed
                return True
        elif skill == "descender":
            # Terminate when a staircase is reached and descended
            if current_depth > previous_depth:
                return True
        elif skill == "ascender":
            # Terminate when a staircase is reached and ascended
            if current_depth < previous_depth:
                return True
        elif skill == "merchant":
            # Terminate when all items are sold
            # For simplicity, let's assume all items are sold after a certain time
            if skill_time >= self.merchant_skill_steps:  # Adjust this value as needed
                return True
        elif skill == "worshipper":
            # Terminate when all items are identified
            # For simplicity, let's assume all items are identified after a certain time
            if skill_time >= self.worshipper_skill_steps:  # Adjust this value as needed
                return True

        return False  # Don't terminate the skill if none of the above conditions are met

    def skill_precondition(self, char_ascii_encodings, char_ascii_colors, num_items, color_map):
        """
        Determine preconditions for Worshipper and Merchant skills.

        Args:
            char_ascii_encodings: numpy array representing ASCII encodings of surrounding characters.
            char_ascii_colors: numpy array representing the colors of the surrounding characters.
            num_items: number of items the agent has.
            color_map: a map from common characters to their expected color.

        Returns:
            Tuple (worshipper_precondition, merchant_precondition):
                - worshipper_precondition: True if Worshipper skill can initiate.
                - merchant_precondition: True if Merchant skill can initiate.
        """
        # Fetch relevant colors from the color map
        shopkeeper_color = color_map.get("@", None)  # Shopkeeper
        altar_color = color_map.get("_", None)       # Altar

        # Initialize preconditions
        worshipper_precondition = False
        merchant_precondition = False

        # Check for Merchant precondition
        if shopkeeper_color is not None and num_items > 0:
            shopkeeper_mask = (char_ascii_encodings == 64) & (char_ascii_colors == shopkeeper_color)
            merchant_precondition = shopkeeper_mask.any()

        # Check for Worshipper precondition
        if altar_color is not None and num_items > 0:
            altar_mask = (char_ascii_encodings == 95) & (char_ascii_colors == altar_color)
            worshipper_precondition = altar_mask.any()

        return worshipper_precondition, merchant_precondition

    def set_initial_values(self,):
        # Initiating values from execution.py and from termination.txt
        self.skill = 'discoverer'
        self.max_depth = 3
        self.discoverer_skill_steps = 500 # either 50 or 500
        self.merchant_skill_steps = self.discoverer_skill_steps * 20
        self.worshipper_skill_steps = self.discoverer_skill_steps * 20
