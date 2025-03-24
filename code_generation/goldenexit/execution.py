
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
