
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

# Starting conditions
skill = "none"
hunger_level = 3  # 0 (not hungry) to 10 (starving)
hp = 12
max_hp = 20
food_in_inventory = 2
scroll_in_inventory = 1

# Start-of-turn state
in_combat = False
scroll_safe = True

player = NetHackPlayer()

for turn in range(20):
    # Environment updates combat/safety for THIS turn first
    in_combat = (turn % 3 == 0)
    scroll_safe = not in_combat

    # Agent observes the environment (synchronization point)
    player.observe_environment(in_combat, scroll_safe)

    print(
        f"Turn {turn + 1}: Skill={skill}, Hunger={hunger_level}, "
        f"HP={hp}/{max_hp}, Food={food_in_inventory}, Scroll={scroll_in_inventory}, "
        f"InCombat={in_combat}, ScrollSafe={scroll_safe}"
    )

    if food_in_inventory == 0 and scroll_in_inventory == 0:
        print("Task was completed!")
        break

    skill = player.perform_task(
        skill,
        hunger_level,
        hp,
        max_hp,
        food_in_inventory,
        scroll_in_inventory,
        in_combat,
        scroll_safe,
    )

    # Apply chosen skill effects using the SAME combat/safety state
    if skill == "food_eat" and food_in_inventory > 0:
        food_in_inventory -= 1
        hunger_level = max(0, hunger_level - 4)
        hp = min(max_hp, hp + 2)
    elif skill == "scroll_read" and scroll_in_inventory > 0 and scroll_safe:
        scroll_in_inventory -= 1
        hp = min(max_hp, hp + 1)

    # Environment hunger progression at end of turn
    hunger_level = min(10, hunger_level + 1)

    # additional method calls to update attributes go here
