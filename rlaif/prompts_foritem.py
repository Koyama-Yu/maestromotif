retry_prompts = {
    "default": """\nSo, which one is the best? Please respond by saying ("best_description": 1), ("best_description": 2), or ("best_description": None).""",
}

regexes = {
    "default": r"\W*[bB][eE][sS][tT]\\*_*\s*[dD][eE][sS][cC][rR][iI][pP][tT][iI][oO][nN]\W*(?:\s*:*\s*)?(?:\w+\s*)?(1|2|[Nn]one)",
}


system_prompts = {
    "default": """<s>[INST] <<SYS>>
You are a helpful and honest judge of NetHack gameplay. Always answer as helpfully as possible, while being truthful.

If you don't know the answer to a question, please don't share false information.
<</SYS>>\n""",
}

prompt_templates = {
    "default": """
I will present you with two short gameplay descriptions.
First, briefly state your knowledge of NetHack and its goal.
Write an analysis describing the semantics of each description strictly using information from the descriptions and your knowledge of NetHack.
Provide a comparative analysis based on first principles. Here is the preference you should seek: {}.
Finally, respond by explicitly declaring which description best fits your preference, writing either ("best_description": 1) or ("best_description": 2). If neither description is desirable for the preference, say ("best_description": None).

{{"description_1":
{}
}}

{{"description_2":
{}
}}
""",
}

goal_strings = {
    'food_eat': "prefer situations where eating a food item is useful or necessary, and avoid eating when it would be risky or wasteful",
    'food_drop': "prefer situations where dropping a food item is useful (e.g., to manage inventory), and avoid dropping when it is wasteful",
    'potion_quaff': "prefer situations where quaffing a potion is useful, and avoid quaffing when it is risky or wasteful",
    'potion_drop': "prefer situations where dropping a potion is useful, and avoid dropping when it is wasteful",
    'scroll_read': "prefer situations where reading a scroll is useful, and avoid reading when it is risky or wasteful",
    'scroll_drop': "prefer situations where dropping a scroll is useful, and avoid dropping when it is wasteful",
    'wand_zap': "prefer situations where zapping a wand is useful, and avoid zapping when it is risky or wasteful",
    'wand_drop': "prefer situations where dropping a wand is useful, and avoid dropping when it is wasteful",
    'weapon_wield': "prefer situations where wielding a weapon is useful, and avoid wielding when it is risky or wasteful",
    'weapon_throw': "prefer situations where throwing or firing a weapon is useful, and avoid throwing when it is risky or wasteful",
    'weapon_drop': "prefer situations where dropping a weapon is useful, and avoid dropping when it is wasteful",
    'armor_wear': "prefer situations where wearing armor is useful, and avoid wearing when it is risky or wasteful",
    'armor_take_off': "prefer situations where taking off armor is useful, and avoid taking it off when it is risky or wasteful",
    'armor_drop': "prefer situations where dropping armor is useful, and avoid dropping when it is wasteful",
}
