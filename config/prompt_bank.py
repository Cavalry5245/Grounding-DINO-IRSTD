# config/prompt_bank.py
# Multi-Granularity Prompt Bank for Infrared Small Target Detection
# 多粒度提示词库 - 用于红外小目标检测的语言空间数据增强

PROMPT_BANK = {
    # ===== Level 1: Generic Descriptions (通用描述) =====
    "generic": [
        "infrared small target",
        "small object",
        "point target",
        "tiny target",
        "small dot",
    ],

    # ===== Level 2: Appearance Features (外观特征) =====
    "appearance": [
        "bright spot",
        "bright point",
        "hot point",
        "luminous dot",
        "glowing spot",
        "bright dot in the image",
    ],

    # ===== Level 3: Physical Characteristics (物理特性) =====
    "physical": [
        "heat source",
        "thermal emitter",
        "infrared radiation source",
        "hot object",
        "warm spot",
    ],

    # ===== Level 4: Contextual Descriptions (场景描述) =====
    "contextual": [
        "small target in the sky",
        "dim object in night vision",
        "bright point against dark background",
        "small airborne target",
        "distant object in infrared imagery",
    ],

    # ===== Level 5: Size-aware Descriptions (尺寸感知) =====
    "size_aware": [
        "sub-pixel target",
        "few-pixel target",
        "compact target",
        "minimal visible spot",
    ],
}

# Default prompt categories to use (可配置)
DEFAULT_PROMPT_CATEGORIES = [
    "generic",
    "appearance",
    "physical",
]

# Sampling strategy configuration
PROMPT_SAMPLING_CONFIG = {
    "mode": "random",              # Options: "random" | "weighted" | "cycle"
    "num_prompts_per_image": 3,    # Number of prompts to sample per image
    "min_categories": 2,           # Minimum number of different categories to sample from
}

def get_all_prompts(categories=None):
    """
    Get all prompts from specified categories.

    Args:
        categories: List of category names. If None, uses DEFAULT_PROMPT_CATEGORIES.

    Returns:
        List of all prompt strings.
    """
    if categories is None:
        categories = DEFAULT_PROMPT_CATEGORIES

    all_prompts = []
    for cat_name in categories:
        if cat_name in PROMPT_BANK:
            all_prompts.extend(PROMPT_BANK[cat_name])

    return list(set(all_prompts))  # Remove duplicates

if __name__ == "__main__":
    # Test the prompt bank
    print("=" * 60)
    print("Multi-Granularity Prompt Bank for Infrared Small Target Detection")
    print("=" * 60)

    for level, (cat_name, prompts) in enumerate(PROMPT_BANK.items(), 1):
        print(f"\nLevel {level}: {cat_name.upper()}")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. {prompt}")

    print("\n" + "=" * 60)
    print(f"Total prompts available: {sum(len(v) for v in PROMPT_BANK.values())}")
    print(f"Default categories: {DEFAULT_PROMPT_CATEGORIES}")
    print(f"Prompts in default categories: {get_all_prompts()}")
    print("=" * 60)
