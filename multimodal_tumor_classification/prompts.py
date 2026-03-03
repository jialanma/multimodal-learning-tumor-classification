"""Few-shot example selection and prompt construction for the Ovis2 VLM."""

import random

from .config import LABEL_NAMES, RANDOM_SEED


def select_few_shot_examples(patients: list, exclude_pid: str) -> list:
    """
    Pick 1 patient per grade (3 total), excluding the current patient.
    Uses a patient-specific deterministic seed.
    """
    rng = random.Random(hash(exclude_pid) + RANDOM_SEED)

    by_grade = {0: [], 1: [], 2: []}
    for p in patients:
        if p["pid"] != exclude_pid:
            by_grade[p["label"]].append(p)

    for grade in by_grade:
        by_grade[grade].sort(key=lambda p: p["pid"])

    examples = []
    for grade in [0, 1, 2]:
        if by_grade[grade]:
            chosen = rng.choice(by_grade[grade])
            examples.append({
                "label": grade,
                "clinical_text": chosen["clinical_text"],
            })
    return examples


def format_few_shot_block(examples: list) -> str:
    """Format examples as text lines."""
    lines = []
    for ex in examples:
        grade_str = LABEL_NAMES[ex["label"]]
        lines.append(f"  Patient: {ex['clinical_text']} -> {grade_str}")
    return "\n".join(lines)


def build_prompt(patient: dict, few_shot_block: str) -> str:
    """Build the full classification prompt for the VLM."""
    return f"""This is a breast DCE-MRI composite image where:
- Red channel = pre-contrast (baseline tissue signal)
- Green channel = first post-contrast (after gadolinium injection)
- Blue channel = subtraction (post-contrast minus pre-contrast, highlights enhancement)
Bright blue/cyan areas indicate strong contrast uptake, often seen in malignant tissue.

Here are example patients with known Nottingham histologic grades:
{few_shot_block}

Now classify this new patient.
Clinical data: {patient['clinical_text']}

Based on the image and clinical data, what is the Nottingham histologic grade?
Reply with ONLY: Grade 1, Grade 2, or Grade 3"""
