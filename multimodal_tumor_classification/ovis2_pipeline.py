"""Ovis2-4B VLM pipeline: model loading, per-slice inference, patient classification."""

import os
import json
import random
import numpy as np
import torch
from PIL import Image
from collections import Counter
from transformers import AutoConfig, AutoModelForCausalLM

from .config import DEVICE, LABEL_NAMES, LABEL_MAP, SLICES_PER_PATIENT, RANDOM_SEED
from .dataset import build_dataset
from .prompts import select_few_shot_examples, format_few_shot_block, build_prompt
from .evaluation import evaluate_predictions


def load_ovis2_model(device: str = DEVICE):
    """Load frozen Ovis2-4B model."""
    print(f"Loading Ovis2-4B on {device}...")
    config = AutoConfig.from_pretrained("AIDC-AI/Ovis2-4B", trust_remote_code=True)
    config.llm_attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        "AIDC-AI/Ovis2-4B",
        config=config,
        torch_dtype=torch.bfloat16,
        multimodal_max_length=32768,
        trust_remote_code=True,
    ).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    print("Model loaded (all parameters frozen)")
    return model, model.get_text_tokenizer(), model.get_visual_tokenizer()


def parse_grade(text: str) -> int:
    """Parse model output into grade label (0/1/2) or -1 if unparseable."""
    t = text.lower()
    if "grade 3" in t:
        return 2
    if "grade 1" in t:
        return 0
    if "grade 2" in t:
        return 1
    return -1


def classify_slice(model, text_tok, vis_tok,
                   image_path: str, prompt_text: str) -> tuple:
    """Run inference on one slice. Returns (pred_label, raw_text)."""
    img = Image.open(image_path).convert("RGB")
    query = f"<image>\n{prompt_text}"

    _, input_ids, pixel_values = model.preprocess_inputs(
        query, [img], max_partition=9
    )
    attn = torch.ne(input_ids, text_tok.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attn = attn.unsqueeze(0).to(model.device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(dtype=vis_tok.dtype, device=vis_tok.device)

    with torch.inference_mode():
        out_ids = model.generate(
            input_ids,
            pixel_values=[pixel_values],
            attention_mask=attn,
            max_new_tokens=16,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tok.pad_token_id,
        )[0]

    answer = text_tok.decode(out_ids, skip_special_tokens=True).strip()
    return parse_grade(answer), answer


def classify_patient(model, text_tok, vis_tok,
                     patient: dict, patients_all: list) -> dict:
    """
    Classify one patient: few-shot prompt -> classify 3 slices -> majority vote.
    """
    examples = select_few_shot_examples(patients_all, exclude_pid=patient["pid"])
    few_shot_block = format_few_shot_block(examples)
    prompt_text = build_prompt(patient, few_shot_block)

    slice_preds = []
    raw_outputs = []

    for spath in patient["slice_paths"]:
        pred, raw = classify_slice(model, text_tok, vis_tok, spath, prompt_text)
        slice_preds.append(pred)
        raw_outputs.append(raw)

    valid = [p for p in slice_preds if p != -1]
    final_pred = Counter(valid).most_common(1)[0][0] if valid else 1

    return {
        "pid": patient["pid"],
        "true_label": patient["label"],
        "final_pred": final_pred,
        "slice_preds": slice_preds,
        "raw_outputs": raw_outputs,
        "clinical": patient["clinical_text"],
    }


def run_ovis2_pipeline(crop_mode: str = "proportional",
                       num_patients: int = None,
                       output_dir: str = None):
    """Full Ovis2 pipeline: build dataset -> load model -> classify -> evaluate -> save."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if output_dir is None:
        from .config import DEFAULT_OUTPUT_DIR
        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"ovis2_{crop_mode}")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Dataset
    print("=" * 60)
    print(f"PHASE 1: Building DCE composites (crop={crop_mode})")
    print("=" * 60)
    patients = build_dataset(crop_mode=crop_mode, output_dir=output_dir,
                             num_patients=num_patients)
    print(f"\nReady: {len(patients)} patients x {SLICES_PER_PATIENT} slices "
          f"= {len(patients) * SLICES_PER_PATIENT} inference calls\n")

    # Phase 2: Inference
    print("=" * 60)
    print("PHASE 2: Few-shot classification")
    print("=" * 60)
    model, text_tok, vis_tok = load_ovis2_model()

    results = []
    for i, patient in enumerate(patients):
        result = classify_patient(model, text_tok, vis_tok, patient, patients)
        results.append(result)

        status = "OK" if result["final_pred"] == result["true_label"] else "MISS"
        print(
            f"[{i + 1:3d}/{len(patients)}] {patient['pid']:20s} | "
            f"true={LABEL_NAMES[result['true_label']]:8s} | "
            f"pred={LABEL_NAMES[result['final_pred']]:8s} | "
            f"slices={result['slice_preds']} {status}"
        )

    # Phase 3: Evaluate
    print("\n" + "=" * 60)
    print("PHASE 3: Evaluation")
    print("=" * 60)
    y_true = [r["true_label"] for r in results]
    y_pred = [r["final_pred"] for r in results]
    metrics = evaluate_predictions(y_true, y_pred, LABEL_NAMES)

    # Slice agreement stats
    agreements = []
    for r in results:
        valid = [p for p in r["slice_preds"] if p != -1]
        if valid:
            agreements.append(len(set(valid)) == 1)
    total_slices = sum(len(r["slice_preds"]) for r in results)
    failed = sum(1 for r in results for p in r["slice_preds"] if p == -1)
    print(f"\nSlice agreement:   {np.mean(agreements):.1%} patients had all slices agree")
    print(f"Parse failures:    {failed}/{total_slices} slices ({failed / total_slices:.1%})")

    # Save
    save_data = {
        "config": {
            "num_patients": len(patients),
            "slices_per_patient": SLICES_PER_PATIENT,
            "crop_mode": crop_mode,
            "model": "AIDC-AI/Ovis2-4B",
            "strategy": "DCE RGB composite + few-shot prompting + majority vote",
        },
        "metrics": metrics,
        "predictions": [
            {
                "pid": r["pid"],
                "true_label": int(r["true_label"]),
                "final_pred": int(r["final_pred"]),
                "slice_preds": r["slice_preds"],
                "raw_outputs": r["raw_outputs"],
                "clinical": r["clinical"],
            }
            for r in results
        ],
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")
