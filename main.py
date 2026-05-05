import os
from pathlib import Path
from typing import Optional

from pipelines.direct_extraction_pipeline import (
    llama_few_shot,
    llama_fine_tuned,
    llama_zero_shot,
    mistral_few_shot,
    mistral_fine_tuned,
    mistral_zero_shot,
    phi_few_shot,
    phi_fine_tuned,
    phi_zero_shot,
)
from xml_generation.pipeline import generate_case_bpmn_xml

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# Available model options:
# - "mistral-finetuned"         -> JulesNuytten/MistralBPMNTuned
# - "phi4-seed42"               -> JulesNuytten/bpmn-phi4-gguf-collection (seed42)
# - "phi4-seed2026"             -> JulesNuytten/bpmn-phi4-gguf-collection (seed2026)
# - "phi4-seed123"              -> JulesNuytten/bpmn-phi4-finetuned
# - "mistral-base"              -> TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# - "llama2-base"               -> TheBloke/Llama-2-7B-Chat-GGUF
# - "phi4-base"                 -> unsloth/phi-4-GGUF
# - "llama-finetuned-local"     -> local _fine_tuned_llm/bpmn-finetuned.gguf
HF_MODELS = {
    "mistral-finetuned": ("JulesNuytten/MistralBPMNTuned", "bpmn-mistral-finetuned.gguf"),
    "phi4-seed42": ("JulesNuytten/bpmn-phi4-gguf-collection", "bpmn-phi4-seed42.q8_0.gguf"),
    "phi4-seed2026": ("JulesNuytten/bpmn-phi4-gguf-collection", "bpmn-phi4-seed2026.q8_0.gguf"),
    "phi4-seed123": ("JulesNuytten/bpmn-phi4-finetuned", "bpmn-phi4-finetuned_seed123.gguf"),
    "mistral-base": ("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "mistral-7b-instruct-v0.2.Q5_K_M.gguf"),
    "llama2-base": ("TheBloke/Llama-2-7B-Chat-GGUF", "llama-2-7b-chat.Q6_K.gguf"),
    "phi4-base": ("unsloth/phi-4-GGUF", "phi-4-Q4_K_M.gguf"),
}

PIPELINES = {
    "mistral_zero_shot": {"kind": "zero_shot", "runner": mistral_zero_shot.extract_bpmn, "default_model": "mistral-base"},
    "llama_zero_shot": {"kind": "zero_shot", "runner": llama_zero_shot.extract_bpmn, "default_model": "llama2-base"},
    "phi_zero_shot": {"kind": "zero_shot", "runner": phi_zero_shot.extract_bpmn, "default_model": "phi4-base"},
    "mistral_few_shot": {"kind": "few_shot", "runner": mistral_few_shot.extract_bpmn_few_shot, "default_model": "mistral-base"},
    "llama_few_shot": {"kind": "few_shot", "runner": llama_few_shot.extract_bpmn_few_shot, "default_model": "llama2-base"},
    "phi_few_shot": {"kind": "few_shot", "runner": phi_few_shot.extract_bpmn_few_shot, "default_model": "phi4-base"},
    "mistral_fine_tuned": {"kind": "fine_tuned", "runner": mistral_fine_tuned.extract_bpmn_fine_tuned, "default_model": "mistral-finetuned"},
    "llama_fine_tuned": {"kind": "fine_tuned", "runner": llama_fine_tuned.extract_bpmn_fine_tuned, "default_model": "llama-finetuned-local"},
    "phi_fine_tuned": {"kind": "fine_tuned", "runner": phi_fine_tuned.extract_bpmn, "default_model": "phi4-seed2026"},
}

PIPELINE_OPTIONS = [
    (1, "mistral_zero_shot"),
    (2, "llama_zero_shot"),
    (3, "phi_zero_shot"),
    (4, "mistral_few_shot"),
    (5, "llama_few_shot"),
    (6, "phi_few_shot"),
    (7, "mistral_fine_tuned"),
    (8, "llama_fine_tuned"),
    (9, "phi_fine_tuned"),
]
PIPELINE_BY_ID = {idx: name for idx, name in PIPELINE_OPTIONS}


def load_model(model_key: str, hf_token: Optional[str] = None, cache_dir: Optional[str] = None):
    if Llama is None:
        raise RuntimeError("llama-cpp-python is required. Run: pip install llama-cpp-python")

    if model_key == "llama-finetuned-local":
        project_root = Path(__file__).resolve().parent
        model_path = project_root / "_fine_tuned_llm" / "bpmn-finetuned.gguf"
        if not model_path.exists():
            raise FileNotFoundError(f"Fine-tuned local model not found: {model_path}")
        return Llama(model_path=str(model_path), n_ctx=2048, n_gpu_layers=-1, verbose=False)

    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required. Run: pip install huggingface_hub")
    if model_key not in HF_MODELS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose one of: {list(HF_MODELS.keys()) + ['llama-finetuned-local']}")

    repo_id, filename = HF_MODELS[model_key]
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token, cache_dir=cache_dir)
    return Llama(model_path=model_path, n_ctx=16384, n_gpu_layers=-1, verbose=False)


def main():
    # ====================== RUN CONFIGURATION ======================
    # Only edit these values:
    case_name = "case_23"
    pipeline = 1
    # Optional overrides:
    model_key = None
    hf_token = os.environ.get("HF_TOKEN")
    few_shot_dir = None
    few_shot_case_ids = None
    # ==============================================================

    project_root = Path(__file__).resolve().parent
    cases_dir = project_root / "cases"
    extraction_output_dir = project_root / "pipelines" / "direct_extraction_pipeline" / "outputs"
    extraction_output_dir.mkdir(parents=True, exist_ok=True)

    if pipeline not in PIPELINE_BY_ID:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Choose one of: {[idx for idx, _ in PIPELINE_OPTIONS]}")

    print("[main] Available pipelines:")
    for idx, name in PIPELINE_OPTIONS:
        print(f"  {idx}. {name} ({PIPELINES[name]['kind']})")

    selected_pipeline_name = PIPELINE_BY_ID[pipeline]
    case_path = cases_dir / f"{case_name}.txt"
    if not case_path.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    with case_path.open("r", encoding="utf-8") as f:
        process_text = f.read()

    selected = PIPELINES[selected_pipeline_name]
    selected_model = model_key or selected["default_model"]
    print(f"[main] Selected case: {case_name}")
    print(f"[main] Selected pipeline: {pipeline} -> {selected_pipeline_name} ({selected['kind']})")
    print(f"[main] Selected model: {selected_model}")

    model = load_model(selected_model, hf_token=hf_token)
    json_output = extraction_output_dir / f"{case_name}_{selected_pipeline_name}_bpmn.json"

    if selected["kind"] == "few_shot":
        resolved_few_shot_dir = few_shot_dir or str(project_root / "few_shot_cases")
        json_result = selected["runner"](
            process_description=process_text,
            case_name=case_name,
            few_shot_dir=resolved_few_shot_dir,
            output_file=str(json_output),
            retries=0,
            case_ids=few_shot_case_ids,
            model=model,
        )
    elif selected_pipeline_name == "phi_fine_tuned":
        json_result = selected["runner"](
            process_description=process_text,
            prompt_type="fine-tuned",
            output_file=str(json_output),
            retries=0,
            model=model,
        )
    elif selected["kind"] == "zero_shot":
        json_result = selected["runner"](
            process_description=process_text,
            prompt_type="zero-shot",
            output_file=str(json_output),
            retries=0,
            model=model,
        )
    else:
        json_result = selected["runner"](
            process_description=process_text,
            case_name=case_name,
            output_file=str(json_output),
            model=model,
        )

    if not json_result:
        print("[main] Extraction failed, skipping XML generation.")
        return

    xml_path = generate_case_bpmn_xml(
        case_name=case_name,
        pipeline_name="direct_extraction_pipeline",
        prompting_strategy=selected_pipeline_name + "_bpmn",
        input_json=str(json_output),
    )
    print(f"[main] JSON output: {json_output}")
    print(f"[main] XML output:  {xml_path}")


if __name__ == "__main__":
    main()
