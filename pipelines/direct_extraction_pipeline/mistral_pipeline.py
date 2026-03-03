from pathlib import Path
from huggingface_hub import snapshot_download
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from huggingface_hub import snapshot_download

import json
import sys

# ── Model path (same as your snapshot_download target) ──────────────────────
MISTRAL_MODELS_PATH = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
MISTRAL_MODELS_PATH.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=MISTRAL_MODELS_PATH)


llm = None
tokenizer = None

def load_model():
    global llm, tokenizer
    if llm is None:
        if not MISTRAL_MODELS_PATH.exists():
            raise RuntimeError(
                f"Mistral model not found at {MISTRAL_MODELS_PATH}. "
                "Run snapshot_download first."
            )
        print("Loading Mistral tokenizer...")
        tokenizer = MistralTokenizer.from_file(
            str(MISTRAL_MODELS_PATH / "tokenizer.model.v3")
        )
        print("Loading Mistral model weights (this may take a minute)...")
        llm = Transformer.from_folder(str(MISTRAL_MODELS_PATH))
        print("Model loaded.")
    return llm, tokenizer


# ── Schema & helpers (unchanged) ─────────────────────────────────────────────
BPMN_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "actor": {"type": "string"}
                },
                "required": ["id", "name", "actor"]
            }
        },
        "sequence_flows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"}
                },
                "required": ["from", "to"]
            }
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["startEvent", "endEvent", "intermediateCatchEvent", "intermediateThrowEvent"]},
                    "participant": {"type": "string"},
                    "lane": {"type": "string"},
                    "eventDefinition": {"type": "string", "enum": ["start", "end", "message", "timer"]}
                },
                "required": ["id", "name", "type", "participant", "eventDefinition"]
            }
        },
        "gateways": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["exclusive", "parallel", "inclusive", "eventBased"]},
                    "from": {"type": "array", "items": {"type": "string"}},
                    "to": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["type", "from", "to"]
            }
        }
    },
    "required": ["tasks", "sequence_flows", "events", "gateways"]
}


def save_json_to_file(obj, path):
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def is_valid_bpmn(obj):
    if not isinstance(obj, dict):
        return False
    tasks = obj.get("tasks")
    seq = obj.get("sequence_flows")
    if not isinstance(tasks, list) or len(tasks) == 0:
        return False
    ids = set()
    for t in tasks:
        if not isinstance(t, dict):
            return False
        tid = t.get("id")
        name = t.get("name")
        actor = t.get("actor")
        if not tid or not name or not actor:
            return False
        if any(str(v).strip() in ("...", "", "None") for v in (name, actor)):
            return False
        ids.add(tid)
    if not isinstance(seq, list):
        return False
    for s in seq:
        if not isinstance(s, dict):
            return False
        if s.get("from") not in ids or s.get("to") not in ids:
            return False
    return True


# ── Core: call Mistral and parse JSON out of its reply ───────────────────────
def _mistral_chat(system_text: str, user_text: str, max_tokens: int = 4096) -> str:
    """
    Tokenize a two-message conversation, run generate(), return the decoded text.
    mistral_inference does NOT support constrained JSON decoding, so we ask the
    model to return JSON in the prompt and extract it afterwards.
    """
    model, tok = load_model()

    request = ChatCompletionRequest(
        messages=[
            SystemMessage(content=system_text),
            UserMessage(content=user_text),
        ]
    )
    encoded = tok.encode_chat_completion(request)
    input_ids = encoded.tokens

    # generate() returns a list of Results; we take the first
    [result] = generate(
        [input_ids],
        model,
        max_tokens=max_tokens,
        temperature=0.0,
        eos_id=tok.instruct_tokenizer.tokenizer.eos_id,
    )
    return tok.instruct_tokenizer.tokenizer.decode(result.tokens)


def _extract_json_block(text: str) -> str:
    """Pull the first {...} block out of the model reply."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return text[start:]  # malformed but let json.loads raise


def extract_bpmn(process_description, prompt_type="zero-shot", output_file=None, retries=2):
    system_msg = (
        "You are a business process modeling expert. "
        "Always respond with a single, valid JSON object and nothing else — "
        "no markdown fences, no explanation."
    )

    if prompt_type == "zero-shot":
        user_msg = f"""Extract all tasks, events, actors, and sequence flows from this process description.

Process description: {process_description}

Return a JSON object with exactly these keys:
- "tasks": array of objects with "id", "name", "actor"
- "events": array of objects with "id", "name", "type", "participant", "eventDefinition"
- "sequence_flows": array of objects with "from" and "to" (task ids)
- "gateways": array of objects with "type", "from" (array), "to" (array)

Each pool must have a start and end event (eventDefinition "start"/"end", name "general").
Use concrete names only — no placeholders."""

    print(f"Initial prompt:\n{user_msg}\n")

    try:
        raw = _mistral_chat(system_msg, user_msg)
    except RuntimeError as e:
        print(e)
        return None

    print("Model output:\n", raw)

    try:
        json_result = json.loads(_extract_json_block(raw))
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        json_result = None

    attempt = 0
    last_raw = raw
    while (not is_valid_bpmn(json_result)) and attempt < retries:
        print(f"\nRetrying ({attempt + 1}/{retries})...")

        retry_msg = f"""The previous output was invalid or contained placeholders.

Process description: {process_description}

Previous invalid output: {last_raw}

Provide a corrected JSON object with concrete task names and actors."""

        print(f"Retry prompt:\n{retry_msg}\n")

        try:
            last_raw = _mistral_chat(system_msg, retry_msg)
        except RuntimeError as e:
            print(e)
            return None

        print("Model output (retry):\n", last_raw)

        try:
            json_result = json.loads(_extract_json_block(last_raw))
        except json.JSONDecodeError as e:
            print(f"JSON parsing error on retry: {e}")
            json_result = None

        attempt += 1

    # ── Persist results ───────────────────────────────────────────────────────
    if json_result is None:
        print("\nError: Failed to parse JSON from model output")
    elif not is_valid_bpmn(json_result):
        print("\nWarning: Parsed JSON did not pass validation")
        if output_file:
            try:
                save_json_to_file(json_result, output_file.replace('.json', '_invalid.json'))
            except Exception as e:
                print(f"Failed to write invalid JSON: {e}")
    else:
        if output_file:
            try:
                save_json_to_file(json_result, output_file)
                print(f"Wrote parsed JSON to {output_file}")
            except Exception as e:
                print(f"Failed to write JSON: {e}")

    return json_result


# ── Entry point (unchanged logic) ────────────────────────────────────────────
import os

if __name__ == "__main__":
    case_name = "case_1"

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    CASES_DIR = os.path.join(PROJECT_ROOT, "cases")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(CASES_DIR, f"{case_name}.txt")
    out_file = os.path.join(OUTPUT_DIR, f"{case_name}_model_extracted_bpmn.json")

    print(f"--- Interactive Run Started ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Reading from: {input_path}")
    print(f"Saving to:    {out_file}")
    print(f"-------------------------------")

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError

        with open(input_path, "r", encoding="utf-8") as f:
            process_text = f.read()

        print(f"LLM is analyzing '{case_name}'...")
        bpmn_json = extract_bpmn(process_text, output_file=out_file, retries=2)

        if bpmn_json:
            print(f"Success! Output: outputs/{case_name}_model_extracted_bpmn.json")
        else:
            print("Model extraction failed. Check the logs above.")

    except FileNotFoundError:
        print(f"Error: '{case_name}.txt' not found in {CASES_DIR}")