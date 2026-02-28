try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except Exception:
    Llama = None
    _LLAMA_CPP_AVAILABLE = False

import json
import os
from pathlib import Path

llm = None

def load_model():
    global llm
    if llm is None:
        if not _LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python is not installed.")
        llm = Llama.from_pretrained(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename="llama-2-7b-chat.Q2_K.gguf",
            n_ctx=4096,
            n_gpu_layers=0,
            verbose=False
        )
    return llm


BPMN_SCHEMA = {
  "type": "object",
  "properties": {
    "participants": {
      "description": "Pools and optional lanes (swimlanes) that execute nodes.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {"type": "string", "enum": ["pool"]},
          "lanes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"}
              },
              "required": ["id", "name"]
            }
          }
        },
        "required": ["id", "name", "type"]
      }
    },
    "tasks": {
      "description": "BPMN task/activity nodes. Do NOT put events or gateways here.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {
            "type": "string",
            "enum": ["task", "userTask", "serviceTask", "scriptTask", "manualTask", "subProcess", "callActivity"]
          },
          "participant": {
            "description": "Must exactly match an id declared in the participants array.",
            "type": "string"
          },
          "lane": {
            "description": "Must exactly match a lane id within the declared participant, if lanes are defined.",
            "type": "string"
          }
        },
        "required": ["id", "name", "type", "participant"]
      }
    },
    "events": {
      "description": "BPMN event nodes only (start, end, intermediate). Do NOT put tasks or gateways here.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {
            "type": "string",
            "enum": ["startEvent", "endEvent", "intermediateCatchEvent", "intermediateThrowEvent"]
          },
          "participant": {
            "description": "Must exactly match an id declared in the participants array.",
            "type": "string"
          },
          "lane": {
            "description": "Must exactly match a lane id within the declared participant, if lanes are defined.",
            "type": "string"
          },
          "eventDefinition": {
            "description": "The trigger type of this event. Use 'none' for plain start/end events with no special trigger.",
            "type": "string",
            "enum": ["none", "message", "timer", "signal", "conditional", "error", "escalation", "link"]
          }
        },
        "required": ["id", "name", "type", "participant", "eventDefinition"]
      }
    },
    "gateways": {
      "description": "BPMN gateway nodes only (splits and joins). Do NOT put tasks or events here.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {
            "type": "string",
            "enum": ["exclusiveGateway", "parallelGateway", "inclusiveGateway", "eventBasedGateway"]
          },
          "participant": {
            "description": "Must exactly match an id declared in the participants array.",
            "type": "string"
          },
          "lane": {
            "description": "Must exactly match a lane id within the declared participant, if lanes are defined.",
            "type": "string"
          },
          "gatewayDirection": {
            "description": "diverging = split (one incoming, multiple outgoing). converging = join (multiple incoming, one outgoing).",
            "type": "string",
            "enum": ["diverging", "converging"]
          }
        },
        "required": ["id", "name", "type", "participant", "gatewayDirection"]
      }
    },
    "sequence_flows": {
      "description": "Control-flow edges. Both 'from' and 'to' must be node ids within the same pool.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "from": {"type": "string"},
          "to": {"type": "string"},
          "condition": {"type": "string"},
          "isDefault": {"type": "boolean"},
          "participant": {"type": "string"}
        },
        "required": ["from", "to"]
      }
    },
    "message_flows": {
      "description": "Communication edges between different pools.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "from": {"type": "string"},
          "to": {"type": "string"},
          "name": {"type": "string"}
        },
        "required": ["from", "to", "name"]
      }
    },
    "data": {
      "description": "Optional data objects referenced in conditions.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "dataType": {"type": "string", "enum": ["string", "number", "boolean", "object"]},
          "description": {"type": "string"}
        },
        "required": ["id", "name", "dataType"]
      }
    }
  },
  "required": ["participants", "tasks", "events", "gateways", "sequence_flows"]
}


# ── Few-shot case loader ───────────────────────────────────────────────────────

def load_few_shot_cases(few_shot_dir, exclude_case=None):
    """
    Load all (text, json) pairs from few_shot_dir.

    Each case must have two files with the same stem:
        <stem>.txt   — the process description
        <stem>.json  — the correct BPMN JSON

    Cases whose stem matches exclude_case are skipped (use this to avoid
    eval contamination when the test case is also a few-shot example).

    Returns a list of dicts: [{"name": stem, "text": ..., "json": ...}, ...]
    """
    few_shot_dir = Path(few_shot_dir)
    cases = []
    missing = []

    # Find all .txt files and look for a matching .json
    for txt_path in sorted(few_shot_dir.glob("*.txt")):
        stem      = txt_path.stem
        json_path = txt_path.with_suffix(".json")

        if exclude_case and stem == exclude_case:
            print(f"  [few-shot loader] Skipping '{stem}' (excluded to avoid eval contamination)")
            continue

        if not json_path.exists():
            missing.append(stem)
            continue

        try:
            text = txt_path.read_text(encoding="utf-8").strip()
            bpmn = json.loads(json_path.read_text(encoding="utf-8"))
            cases.append({"name": stem, "text": text, "json": bpmn})
        except Exception as e:
            print(f"  [few-shot loader] WARNING: Could not load '{stem}': {e}")

    if missing:
        print(f"  [few-shot loader] WARNING: No matching .json for: {missing}")

    print(f"  [few-shot loader] Loaded {len(cases)} few-shot case(s) from '{few_shot_dir}'")
    return cases


def build_few_shot_block(cases):
    """
    Turn a list of loaded cases into a formatted prompt block.
    Each example is separated clearly so the model can pattern-match on all of them.
    """
    if not cases:
        return ""

    lines = []
    for i, case in enumerate(cases, 1):
        lines.append(f"--- EXAMPLE {i}: {case['name']} ---")
        lines.append("PROCESS DESCRIPTION:")
        lines.append(case["text"])
        lines.append("")
        lines.append("CORRECT BPMN JSON OUTPUT:")
        lines.append(json.dumps(case["json"], indent=2))
        lines.append("")

    return "\n".join(lines)


# ── Validation ────────────────────────────────────────────────────────────────

def save_json_to_file(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def is_valid_bpmn(obj):
    if not isinstance(obj, dict):
        return False, "Output is not a JSON object"

    if "nodes" in obj:
        return False, (
            "Output contains a 'nodes' array, which is not valid. "
            "You must use three separate arrays: 'tasks', 'events', and 'gateways'."
        )

    for key in ("participants", "tasks", "events", "gateways", "sequence_flows"):
        if key not in obj:
            return False, (
                f"Missing required top-level field: '{key}'. "
                f"Required fields are: participants, tasks, events, gateways, sequence_flows."
            )

    participants = obj.get("participants")
    if not isinstance(participants, list) or len(participants) == 0:
        return False, "participants must be a non-empty array"

    participant_ids = set()
    for p in participants:
        if not isinstance(p, dict):
            return False, "Each participant must be an object"
        if not p.get("id") or not p.get("name") or p.get("type") != "pool":
            return False, f"Participant missing 'id', 'name', or type != 'pool': {p}"
        participant_ids.add(p["id"])
        for lane in p.get("lanes", []):
            if not isinstance(lane, dict) or not lane.get("id") or not lane.get("name"):
                return False, f"Invalid lane in participant '{p['id']}'"

    VALID_TASK_TYPES    = {"task", "userTask", "serviceTask", "scriptTask", "manualTask", "subProcess", "callActivity"}
    VALID_EVENT_TYPES   = {"startEvent", "endEvent", "intermediateCatchEvent", "intermediateThrowEvent"}
    VALID_GATEWAY_TYPES = {"exclusiveGateway", "parallelGateway", "inclusiveGateway", "eventBasedGateway"}
    VALID_EVENT_DEFS    = {"none", "message", "timer", "signal", "conditional", "error", "escalation", "link"}

    node_ids = set()

    def validate_nodes(nodes, valid_types, label):
        for node in nodes:
            if not isinstance(node, dict):
                return f"Each {label} must be an object"
            nid         = node.get("id")
            ntype       = node.get("type")
            participant = node.get("participant")
            if not nid or not ntype or not participant:
                return f"{label} is missing 'id', 'type', or 'participant': {node}"
            if ntype not in valid_types:
                return f"Invalid {label} type '{ntype}'. Must be one of: {sorted(valid_types)}"
            if participant not in participant_ids:
                return (
                    f"{label} '{nid}' references unknown participant '{participant}'. "
                    f"Declared participant ids are: {sorted(participant_ids)}"
                )
            lane_id = node.get("lane")
            if lane_id:
                parent   = next((p for p in participants if p["id"] == participant), None)
                lane_ids = {l["id"] for l in parent.get("lanes", [])} if parent else set()
                if lane_id not in lane_ids:
                    return f"{label} '{nid}' references unknown lane '{lane_id}'"
            if str(node.get("name", "")).strip() in ("...", "None", ""):
                return f"{label} '{nid}' has a placeholder or empty name"
            node_ids.add(nid)
        return None

    tasks    = obj.get("tasks", [])
    events   = obj.get("events", [])
    gateways = obj.get("gateways", [])

    if not isinstance(tasks, list) or len(tasks) == 0:
        return False, "tasks must be a non-empty array"
    if not isinstance(events, list):
        return False, "events must be an array"
    if not isinstance(gateways, list):
        return False, "gateways must be an array"

    for nodes, valid_types, label in [
        (tasks,    VALID_TASK_TYPES,    "task"),
        (events,   VALID_EVENT_TYPES,   "event"),
        (gateways, VALID_GATEWAY_TYPES, "gateway"),
    ]:
        err = validate_nodes(nodes, valid_types, label)
        if err:
            return False, err

    all_ids = [n["id"] for n in tasks + events + gateways]
    if len(all_ids) != len(set(all_ids)):
        return False, "Duplicate node ids found across tasks, events, and gateways"

    for event in events:
        ed = event.get("eventDefinition")
        if ed is not None and ed not in VALID_EVENT_DEFS:
            return False, f"Invalid eventDefinition '{ed}'"

    seq = obj.get("sequence_flows")
    if not isinstance(seq, list):
        return False, "sequence_flows must be an array"
    for s in seq:
        if not isinstance(s, dict):
            return False, "Each sequence_flow must be an object"
        if s.get("from") not in node_ids:
            return False, f"sequence_flow 'from' id '{s.get('from')}' does not match any known node id"
        if s.get("to") not in node_ids:
            return False, f"sequence_flow 'to' id '{s.get('to')}' does not match any known node id"

    for mf in obj.get("message_flows", []):
        if not isinstance(mf, dict):
            return False, "Each message_flow must be an object"
        if not mf.get("from") or not mf.get("to"):
            return False, "Each message_flow must have 'from' and 'to'"
        valid_refs = node_ids | participant_ids
        if mf["from"] not in valid_refs or mf["to"] not in valid_refs:
            return False, f"message_flow references unknown id: {mf}"

    return True, None


# ── Model ─────────────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a business process modeling expert. "
    "Extract structured BPMN models from process descriptions. "
    "Always output valid JSON matching the provided schema. "
    "Never use a 'nodes' array — always use separate 'tasks', 'events', and 'gateways' arrays. "
    "Every node must belong to a declared participant. "
    "Sequence flows stay within pools; cross-pool communication uses message_flows."
)


def _call_model(model, prompt):
    result = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt}
        ],
        response_format={"type": "json_object", "schema": BPMN_SCHEMA},
        temperature=0.0,
        max_tokens=2048
    )
    if isinstance(result, dict) and "choices" in result:
        return result["choices"][0]["message"]["content"]
    return str(result)


# ── Main extraction function ───────────────────────────────────────────────────

def extract_bpmn_few_shot(process_description, case_name, few_shot_dir, output_file=None, retries=2):
    """
    Extract a BPMN JSON from process_description using few-shot examples
    loaded from few_shot_dir. The case matching case_name is excluded from
    the few-shot examples to avoid eval contamination.
    """
    # Load all few-shot cases except the one being evaluated
    cases          = load_few_shot_cases(few_shot_dir, exclude_case=case_name)
    few_shot_block = build_few_shot_block(cases)

    if not few_shot_block:
        print("WARNING: No few-shot examples were loaded. Running zero-shot.")

    prompt = f"""Extract a structured BPMN model from the process description below.

Here are {len(cases)} example(s) showing correct process descriptions and their BPMN JSON outputs:

{few_shot_block}
--- NOW EXTRACT ---
Process description: {process_description}

Apply the same structure to the process description above. Rules:
- Use concrete names from the description. Do not use placeholders like "...", "None", or empty strings.
- Every task, event, and gateway must reference a valid participant id declared in participants.
- sequence_flows must stay within a single pool. Cross-pool communication goes in message_flows.
- Every pool must have at least one startEvent and one endEvent.
- Output exactly these top-level keys: participants, tasks, events, gateways, sequence_flows.
- Do NOT use a 'nodes' array. Tasks, events, and gateways are always separate arrays.
- Every gateway must have gatewayDirection of either "diverging" or "converging".
- Every event must have an eventDefinition."""

    try:
        model = load_model()
    except RuntimeError as e:
        print(e)
        return None

    print(f"Running few-shot extraction for '{case_name}' with {len(cases)} example(s)...")
    result_text = _call_model(model, prompt)
    print("Model output:\n", result_text)

    try:
        json_result = json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        json_result = None

    valid, reason = is_valid_bpmn(json_result)

    attempt = 0
    while not valid and attempt < retries:
        attempt += 1
        print(f"\nValidation failed: {reason}. Retrying ({attempt}/{retries})...")

        retry_prompt = f"""The previous BPMN output was invalid for the following reason:

{reason}

Process description: {process_description}

Previous invalid output: {result_text}

Re-extract the full BPMN model and fix the issue described above. Remember:
- Output exactly these top-level keys: participants, tasks, events, gateways, sequence_flows.
- Do NOT use a 'nodes' array. Tasks, events, and gateways are always separate arrays.
- Every node must have a participant id that exactly matches an id declared in participants.
- Every gateway must have gatewayDirection of either "diverging" or "converging".
- Every event must have an eventDefinition."""

        print(f"Retry prompt:\n{retry_prompt}\n")
        result_text = _call_model(model, retry_prompt)
        print(f"Model output (retry {attempt}):\n", result_text)

        try:
            json_result = json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error on retry: {e}")
            json_result = None

        valid, reason = is_valid_bpmn(json_result)

    if json_result is None:
        print("\nError: Failed to parse JSON from model output")
    elif not valid:
        print(f"\nWarning: Output did not pass validation after all retries. Reason: {reason}")
        if output_file:
            invalid_path = output_file.replace(".json", "_invalid.json")
            try:
                save_json_to_file(json_result, invalid_path)
                print(f"Wrote invalid output to {invalid_path}")
            except Exception as e:
                print(f"Failed to write invalid JSON: {e}")
    else:
        if output_file:
            try:
                save_json_to_file(json_result, output_file)
                print(f"Wrote output to {output_file}")
            except Exception as e:
                print(f"Failed to write JSON: {e}")

    return json_result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    case_name = "case_1"

    SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    CASES_DIR     = os.path.join(PROJECT_ROOT, "cases")
    FEW_SHOT_DIR  = os.path.join(PROJECT_ROOT, "few_shot_cases")  
    OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(CASES_DIR, f"{case_name}.txt")
    out_file   = os.path.join(OUTPUT_DIR, f"{case_name}_few_shot_bpmn.json")

    print("--- Few-Shot Pipeline Started ---")
    print(f"Case:          {case_name}")
    print(f"Cases dir:     {CASES_DIR}")
    print(f"Few-shot dir:  {FEW_SHOT_DIR}")
    print(f"Output:        {out_file}")
    print("---------------------------------")

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Case file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            process_text = f.read()

        bpmn_json = extract_bpmn_few_shot(
            process_description=process_text,
            case_name=case_name,
            few_shot_dir=FEW_SHOT_DIR,
            output_file=out_file,
            retries=2
        )

        if bpmn_json:
            print(f"Success! Output saved to: outputs/{case_name}_few_shot_bpmn.json")
        else:
            print("Model extraction failed. Check the logs above.")

    except FileNotFoundError as e:
        print(f"Error: {e}")