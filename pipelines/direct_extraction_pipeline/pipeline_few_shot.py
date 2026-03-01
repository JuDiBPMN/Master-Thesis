try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except Exception:
    Llama = None
    _LLAMA_CPP_AVAILABLE = False

import json
import os
from pathlib import Path
from collections import defaultdict

llm = None

def load_model():
    global llm
    if llm is None:
        if not _LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python is not installed.")
        llm = Llama.from_pretrained(
            repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
            filename="llama-2-7b-chat.Q2_K.gguf",
            n_ctx=8192,
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
    few_shot_dir = Path(few_shot_dir)
    cases, missing = [], []

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
    if not cases:
        return ""
    lines = []
    for i, case in enumerate(cases, 1):
        lines.append(f"--- EXAMPLE {i}: {case['name']} ---")
        lines.append("PROCESS DESCRIPTION:")
        lines.append(case["text"])
        lines.append("")
        lines.append("CORRECT BPMN JSON OUTPUT:")
        # Compact JSON to save tokens
        lines.append(json.dumps(case["json"], separators=(',', ':')))
        lines.append("")
    return "\n".join(lines)


# ── Validation ────────────────────────────────────────────────────────────────

def save_json_to_file(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def is_valid_bpmn(obj):
    """
    Returns (is_fully_valid, errors, warnings) where:
    - is_fully_valid: True only when both errors AND warnings are empty
    - errors:   list of blocking problems (XML will be broken)
    - warnings: list of structural problems (XML imports but diagram is incomplete)
    """
    errors   = []
    warnings = []

    if not isinstance(obj, dict):
        return False, ["Output is not a JSON object"], []

    if "nodes" in obj:
        errors.append(
            "Output contains a 'nodes' array. "
            "Use three separate arrays: 'tasks', 'events', and 'gateways'."
        )

    for key in ("participants", "tasks", "events", "gateways", "sequence_flows"):
        if key not in obj:
            errors.append(
                f"Missing required top-level field: '{key}'. "
                f"Required: participants, tasks, events, gateways, sequence_flows."
            )

    if errors:
        return False, errors, warnings

    participants = obj.get("participants", [])
    if not isinstance(participants, list) or len(participants) == 0:
        return False, ["participants must be a non-empty array"], []

    participant_ids = set()
    for p in participants:
        if not isinstance(p, dict):
            errors.append("Each participant must be an object")
            continue
        if not p.get("id") or not p.get("name") or p.get("type") != "pool":
            errors.append(f"Participant missing 'id', 'name', or type != 'pool': {p}")
            continue
        participant_ids.add(p["id"])
        for lane in p.get("lanes", []):
            if not isinstance(lane, dict) or not lane.get("id") or not lane.get("name"):
                errors.append(f"Invalid lane in participant '{p['id']}'")

    VALID_TASK_TYPES    = {"task","userTask","serviceTask","scriptTask","manualTask","subProcess","callActivity"}
    VALID_EVENT_TYPES   = {"startEvent","endEvent","intermediateCatchEvent","intermediateThrowEvent"}
    VALID_GATEWAY_TYPES = {"exclusiveGateway","parallelGateway","inclusiveGateway","eventBasedGateway"}
    VALID_EVENT_DEFS    = {"none","message","timer","signal","conditional","error","escalation","link"}

    node_ids    = set()
    node_to_pool = {}

    def validate_nodes(nodes, valid_types, label):
        for node in nodes:
            if not isinstance(node, dict):
                errors.append(f"Each {label} must be an object")
                continue
            nid         = node.get("id")
            ntype       = node.get("type")
            participant = node.get("participant")
            if not nid or not ntype or not participant:
                errors.append(f"{label} is missing 'id', 'type', or 'participant': {node}")
                continue
            if ntype not in valid_types:
                errors.append(f"Invalid {label} type '{ntype}'. Must be one of: {sorted(valid_types)}")
            if participant not in participant_ids:
                errors.append(
                    f"NODE '{nid}' has participant='{participant}' which is NOT declared in participants. "
                    f"Declared participant ids are: {sorted(participant_ids)}. "
                    f"Either add '{participant}' to the participants array, or change the participant field "
                    f"to one of the declared ids."
                )
            lane_id = node.get("lane")
            if lane_id:
                parent   = next((p for p in participants if p["id"] == participant), None)
                lane_ids = {l["id"] for l in parent.get("lanes", [])} if parent else set()
                if lane_id not in lane_ids:
                    errors.append(f"{label} '{nid}' references unknown lane '{lane_id}'")
            if str(node.get("name", "")).strip() in ("...", "None", ""):
                errors.append(f"{label} '{nid}' has a placeholder or empty name")
            node_ids.add(nid)
            node_to_pool[nid] = participant

    tasks    = obj.get("tasks", [])
    events   = obj.get("events", [])
    gateways = obj.get("gateways", [])

    if not isinstance(tasks, list) or len(tasks) == 0:
        errors.append("tasks must be a non-empty array")
    if not isinstance(events, list):
        errors.append("events must be an array")
    if not isinstance(gateways, list):
        errors.append("gateways must be an array")

    if errors:
        return False, errors, warnings

    validate_nodes(tasks,    VALID_TASK_TYPES,    "task")
    validate_nodes(events,   VALID_EVENT_TYPES,   "event")
    validate_nodes(gateways, VALID_GATEWAY_TYPES, "gateway")

    all_ids = [n["id"] for n in tasks + events + gateways]
    if len(all_ids) != len(set(all_ids)):
        errors.append("Duplicate node ids found across tasks, events, and gateways")

    for event in events:
        ed = event.get("eventDefinition")
        if ed is not None and ed not in VALID_EVENT_DEFS:
            errors.append(f"Invalid eventDefinition '{ed}'")

    seq = obj.get("sequence_flows", [])
    if not isinstance(seq, list):
        errors.append("sequence_flows must be an array")
    else:
        for s in seq:
            if not isinstance(s, dict):
                errors.append("Each sequence_flow must be an object")
                continue
            if s.get("from") not in node_ids:
                errors.append(
                    f"sequence_flow 'from' id '{s.get('from')}' does not exist. "
                    f"Every 'from' and 'to' in sequence_flows must exactly match a node id "
                    f"declared in tasks, events, or gateways."
                )
            if s.get("to") not in node_ids:
                errors.append(
                    f"sequence_flow 'to' id '{s.get('to')}' does not exist. "
                    f"Every 'from' and 'to' in sequence_flows must exactly match a node id "
                    f"declared in tasks, events, or gateways."
                )

    for mf in obj.get("message_flows", []):
        if not isinstance(mf, dict):
            errors.append("Each message_flow must be an object")
            continue
        if not mf.get("from") or not mf.get("to"):
            errors.append("Each message_flow must have 'from' and 'to'")
            continue
        valid_refs = node_ids | participant_ids
        if mf["from"] not in valid_refs or mf["to"] not in valid_refs:
            errors.append(f"message_flow references unknown id: {mf}")

    # ── Warnings: structural completeness ─────────────────────────────────────
    pool_has_start = defaultdict(bool)
    pool_has_end   = defaultdict(bool)
    for event in events:
        pid = event.get("participant")
        if event.get("type") == "startEvent":
            pool_has_start[pid] = True
        if event.get("type") == "endEvent":
            pool_has_end[pid] = True

    for pid in participant_ids:
        if not pool_has_start[pid]:
            warnings.append(
                f"POOL '{pid}' has no startEvent. "
                f"Every pool must have at least one startEvent. "
                f"Add a startEvent with participant='{pid}' and eventDefinition='none' "
                f"(or 'message' if it is triggered by another pool)."
            )
        if not pool_has_end[pid]:
            warnings.append(
                f"POOL '{pid}' has no endEvent. "
                f"Every pool must have at least one endEvent. "
                f"Add an endEvent with participant='{pid}' and eventDefinition='none', "
                f"and connect it with a sequence_flow from the last node in that pool."
            )

    # Cross-pool sequence flows (structural warning, auto-promoted but still wrong)
    for s in seq:
        src_pool = node_to_pool.get(s.get("from"))
        tgt_pool = node_to_pool.get(s.get("to"))
        if src_pool and tgt_pool and src_pool != tgt_pool:
            warnings.append(
                f"sequence_flow '{s.get('from')}' -> '{s.get('to')}' crosses pools "
                f"('{src_pool}' -> '{tgt_pool}'). "
                f"Move this to message_flows instead."
            )

    is_fully_valid = len(errors) == 0 and len(warnings) == 0
    return is_fully_valid, errors, warnings


def _format_issues(errors, warnings):
    """Format errors and warnings into a clear, actionable string for the retry prompt."""
    lines = []
    if errors:
        lines.append("ERRORS (these must be fixed — the XML will be broken):")
        for e in errors:
            lines.append(f"  - {e}")
    if warnings:
        lines.append("WARNINGS (these must also be fixed — the diagram will be incomplete):")
        for w in warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


# ── Model ─────────────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a business process modeling expert. "
    "Extract structured BPMN models from process descriptions. "
    "Always output valid JSON matching the provided schema. "
    "Never use a 'nodes' array — always use separate 'tasks', 'events', and 'gateways' arrays. "
    "Every node must belong to a declared participant. "
    "Every pool must have a startEvent and an endEvent. "
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

def extract_bpmn_few_shot(process_description, case_name, few_shot_dir, output_file=None, retries=3):
    cases          = load_few_shot_cases(few_shot_dir, exclude_case=case_name)
    few_shot_block = build_few_shot_block(cases)

    if not few_shot_block:
        print("WARNING: No few-shot examples were loaded. Running zero-shot.")

    prompt = f"""Extract a structured BPMN model from the process description below.

Here are {len(cases)} example(s) showing correct process descriptions and their BPMN JSON outputs:

{few_shot_block}
--- NOW EXTRACT ---
Process description: {process_description}

Apply the same structure. Follow these rules exactly:
- Identify ALL participants (pools) mentioned in the description and declare every one of them in the participants array. If a node belongs to a pool, that pool MUST be in the participants array.
- Every participant pool must have at least one startEvent and at least one endEvent. Do not leave any pool without these.
- Every task, event, and gateway must have a 'participant' field that exactly matches an id declared in the participants array.
- sequence_flows must only connect nodes within the same pool. Cross-pool communication goes in message_flows.
- Every 'from' and 'to' in sequence_flows must exactly match a node id declared in tasks, events, or gateways.
- Output exactly these top-level keys: participants, tasks, events, gateways, sequence_flows.
- Do NOT use a 'nodes' array. Tasks, events, and gateways are always separate arrays.
- Every gateway must have gatewayDirection set to either "diverging" or "converging".
- Every event must have an eventDefinition field."""

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

    fully_valid, errors, warnings = is_valid_bpmn(json_result) if json_result else (False, ["Failed to parse JSON"], [])

    attempt = 0
    while not fully_valid and attempt < retries:
        attempt += 1
        issues_text = _format_issues(errors, warnings)
        print(f"\nIssues found (attempt {attempt}/{retries}):\n{issues_text}")

        retry_prompt = f"""The BPMN JSON you produced has the following issues that MUST ALL be fixed:

{issues_text}

Original process description: {process_description}

Your previous (broken) output:
{result_text}

Produce a corrected version of the complete BPMN JSON. Fix every issue listed above:
- If a node has an undeclared participant, add that participant to the participants array OR change the node's participant to an already-declared id.
- If a pool is missing a startEvent or endEvent, add one and connect it with a sequence_flow.
- If a sequence_flow 'from' or 'to' id does not exist, either fix the id to match a real node or remove the flow.
- If a sequence_flow crosses pools, move it to message_flows.
Output the complete corrected JSON. Do not omit any part of the model."""

        print(f"Retry prompt (attempt {attempt}):\n{retry_prompt}\n")
        result_text = _call_model(model, retry_prompt)
        print(f"Model output (retry {attempt}):\n", result_text)

        try:
            json_result = json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error on retry: {e}")
            json_result = None

        fully_valid, errors, warnings = is_valid_bpmn(json_result) if json_result else (False, ["Failed to parse JSON"], [])

    # ── Save output ───────────────────────────────────────────────────────────
    if json_result is None:
        print("\nError: Failed to parse JSON from model output after all attempts.")
    elif not fully_valid:
        remaining = _format_issues(errors, warnings)
        print(f"\nWarning: Output still has issues after {retries} retries:\n{remaining}")
        if output_file:
            invalid_path = output_file.replace(".json", "_invalid.json")
            try:
                save_json_to_file(json_result, invalid_path)
                print(f"Wrote invalid output to {invalid_path}")
            except Exception as e:
                print(f"Failed to write invalid JSON: {e}")
    else:
        print("\nValidation passed.")
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
            retries=3
        )

        if bpmn_json:
            print(f"Output saved to: outputs/{case_name}_few_shot_bpmn.json")
        else:
            print("Model extraction failed. Check the logs above.")

    except FileNotFoundError as e:
        print(f"Error: {e}")