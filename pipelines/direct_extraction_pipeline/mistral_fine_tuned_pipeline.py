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

        SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
        model_path   = os.path.join(PROJECT_ROOT, "_fine_tuned_llm", "bpmn-mistral-finetuned.gguf")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at: {model_path}\n"
                f"Make sure bpmn-mistral-finetuned.gguf is inside the 'fine-tuned-llm/' folder "
                f"at the project root."
            )

        llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False
        )
    return llm


BPMN_SCHEMA = {
    "type": "object",
    "properties": {
        "pools": {
            "type": "array",
            "description": "Top-level containers representing an organisation or external entity (e.g. 'Rotterdam Sweater Shop', 'Customer'). A pool groups lanes together. If only one lane exists with no subdivision, still create a pool with one lane.",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string"},
                    "name": {"type": "string"}
                },
                "required": ["id", "name"],
                "additionalProperties": False
            }
        },
        "lanes": {
            "type": "array",
            "description": "The actors within a pool: people, roles, or departments (e.g. 'Anouk', 'Jan', 'Accountancy'). Every task/event/gateway is performed by a lane. Each lane belongs to exactly one pool.",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string"},
                    "name": {"type": "string"},
                    "pool": {"type": "string", "description": "The id of the parent pool this lane belongs to."}
                },
                "required": ["id", "name", "pool"],
                "additionalProperties": False
            }
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string"},
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ["task", "userTask", "serviceTask", "scriptTask", "manualTask", "subProcess", "callActivity"]},
                    "lane": {"type": "string", "description": "The id of the lane responsible for this task."}
                },
                "required": ["id", "name", "type", "lane"],
                "additionalProperties": False
            }
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":              {"type": "string"},
                    "name":            {"type": "string"},
                    "type":            {"type": "string", "enum": ["startEvent", "endEvent", "intermediateCatchEvent", "intermediateThrowEvent"]},
                    "lane":            {"type": "string", "description": "The id of the lane this event belongs to."},
                    "eventDefinition": {"type": "string", "enum": ["none", "message", "timer", "signal", "error", "escalation", "conditional", "link"]}
                },
                "required": ["id", "name", "type", "lane", "eventDefinition"],
                "additionalProperties": False
            }
        },
        "gateways": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":               {"type": "string"},
                    "name":             {"type": "string"},
                    "type":             {"type": "string", "enum": ["exclusiveGateway", "parallelGateway", "inclusiveGateway", "eventBasedGateway"]},
                    "lane":             {"type": "string", "description": "The id of the lane this gateway belongs to."},
                    "gatewayDirection": {"type": "string", "enum": ["diverging", "converging"]}
                },
                "required": ["id", "name", "type", "lane", "gatewayDirection"],
                "additionalProperties": False
            }
        },
      "sequence_flows": {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":        {"type": "string"},
            "from":      {"type": "string"},
            "to":        {"type": "string"},
            "condition": {
                "type": "string",
                "description": "Only on flows leaving an exclusiveGateway or inclusiveGateway. Describes the condition under which this path is taken (e.g. 'order below 25.000 EUR' or 'approved')."
            }
        },
        "required": ["id", "from", "to"]
    }
},
        "message_flows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":   {"type": "string"},
                    "from": {"type": "string"},
                    "to":   {"type": "string"},
                    "name": {"type": "string"}
                },
                "required": ["id", "from", "to", "name"],
                "additionalProperties": False
            }
        },
    },
    "required": ["pools", "lanes", "tasks", "events", "gateways", "sequence_flows", "message_flows"]
}

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

    node_ids     = set()
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

INSTRUCTION = (
    "Extract a structured BPMN model from the following process description. "
    "Output valid JSON with exactly these top-level keys: participants, tasks, "
    "events, gateways, sequence_flows. Every node must reference a declared "
    "participant. Every pool must have a startEvent and endEvent. "
    "Sequence flows stay within pools; cross-pool communication uses message_flows."
)

SYSTEM_MSG = (
    "You are a business process modeling expert. "
    "Extract structured BPMN models from process descriptions. "
    "Always output valid JSON matching the provided schema. "
    "Never use a 'nodes' array — always use separate 'tasks', 'events', and 'gateways' arrays. "
    "Every node must belong to a declared participant. "
    "Every pool must have a startEvent and an endEvent. "
    "Sequence flows stay within pools; cross-pool communication uses message_flows."
)


def _build_prompt(process_description):
    """
    Alpaca-style prompt — must match the format used during fine-tuning.
    """
    return (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{process_description}\n\n"
        f"### Response:\n"
    )


def _call_model(model, prompt):
    result = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt}
        ],
        response_format={"type": "json_object", "schema": BPMN_SCHEMA},
        temperature=0.0,
        max_tokens=8192
    )
    if isinstance(result, dict) and "choices" in result:
        return result["choices"][0]["message"]["content"]
    return str(result)


# ── Main extraction function ───────────────────────────────────────────────────

def extract_bpmn_fine_tuned(process_description, case_name, output_file=None):
    """ You are a BPMN 2.0 expert. Extract a structured BPMN model from the process description below.

## STEP 1 — IDENTIFY POOLS (DO THIS FIRST)

A pool = one organisation or external entity.
Ask yourself: how many distinct organisations or external parties are involved?

RULES:
- External parties (customers, suppliers, government bodies) = their own pool
- Internal departments or roles within the SAME organisation = lanes inside ONE pool
- NEVER split one organisation into multiple pools just because it has multiple departments

EXAMPLE (CORRECT):
  Pools represent distinct organizations or external participants.
  Each pool contains lanes that represent departments or roles within that organization.

  Pool: "Company" → lanes: Customer Service, Operations, Finance
  Pool: "Customer" → lanes: Customer

EXAMPLE (WRONG):
  Pool: "Customer Service"
  Pool: "Operations"
  Pool: "Finance"   ← these are departments/roles, not separate organizations (pools)


## STEP 2 — ASSIGN LANES

Each role, department, or actor within a pool = one lane.
Every task, event, and gateway must belong to exactly one lane.


## STEP 3 — MODEL EACH POOL'S FLOW INDEPENDENTLY

Each pool has its own internal sequence flow.
Pools NEVER share gateways or sequence flows.
A parallel gateway that splits work inside one pool CANNOT join work from another pool.

If two departments in different pools both do work, and then results are combined:
- Each pool finishes its own branch with a message throw event
- The receiving pool uses a parallel SPLIT + parallel JOIN to wait for all messages


## STEP 4 — CROSS-POOL COMMUNICATION

ONLY message_flows may cross pool boundaries.
- Sending node: intermediateThrowEvent (message)
- Receiving node: intermediateCatchEvent (message) or message startEvent
- message_flow targets must be intermediateCatch EVENTS — NEVER tasks

FORBIDDEN:
  sequence_flow from pool A → pool B
  message_flow "to": "some_task_id"


## STEP 5 — GATEWAYS

Decisions (if/else) → exclusiveGateway
Parallel work (do both simultaneously) → parallelGateway

PARALLEL GATEWAY RULES (STRICT):
- A parallel split (diverging) fans out to 2+ branches
- A parallel join (converging) waits for ALL branches to finish
- You MUST always have BOTH a split AND a join, 
- The split AND join must be in the SAME pool
- You cannot join work that happens in different pools using a gateway

XOR GATEWAY RULES:
- An exclusive split (diverging) fans out to 2+ branches
- An exclusive join (converging) waits for ONE branch to finish or has a clear end event for each branch 
- The split AND join must be in the SAME pool or if branches end in separate endEvents, they must be in the same pool
- You cannot join work that happens in different pools using a gateway

## STEP 6 — COMPLETE FLOWS

Every task must have (STRICT: CHECK THIS MUTLIPLE TIMES):
- exactly 1 incoming sequence flow
- exactly 1 outgoing sequence flow
- NO MESSAGE FLOWS directly to or from tasks — use intermediate events aftet the task for messaging

Every pool must have:
- exactly 1 startEvent
- at least 1 endEvent

Every decision (exclusiveGateway diverging) must eventually be merged by a corresponding exclusiveGateway (converging), unless branches end in separate endEvents.

## STEP 7 — HOW TO CORRECTLY USE INTERMEDIATE MESSAGE EVENTS IN A FLOW

INTERMEDIATE Message events are NOT floating annotations. They MUST be connected into the flow with the correct connection types.

CORRECT SEQUENCE OR MESSAGE FLOW PATTERNS FOR INTERMEDIATE EVENTS:

  intermediateThrowEvent (send):
    incoming: 1 sequence flow  (from previous TASK OR ExclusiveGateway in same lane)
    outgoing: 1 message flow   (to a intermediateCatchEvent in a DIFFERENT pool)
    NO outgoing sequence flow

  intermediateCatchEvent (receive):
    incoming: 1 message flow   (from a intermediateThrowEvent in a DIFFERENT pool)
    outgoing: 1 sequence flow  (to next node in same lane)
    NO incoming sequence flow

CORRECT pattern for cross-pool communication:

  Pool A (sender lane):
    ... → task_A → intermediateThrowEvent ──(message flow)──→ intermediateCatchEvent → endEvent → (Pool B, receiver lane)


## FINAL SELF-CHECK (MANDATORY — fix violations before output)

[ ] How many organisations/external parties are there? Does each have its own pool?
[ ] Are departments/roles within the same organisation modeled as lanes (not pools)?
[ ] Do message flows point to events only? (Never to tasks)
[ ] Does every parallel split have a corresponding parallel join IN THE SAME pool?
[ ] Does every diverging exclusiveGateway have a converging counterpart or endEvents for each branch?
[ ] Does every task have both an incoming and outgoing sequence flow (MOST IMPORTANT TO CHECK)?
[ ] Does every pool have exactly one startEvent and at least one endEvent?

---

Return ONLY valid JSON matching the schema. Do not explain anything.

## Process description
{process_description}"""
    prompt = _build_prompt(process_description)

    try:
        model = load_model()
    except (RuntimeError, FileNotFoundError) as e:
        print(e)
        return None

    print(f"Running fine-tuned extraction for '{case_name}'...")
    result_text = _call_model(model, prompt)
    print("Model output:\n", result_text)

    try:
        json_result = json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        json_result = None

    # ── Validate and report issues (no retries) ───────────────────────────────
    if json_result is not None:
        fully_valid, errors, warnings = is_valid_bpmn(json_result)
        if not fully_valid:
            issues_text = _format_issues(errors, warnings)
            print(f"\nValidation issues (no retry — recording as-is):\n{issues_text}")

    # ── Save output ───────────────────────────────────────────────────────────
    if json_result is None:
        print("\nError: Failed to parse JSON from model output.")
    else:
        if output_file:
            try:
                save_json_to_file(json_result, output_file)
                print(f"\nWrote output to {output_file}")
            except Exception as e:
                print(f"Failed to write JSON: {e}")

    return json_result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- CONFIGURATION ---
    case_name = "case_23"
    # ---------------------

    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    CASES_DIR    = os.path.join(PROJECT_ROOT, "cases")
    OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(CASES_DIR, f"{case_name}.txt")
    out_file   = os.path.join(OUTPUT_DIR, f"{case_name}_fine_tuned_mistral_bpmn.json")

    print("--- Fine-Tuned Pipeline Started ---")
    print(f"Case:          {case_name}")
    print(f"Cases dir:     {CASES_DIR}")
    print(f"Output:        {out_file}")
    print("-----------------------------------")

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Case file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            process_text = f.read()

        bpmn_json = extract_bpmn_fine_tuned(
            process_description=process_text,
            case_name=case_name,
            output_file=out_file,
        )

        if bpmn_json:
            print(f"Output saved to: outputs/{case_name}_fine_tuned_bpmn.json")
        else:
            print("Model extraction failed. Check the logs above.")

    except FileNotFoundError as e:
        print(f"Error: {e}")