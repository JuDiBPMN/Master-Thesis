try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except Exception:
    Llama = None
    _LLAMA_CPP_AVAILABLE = False

import json
import sys
import os

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


def save_json_to_file(obj, path):
    from pathlib import Path
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


from collections import defaultdict

def is_valid_bpmn(obj):
    """
    Returns (is_fully_valid, errors, warnings).
    """
    errors, warnings = [], []
    if not isinstance(obj, dict):
        return False, ["Output is not a JSON object"], []

    for key in ("pools", "lanes", "tasks", "events", "gateways", "sequence_flows", "message_flows"):
        if key not in obj:
            errors.append(f"Missing required field: '{key}'")
    if errors:
        return False, errors, []

    pools    = obj["pools"]
    lanes    = obj["lanes"]
    tasks    = obj["tasks"]
    events   = obj["events"]
    gateways = obj["gateways"]
    seq      = obj["sequence_flows"]
    mflows   = obj["message_flows"]

    pool_ids = set()
    for p in pools:
        if not p.get("id") or not p.get("name"):
            errors.append(f"Pool missing 'id' or 'name': {p}")
        else:
            pool_ids.add(p["id"])

    lane_ids = set()
    for l in lanes:
        if not l.get("id") or not l.get("name"):
            errors.append(f"Lane missing 'id' or 'name': {l}")
        elif l.get("pool") not in pool_ids:
            errors.append(f"Lane '{l['id']}' references unknown pool '{l.get('pool')}'")
        else:
            lane_ids.add(l["id"])

    VALID_TASK_TYPES    = {"task","userTask","serviceTask","scriptTask","manualTask","subProcess","callActivity"}
    VALID_EVENT_TYPES   = {"startEvent","endEvent","intermediateCatchEvent","intermediateThrowEvent"}
    VALID_EVENT_DEFS    = {"none","message","timer","signal","error","escalation","conditional","link"}
    VALID_GATEWAY_TYPES = {"exclusiveGateway","parallelGateway","inclusiveGateway","eventBasedGateway"}
    VALID_GW_DIRS       = {"diverging","converging"}

    node_ids     = set()
    node_to_lane = {}

    def validate_nodes(nodes, valid_types, label, extra_checks=None):
        for n in nodes:
            nid = n.get("id")
            if not nid or not n.get("name") or not n.get("type") or not n.get("lane"):
                errors.append(f"{label} missing 'id', 'name', 'type', or 'lane': {n}")
                continue
            if n["type"] not in valid_types:
                errors.append(f"{label} '{nid}': invalid type '{n['type']}'")
            if n["lane"] not in lane_ids:
                errors.append(f"{label} '{nid}': unknown lane '{n['lane']}'")
            if nid in node_ids:
                errors.append(f"Duplicate node id '{nid}'")
            node_ids.add(nid)
            node_to_lane[nid] = n["lane"]
            if extra_checks:
                extra_checks(n)

    def check_event(e):
        ed = e.get("eventDefinition")
        if not ed or ed not in VALID_EVENT_DEFS:
            errors.append(f"Event '{e.get('id')}': invalid or missing eventDefinition '{ed}'")

    def check_gateway(g):
        if g.get("gatewayDirection") not in VALID_GW_DIRS:
            errors.append(f"Gateway '{g.get('id')}': invalid gatewayDirection '{g.get('gatewayDirection')}'")

    validate_nodes(tasks,    VALID_TASK_TYPES,    "Task")
    validate_nodes(events,   VALID_EVENT_TYPES,   "Event",   check_event)
    validate_nodes(gateways, VALID_GATEWAY_TYPES, "Gateway", check_gateway)

    incoming = defaultdict(int)
    outgoing = defaultdict(int)

    for sf in seq:
        src, tgt = sf.get("from"), sf.get("to")
        if src not in node_ids:
            errors.append(f"sequence_flow '{sf.get('id')}': unknown 'from' node '{src}'")
        if tgt not in node_ids:
            errors.append(f"sequence_flow '{sf.get('id')}': unknown 'to' node '{tgt}'")
        if src in node_ids and tgt in node_ids:
            src_pool = next((l["pool"] for l in lanes if l["id"] == node_to_lane.get(src)), None)
            tgt_pool = next((l["pool"] for l in lanes if l["id"] == node_to_lane.get(tgt)), None)
            if src_pool and tgt_pool and src_pool != tgt_pool:
                warnings.append(f"sequence_flow '{sf.get('id')}' crosses pools.")
            outgoing[src] += 1
            incoming[tgt] += 1

    valid_refs = node_ids | pool_ids
    for mf in mflows:
        if mf.get("from") not in valid_refs or mf.get("to") not in valid_refs:
            errors.append(f"message_flow '{mf.get('id')}': unknown 'from' or 'to' reference")
        if not mf.get("name"):
            errors.append(f"message_flow '{mf.get('id')}': missing 'name'")

    return len(errors) == 0 and len(warnings) == 0, errors, warnings


def extract_bpmn(process_description, prompt_type="zero-shot", output_file=None, retries=2):
    # Prompt is left as defined in your original script
    prompt = f""" You are a BPMN 2.0 expert. Extract a structured BPMN model from the process description below.

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
    
    try:
        model = load_model()
    except RuntimeError as e:
        print(e)
        return None

    # Phi-4 follows standard chat completion API
    result = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a business process modeling expert. "
    "Extract structured BPMN elements from process descriptions. "
    "Always output valid JSON matching the provided schema. "
    "Never use a 'nodes' array — always use separate 'tasks', 'events', and 'gateways' arrays. "
    "Every task, event, and gateway must belong to a declared lane, and every lane must belong to a declared pool. "
    "Every pool must have at most one startEvent and at least one endEvent. "
    "A pool represents one organisation or external entity like a company with multiple departments or a customer as an external entity. "
    "Roles and departments within the same organisation are LANES inside ONE pool, not separate pools. "
    "External parties (customers, suppliers, third parties) are ALWAYS modeled as their own pool."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_object",
            "schema": BPMN_SCHEMA
        },
        temperature=0.0,
        max_tokens=8192
    )
    
    if isinstance(result, dict) and 'choices' in result:
        result_text = result['choices'][0]['message']['content']
    else:
        result_text = str(result)
    
    try:
        json_result = json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        json_result = None

    # Validation/Retry logic remains same as original script
    if json_result and output_file:
        save_json_to_file(json_result, output_file)
        
    return json_result

if __name__ == "__main__":
    case_name = "case_21" 

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    CASES_DIR = os.path.join(PROJECT_ROOT, "cases")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(CASES_DIR, f"{case_name}.txt")
    out_file = os.path.join(OUTPUT_DIR, f"{case_name}_zero_shot_phi4.json")

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError

        with open(input_path, "r", encoding="utf-8") as f:
            process_text = f.read()
        
        print(f"🤖 Phi-4 is analyzing '{case_name}'...")
        bpmn_json = extract_bpmn(process_text, output_file=out_file, retries=0)
        
    except FileNotFoundError:
        print(f"Error: The file '{case_name}.txt' was not found.")