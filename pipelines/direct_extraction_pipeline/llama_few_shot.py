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
            filename="llama-2-7b-chat.Q6_K.gguf",
            n_ctx=4096,
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

def load_case(case_id: int, cases_dir: str = "few_shot_cases") -> dict:
    """Load a single case by its ID number."""
    txt_path = os.path.join(cases_dir, f"case_{case_id}.txt")
    json_path = os.path.join(cases_dir, f"case_{case_id}.json")
    
    with open(txt_path, "r") as f:
        text = f.read()
    with open(json_path, "r") as f:
        data = json.load(f)
    
    return {
        "name": f"Case {case_id}",
        "text": text,
        "json": data
    }

def load_selected_cases(case_ids: list[int], cases_dir: str = "few_shot_cases") -> list[dict]:
    """Load only the specific cases you want by ID."""
    return [load_case(case_id, cases_dir) for case_id in case_ids]


# ── Validation ────────────────────────────────────────────────────────────────

def save_json_to_file(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


from collections import defaultdict

def is_valid_bpmn(obj):
    """Returns a dict:
      {
        "valid": bool,       # True only if errors == [] and warnings == []
        "errors":   [...],   # Blocking problems (would produce invalid/unusable BPMN)
        "warnings": [...],   # Structural problems (incomplete or incorrect diagram)
      }
    
    Error categories (matching thesis):
      1. Undeclared participant references   → error
      2. Missing start or end events         → error
      3. Dangling sequence flow references   → error
      4. Cross-pool sequence flows           → error
    
    Additional structural checks:
      - Required top-level fields present    → error
      - Required fields on each element     → error
      - Placeholder / empty values          → warning
      - Message flows pointing to tasks     → warning
      - Isolated tasks (no in/out flow)     → warning
    """
    errors   = []
    warnings = []

    # ── 0. Top-level type & required fields ──────────────────────────────────
    if not isinstance(obj, dict):
        return {"valid": False, "errors": ["Output is not a JSON object"], "warnings": []}

    for field in ("pools", "lanes", "tasks", "events", "gateways",
                  "sequence_flows", "message_flows"):
        if field not in obj:
            errors.append(f"Missing required top-level field: '{field}'")
        elif not isinstance(obj[field], list):
            errors.append(f"Field '{field}' must be an array")

    # If the basic structure is broken, stop early — further checks would crash.
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    pools     = obj.get("pools", [])
    lanes     = obj.get("lanes", [])
    tasks     = obj.get("tasks", [])
    events    = obj.get("events", [])
    gateways  = obj.get("gateways", [])
    seq_flows = obj.get("sequence_flows", [])
    msg_flows = obj.get("message_flows", [])

    PLACEHOLDER = {"", "...", "none", "null", "n/a", "tbd", "unknown"}

    def is_placeholder(v):
        return v is None or str(v).strip().lower() in PLACEHOLDER

    # ── 1. Build lookup maps ──────────────────────────────────────────────────
    pool_ids = {}   # id → pool obj
    for p in pools:
        pid = p.get("id", "").strip()
        if not pid:
            errors.append("A pool is missing its 'id' field")
            continue
        if not p.get("name", "").strip():
            warnings.append(f"Pool '{pid}' has an empty or missing 'name'")
        pool_ids[pid] = p

    lane_ids     = {}   # lane_id → pool_id
    lane_to_pool = {}   # lane_id → pool_id  (alias for readability)
    for l in lanes:
        lid  = l.get("id",   "").strip()
        lpool = l.get("pool", "").strip()
        if not lid:
            errors.append("A lane is missing its 'id' field")
            continue
        if not l.get("name", "").strip():
            warnings.append(f"Lane '{lid}' has an empty or missing 'name'")
        # ── ERROR category 1: undeclared participant references ──
        if lpool not in pool_ids:
            errors.append(
                f"[Undeclared participant] Lane '{lid}' references unknown pool '{lpool}'"
            )
        lane_ids[lid]     = lpool
        lane_to_pool[lid] = lpool

    # All flow nodes: tasks, events, gateways
    node_ids     = {}   # node_id → pool_id  (resolved via lane)
    node_to_lane = {}   # node_id → lane_id

    def register_node(element, kind):
        eid   = element.get("id",   "").strip()
        ename = element.get("name", "").strip()
        elane = element.get("lane", "").strip()

        if not eid:
            errors.append(f"A {kind} is missing its 'id' field")
            return
        if is_placeholder(ename):
            warnings.append(f"[Placeholder value] {kind} '{eid}' has empty/placeholder name '{ename}'")
        # ── ERROR category 1: undeclared participant references ──
        if elane not in lane_ids:
            errors.append(
                f"[Undeclared participant] {kind} '{eid}' references unknown lane '{elane}'"
            )
            node_ids[eid]     = None
            node_to_lane[eid] = elane
        else:
            node_ids[eid]     = lane_ids[elane]   # pool_id
            node_to_lane[eid] = elane

    for t in tasks:
        register_node(t, "task")
    for e in events:
        register_node(e, "event")
    for g in gateways:
        register_node(g, "gateway")

    # ── 2. Missing start / end events per pool ────────────────────────────────
    # ── ERROR category 2: missing start or end events ────────────────────────
    pool_has_start = {pid: False for pid in pool_ids}
    pool_has_end   = {pid: False for pid in pool_ids}

    for e in events:
        etype = e.get("type", "")
        elane = e.get("lane", "").strip()
        pool  = lane_to_pool.get(elane)
        if pool is None:
            continue
        if etype == "startEvent":
            pool_has_start[pool] = True
        elif etype == "endEvent":
            pool_has_end[pool]   = True

    for pid in pool_ids:
        if not pool_has_start[pid]:
            errors.append(
                f"[Missing start/end event] Pool '{pid}' has no startEvent"
            )
        if not pool_has_end[pid]:
            errors.append(
                f"[Missing start/end event] Pool '{pid}' has no endEvent"
            )

    # ── 3. Sequence flow validation ───────────────────────────────────────────
    task_ids_set = {t.get("id", "") for t in tasks}

    for sf in seq_flows:
        sf_id   = sf.get("id",   "?")
        sf_from = sf.get("from", "").strip()
        sf_to   = sf.get("to",   "").strip()

        # ── ERROR category 3: dangling sequence flow references ──
        if sf_from not in node_ids:
            errors.append(
                f"[Dangling sequence flow] '{sf_id}' references undeclared source '{sf_from}'"
            )
        if sf_to not in node_ids:
            errors.append(
                f"[Dangling sequence flow] '{sf_id}' references undeclared target '{sf_to}'"
            )

        # ── ERROR category 4: cross-pool sequence flows ──
        if sf_from in node_ids and sf_to in node_ids:
            pool_from = node_ids[sf_from]
            pool_to   = node_ids[sf_to]
            if pool_from is not None and pool_to is not None and pool_from != pool_to:
                errors.append(
                    f"[Cross-pool sequence flow] '{sf_id}' crosses from pool '{pool_from}' "
                    f"to pool '{pool_to}' — use message flows instead"
                )

    # ── 4. Message flow validation ────────────────────────────────────────────
    # Message flows may legitimately connect tasks/events across pools.
    # We only warn if the target is a plain task (not an event), since the
    # thesis rules require message flows to target catch events.
    for mf in msg_flows:
        mf_id   = mf.get("id",   "?")
        mf_to   = mf.get("to",   "").strip()
        mf_from = mf.get("from", "").strip()
        if not mf.get("name", "").strip():
            warnings.append(f"[Placeholder value] message_flow '{mf_id}' has no name")
        if mf_to in task_ids_set:
            warnings.append(
                f"[Message flow to task] '{mf_id}' targets task '{mf_to}' — "
                f"message flows should target intermediateCatchEvent or startEvent"
            )
        if mf_from in task_ids_set:
            warnings.append(
                f"[Message flow from task] '{mf_id}' originates from task '{mf_from}' — "
                f"consider using an intermediateThrowEvent"
            )

    # ── 5. Isolated nodes (no incoming or outgoing sequence flow) ─────────────
    nodes_with_incoming = {sf.get("to",   "") for sf in seq_flows}
    nodes_with_outgoing = {sf.get("from", "") for sf in seq_flows}

    for t in tasks:
        tid = t.get("id", "")
        if not tid:
            continue
        if tid not in nodes_with_incoming and tid not in nodes_with_outgoing:
            warnings.append(
                f"[Isolated node] Task '{tid}' has no sequence flows at all"
            )
        elif tid not in nodes_with_incoming:
            warnings.append(
                f"[Isolated node] Task '{tid}' has no incoming sequence flow"
            )
        elif tid not in nodes_with_outgoing:
            warnings.append(
                f"[Isolated node] Task '{tid}' has no outgoing sequence flow"
            )

    valid = (len(errors) == 0 and len(warnings) == 0)
    return {"valid": valid, "errors": errors, "warnings": warnings}


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
        max_tokens=8196
    )
    if isinstance(result, dict) and "choices" in result:
        return result["choices"][0]["message"]["content"]
    return str(result)


# ── Main extraction function ───────────────────────────────────────────────────

def extract_bpmn_few_shot(process_description, case_name, few_shot_dir, output_file=None, retries=0, case_ids=None):
    if case_ids is not None:
        cases = load_selected_cases(case_ids, cases_dir=few_shot_dir)
    else:
        cases = load_few_shot_cases(few_shot_dir, exclude_case=case_name)
    
    few_shot_block = build_few_shot_block(cases)

    prompt = f"""You are a BPMN 2.0 expert. Extract a structured BPMN model from the process description below.

Here are {len(cases)} example(s) showing process descriptions and their correct BPMN JSON outputs:

{few_shot_block}

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

    print(f"Running few-shot extraction for '{case_name}' with {len(cases)} example(s)...")
    result_text = _call_model(model, prompt)
    print("Model output:\n", result_text)

    try:
        json_result = json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        json_result = None

    validation  = is_valid_bpmn(json_result) if json_result else {"valid": False, "errors": ["Failed to parse JSON"], "warnings": []}
    fully_valid = validation["valid"]
    errors      = validation["errors"]
    warnings    = validation["warnings"]
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

        validation = is_valid_bpmn(json_result) if json_result else {"valid": False, "errors": ["Failed to parse JSON"], "warnings": []}
        fully_valid = validation["valid"]
        errors      = validation["errors"]
        warnings    = validation["warnings"]

    # ── Save output ───────────────────────────────────────────────────────────
    if json_result is None:
        print("\nError: Failed to parse JSON from model output")
    else:
        validation = is_valid_bpmn(json_result)
        if not validation["valid"]:
            print("\nWarning: Parsed JSON did not pass validation.")
            for e in validation["errors"]:
                print(f"  ERROR:   {e}")
            for w in validation["warnings"]:
                print(f"  WARNING: {w}")
            if output_file:
                try:
                    save_json_to_file(json_result, output_file.replace('.json', '_invalid.json'))
                    print(f"Wrote invalid parsed JSON to {output_file.replace('.json', '_invalid.json')}")
                except Exception as e:
                    print(f"Failed to write invalid JSON to {output_file}: {e}")
        else:
            if output_file:
                try:
                    save_json_to_file(json_result, output_file)
                    print(f"Wrote parsed JSON to {output_file}")
                except Exception as e:
                    print(f"Failed to write JSON to {output_file}: {e}")

    return json_result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    case_name = "case_21" # <-- Just change this to run a different case with few-shot examples

    SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    CASES_DIR     = os.path.join(PROJECT_ROOT, "cases")
    FEW_SHOT_DIR  = os.path.join(PROJECT_ROOT, "few_shot_cases")
    OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(CASES_DIR, f"{case_name}.txt")
    out_file   = os.path.join(OUTPUT_DIR, f"{case_name}_llama_B_few_shot.json")

    print("--- Few-Shot Pipeline Started ---")
    print(f"Case:          {case_name}")
    print(f"Cases dir:     {CASES_DIR}")
    print(f"Few-shot dir:  {FEW_SHOT_DIR}")
    print(f"Output:        {out_file}")
    print("---------------------------------")

    if not os.path.exists(input_path):
        print(f"Error: Case file not found: {input_path}")
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            process_text = f.read()

        bpmn_json = extract_bpmn_few_shot(
            process_description=process_text,
            case_name=case_name,
            few_shot_dir=FEW_SHOT_DIR,
            output_file=out_file,
            retries=0,
            case_ids=[12] # choose cases for few-shot examples
        )

        if bpmn_json:
            print(f"Output saved to: outputs/{case_name}_llama_few_shot.json")
        else:
            print("Model extraction failed. Check the logs above.")