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
            repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            filename="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            n_ctx=8192,
            n_gpu_layers=1,
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
                    "from_pool": {"type": "string", "description": "The pool id of the sending node."},
                    "to_pool":   {"type": "string", "description": "The pool id of the receiving node."}
                },
                "required": ["id", "from", "to", "from_pool", "to_pool"],
                "additionalProperties": False
            }
        },
        "data": {
            "type": "array",
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
        "data_associations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to":   {"type": "string"},
                    "type": {"type": "string", "enum": ["input", "output"]}
                },
                "required": ["from", "to", "type"],
                "additionalProperties": False
            }
        }
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
    """
    Returns (is_fully_valid, errors, warnings).
    - errors:   blocking problems
    - warnings: structural problems (missing start/end events, cross-pool sequence flows)
    """
    errors, warnings = [], []

    if not isinstance(obj, dict):
        return False, ["Output is not a JSON object"], []

    # ── Required top-level keys ───────────────────────────────────────────────
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

    # ── Pools & lanes ─────────────────────────────────────────────────────────
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

    # ── Node validation ───────────────────────────────────────────────────────
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

    # ── Sequence flows ────────────────────────────────────────────────────────
    incoming = defaultdict(int)
    outgoing = defaultdict(int)

    for sf in seq:
        src, tgt = sf.get("from"), sf.get("to")
        if src not in node_ids:
            errors.append(f"sequence_flow '{sf.get('id')}': unknown 'from' node '{src}'")
        if tgt not in node_ids:
            errors.append(f"sequence_flow '{sf.get('id')}': unknown 'to' node '{tgt}'")
        if src in node_ids and tgt in node_ids:
            # Cross-lane within same pool is fine; cross-pool is not
            src_pool = next((l["pool"] for l in lanes if l["id"] == node_to_lane.get(src)), None)
            tgt_pool = next((l["pool"] for l in lanes if l["id"] == node_to_lane.get(tgt)), None)
            if src_pool and tgt_pool and src_pool != tgt_pool:
                warnings.append(
                    f"sequence_flow '{sf.get('id')}' crosses pools ('{src_pool}' -> '{tgt_pool}'). "
                    f"Use message_flows for inter-pool communication."
                )
            outgoing[src] += 1
            incoming[tgt] += 1

    # ── Message flows ─────────────────────────────────────────────────────────
    valid_refs = node_ids | pool_ids
    for mf in mflows:
        if mf.get("from") not in valid_refs or mf.get("to") not in valid_refs:
            errors.append(f"message_flow '{mf.get('id')}': unknown 'from' or 'to' reference")
        if not mf.get("name"):
            errors.append(f"message_flow '{mf.get('id')}': missing 'name'")

    # ── Structural warnings ───────────────────────────────────────────────────
    # Each pool needs at least one start and end event
    lane_to_pool = {l["id"]: l["pool"] for l in lanes}
    pool_has_start = defaultdict(bool)
    pool_has_end   = defaultdict(bool)
    for e in events:
        pid = lane_to_pool.get(e.get("lane"))
        if e.get("type") == "startEvent": pool_has_start[pid] = True
        if e.get("type") == "endEvent":   pool_has_end[pid]   = True
    for pid in pool_ids:
        if not pool_has_start[pid]:
            warnings.append(f"Pool '{pid}' has no startEvent")
        if not pool_has_end[pid]:
            warnings.append(f"Pool '{pid}' has no endEvent")

    # Each task must have at least one incoming and one outgoing flow
    for t in tasks:
        tid = t.get("id")
        if tid not in node_ids: continue
        if incoming[tid] == 0:
            warnings.append(f"Task '{tid}' has no incoming sequence flow")
        if outgoing[tid] == 0:
            warnings.append(f"Task '{tid}' has no outgoing sequence flow")

    # Each gateway should have correct in/out counts
    for g in gateways:
        gid = g.get("id")
        if gid not in node_ids: continue
        direction = g.get("gatewayDirection")
        if direction == "diverging" and incoming[gid] == 0:
            warnings.append(f"Diverging gateway '{gid}' has no incoming flow")
        if direction == "diverging" and outgoing[gid] < 2:
            warnings.append(f"Diverging gateway '{gid}' has fewer than 2 outgoing flows")
        if direction == "converging" and incoming[gid] < 2:
            warnings.append(f"Converging gateway '{gid}' has fewer than 2 incoming flows")
        if direction == "converging" and outgoing[gid] == 0:
            warnings.append(f"Converging gateway '{gid}' has no outgoing flow")

    # Start events should have no incoming, end events no outgoing
    for e in events:
        eid = e.get("id")
        if eid not in node_ids: continue
        if e.get("type") == "startEvent" and incoming[eid] > 0:
            warnings.append(f"startEvent '{eid}' has incoming sequence flows")
        if e.get("type") == "endEvent" and outgoing[eid] > 0:
            warnings.append(f"endEvent '{eid}' has outgoing sequence flows")

    return len(errors) == 0 and len(warnings) == 0, errors, warnings


def _format_issues(errors, warnings):
    lines = []
    if errors:
        lines.append("ERRORS:")
        lines.extend(f"  - {e}" for e in errors)
    if warnings:
        lines.append("WARNINGS:")
        lines.extend(f"  - {w}" for w in warnings)
    return "\n".join(lines)


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

Your goal is NOT to simplify the process, but to faithfully model ALL BPMN elements:
- pools
- lanes
- tasks
- events
- gateways
- sequence flows
- message flows


--- CORE MODELING PRINCIPLES ---


1. MODEL THE FULL PROCESS (NO SIMPLIFICATION)
Do not omit steps. Do not merge tasks. Do not reduce structure.


2. POOLS AND LANES (CRITICAL)
- Each organisation = one pool
- Each role/actor = one lane inside its organisation
- Cross-organisation interaction MUST use message flows
- NEVER connect sequence flows across pools


3. MESSAGE FLOWS VS TASKS (VERY IMPORTANT)
- If information is sent between pools → use message flows
- Do NOT model communication as tasks
- from_pool and to_pool in message_flows cannot have the same name (even if they are the same pool) — if communication happens within the same pool, use sequence flows instead
- Use:
  - intermediateThrowEvent for sending
  - intermediateCatchEvent for receiving



4. EVENTS ARE REQUIRED
Each pool MUST contain:
- Exactly one startEvent, not all lanes needs a startEvent
- at least one endEvent


Use message events when communication occurs.


5. GATEWAYS (STRICT RULES)
- Decisions → exclusiveGateway (XOR)
- Parallel work → parallelGateway (AND)


CRITICAL:
- Parallel always has BOTH:
  - a split (diverging)
  - a join (converging)


- If two branches must finish before continuing → you MUST use a parallel join


6. SEQUENCE FLOW STRUCTURE (CRITICAL)
- Tasks must NOT be isolated
- Each task must be part of a continuous flow
- Typical structure:
  startEvent → tasks → gateways → tasks → endEvent


- A task must:
  - have 1 incoming flow
  - have 1 outgoing flow


7. CORRECT FLOW BEFORE DECISIONS
- Do NOT decide before all required inputs are available
- If multiple evaluations happen (e.g. medical + insurance):
  → first split in parallel
  → then join
  → THEN make decision


8. NO FAKE STRUCTURE
- Do NOT invent flows just to satisfy constraints
- Structure must reflect real process logic


--- FINAL SELF-CHECK (MANDATORY) ---


Before outputting JSON, verify:


- Are there multiple pools if multiple organisations exist?
- Are all cross-pool interactions modeled as message flows (NOT sequence flows)?
- Are there any message flows within the same pool? If so, change them to sequence flows.
- Are parallel branches correctly split AND joined?
- Does each pool have at least one endEvent?
- Are decisions only made AFTER required information is available?
- Are tasks connected in a continuous flow (no isolated tasks)?
- Are message events used instead of tasks for communication?


If ANY of these are violated, fix the model before output.


---


Return ONLY valid JSON matching the schema.
Do not explain anything.
## Process description
{process_description}


Before returning, verify: does every task have an incoming AND outgoing sequence_flow entry? If not, add the missing flows now.
Message flows may only connect nodes that belong to different pools. Never use a message flow between two nodes in the same pool. Communication within the same pool must be modeled with sequence flows.
Sequence flows may never cross pool boundaries. Cross-pool communication must be modeled with message flows between a sending node in one pool and a receiving event in another pool.
Return a JSON object matching the schema exactly. Do not merge tasks. Do not skip gateways."""
    
    

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
    case_name = "case_21" # <-- Just change this to run a different case with few-shot examples

    SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    CASES_DIR     = os.path.join(PROJECT_ROOT, "cases")
    FEW_SHOT_DIR  = os.path.join(PROJECT_ROOT, "few_shot_cases")
    OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(CASES_DIR, f"{case_name}.txt")
    out_file   = os.path.join(OUTPUT_DIR, f"{case_name}_few_shot_mistral.json")

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
            case_ids=[13, 14] # choose cases for few-shot examples
        )

        if bpmn_json:
            print(f"Output saved to: outputs/{case_name}_few_shot_mistral.json")
        else:
            print("Model extraction failed. Check the logs above.")