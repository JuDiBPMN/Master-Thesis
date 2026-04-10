try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    _LLAMA_CPP_AVAILABLE = False

import json
import sys

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
            n_gpu_layers=0,
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


def extract_bpmn(process_description, prompt_type="zero-shot", output_file=None, retries=2):
    if prompt_type == "zero-shot":
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
    
    print(f"Initial prompt:\n{prompt}\n")
    
    try:
        model = load_model()
    except RuntimeError as e:
        print(e)
        return None

    result = model.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a business process modeling expert. Extract tasks and flows from process descriptions."},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_object",
            "schema": BPMN_SCHEMA
        },
        temperature=0.0,
        max_tokens=4096 # dit kan hoger als de business case gigantisch is en de json is afgesneden
    )
    
    if isinstance(result, dict) and 'choices' in result:
        result_text = result['choices'][0]['message']['content']
    else:
        result_text = str(result)
    
    print("Model output:\n", result_text)
    
    try:
        json_result = json.loads(result_text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        json_result = None

    attempt = 0
    last_result_text = result_text
    while (not is_valid_bpmn(json_result)) and attempt < retries:
        print(f"\nParsed JSON invalid or contained placeholders; retrying ({attempt+1}/{retries})...")

        clar_prompt = f"""The previous output contained placeholders or empty values. Extract concrete task names and actors from the process description.

Process description: {process_description}

Previous invalid output: {last_result_text}

Provide concrete names for all tasks and actors based on the process description."""
        
        print(f"Retry prompt:\n{clar_prompt}\n")
        
        result = model.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a business process modeling expert. Extract tasks and flows from process descriptions with concrete names only."},
                {"role": "user", "content": clar_prompt}
            ],
            response_format={
                "type": "json_object",
                "schema": BPMN_SCHEMA
            },
            temperature=0.0,
            max_tokens=4096
        )
        
        if isinstance(result, dict) and 'choices' in result:
            last_result_text = result['choices'][0]['message']['content']
        else:
            last_result_text = str(result)
            
        print("Model output (retry):\n", last_result_text)
        
        try:
            json_result = json.loads(last_result_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error on retry: {e}")
            json_result = None
            
        attempt += 1

    if json_result is None:
        print("\nError: Failed to parse JSON from model output")
    else:
        if not is_valid_bpmn(json_result):
            print("\nWarning: Parsed JSON did not pass validation (contained placeholders or inconsistent ids)")
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

import os
import sys

if __name__ == "__main__":
    # ---------------------------------------------------------
    # CONFIGURATION: Hier gewoon case namen invullen 
    case_name = "case_21" 
    # ---------------------------------------------------------

    # 1. Path Discovery (Stays Partner-Proof & Subfolder-Aware)
    # Finds the 'Master-Thesis' root by hopping up two levels
    # Dit is best omdat we zo dezelfde pathnames verkrijgen 
    # (de vorige code gaf andere pathnames (door andere pc))

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    CASES_DIR = os.path.join(PROJECT_ROOT, "cases")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Construct dynamic paths
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
        
        # 3. Run extraction
        print(f"🤖 LLM is analyzing '{case_name}'...")
        bpmn_json = extract_bpmn(process_text, output_file=out_file, retries=2)
        
        if bpmn_json:
            print(f"Success! View the output in the sidebar under: outputs/{case_name}_model_extracted_bpmn.json")
        else:
            print("Model extraction failed. Check the logs above.")
            
    except FileNotFoundError:
        print(f"Error: The file '{case_name}.txt' was not found in {CASES_DIR}")
        print(f"Check if you have a file named '{case_name}.txt' in that folder.")