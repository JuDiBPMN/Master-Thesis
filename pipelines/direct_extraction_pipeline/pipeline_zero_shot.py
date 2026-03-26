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
        prompt = f"""Extract all tasks, events, actors, and sequence flows from this process description.

Process description: {process_description}

### Rules
1. Each distinct action by a single lane = one task. Never merge actions.
2. Decision points (if/else, approved/rejected) = EXCLUSIVE gateway (XOR). Add a gateway node, not just branching flows.
3. Simultaneous/parallel actions ("at the same time", "simultaneously", "while") = PARALLEL gateway (AND). Add both a split and a join gateway.
4. Every pool must have exactly one startEvent and one endEvent.
5. Gateway IDs must appear in sequence_flows; tasks that are "behind" a gateway connect TO the gateway, not directly to each other.
6. NEVER leave gateways empty if the process has decisions or parallelism.

## Signal words → gateway type
- "if", "approved", "rejected", "otherwise", "either" → exclusive (XOR)
- "simultaneously", "at the same time", "in parallel", "while" → parallel (AND)
- "one of", "any of" → inclusive (OR)

- A parallel gateway ALWAYS comes in pairs: one split (1 in, many out) AND one join (many in, 1 out)
- An exclusive split (if/else) must have exactly 2 outgoing flows with conditions, one per outcome
- Never use an exclusive gateway to join parallel branches — use a parallel gateway join
- For each out exclusive split there must be an exclusive join, and for each parallel split there must be a parallel join.
- Give the gateway a meaningfull name like "gateway_approved_split" or "gateway_approved_join" to make it clear which split and join belong together and what the gateway represents. Never name them "gateway1", "gateway2", etc.

- If a task produces a document or artifact mentioned in the description, add it to "data" and create a data_association with type "output"
- For sequence flow: A task should have one incoming sequence flow and one outgoing sequence flow, except for the start and end event in the pool. 
- Task "name" should be a short verb phrase only (e.g. "Assess request"). Never include the lane's name in the task name.
- If a task consumes a document, add a data_association with type "input"
- Common signals for data objects: "record X", "submit a X", "send a X", "initiate a X", 
  "receives a X", "notified of X" → these produce or consume data objects- An exclusive gateway splits AFTER the decision task. The gateway's "from" is the task that makes 
  the decision, "to" lists the outcome branches. Never leave "from" empty.
- data_association "from"/"to" values must exactly match an id already defined in the "tasks" or 
  "data" arrays. Never invent new ids.

  ## Pools vs Lanes
- A pool = an organisation. A lane = a role/person within that organisation.
- If actors work for the same organisation → ONE pool, each actor is a lane.
- Never create one pool per role or person. Pool names are organisations, not roles.

## Example:
""pools": [ "id": "company", "name": "Company" ],
"lanes": [
  "id": "employee", "name": "Employee", "pool": "company",
   "id": "manager",  "name": "Manager",  "pool": "company" ]
## Wrong example (never do this):
"pools": [ "id": "employee_pool", "name": "Employee Pool" ,  "id": "manager_pool", "name": "Manager Pool" ]"""
    
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