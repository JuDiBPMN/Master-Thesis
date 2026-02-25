try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except Exception:
    Llama = None
    _LLAMA_CPP_AVAILABLE = False

import json
import os

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
            n_gpu_layers=-1,
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
      "description": "BPMN event nodes only (start, end, intermediate). Do NOT put tasks or gateways here. Use startEvent for the beginning of a process, endEvent for the end, intermediateCatchEvent for waiting points (e.g. 'food is prepared'), intermediateThrowEvent for sending signals.",
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
      "description": "BPMN gateway nodes only (splits and joins). Do NOT put tasks or events here. Use exclusiveGateway (XOR) for either/or decisions, parallelGateway (AND) for concurrent splits/joins, inclusiveGateway (OR) for one-or-more branches.",
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
      "description": "Control-flow edges. Both 'from' and 'to' must be node ids within the same pool. Cross-pool communication must use message_flows instead.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "from": {
            "description": "Id of the source node. Must exist in tasks, events, or gateways.",
            "type": "string"
          },
          "to": {
            "description": "Id of the target node. Must exist in tasks, events, or gateways.",
            "type": "string"
          },
          "condition": {
            "description": "Condition label for flows leaving an exclusiveGateway or inclusiveGateway.",
            "type": "string"
          },
          "isDefault": {
            "description": "True if this is the default outgoing flow from a gateway when no condition matches.",
            "type": "boolean"
          },
          "participant": {
            "description": "Pool id that both 'from' and 'to' belong to.",
            "type": "string"
          }
        },
        "required": ["from", "to"]
      }
    },
    "message_flows": {
      "description": "Communication edges between different pools. 'from' and 'to' must belong to different pools.",
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "from": {"type": "string"},
          "to": {"type": "string"},
          "name": {
            "description": "Label describing what is being communicated.",
            "type": "string"
          }
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


# --- Few-shot example (case_6) ---
# This example is excluded when running on case_6 itself to avoid eval contamination.
FEW_SHOT_EXAMPLE = {
  "participants": [
    { "id": "restaurant", "name": "Restaurant", "type": "pool" },
    { "id": "platform", "name": "Platform", "type": "pool" },
    {
      "id": "courier_network",
      "name": "Courier Network",
      "type": "pool",
      "lanes": [
        { "id": "scooter_courier", "name": "Scooter Courier" },
        { "id": "bike_courier", "name": "Bike Courier" }
      ]
    }
  ],
  "tasks": [
    { "id": "prepare_food", "name": "Prepare Food", "type": "task", "participant": "restaurant" },

    { "id": "forward_order", "name": "Forward Order to Restaurant", "type": "task", "participant": "platform" },
    { "id": "reach_out_courier", "name": "Reach Out to Courier Network", "type": "task", "participant": "platform" },

    {
      "id": "scooter_ride_to_restaurant",
      "name": "Ride to Restaurant",
      "type": "task",
      "participant": "courier_network",
      "lane": "scooter_courier"
    },
    {
      "id": "bike_ride_to_restaurant",
      "name": "Ride to Restaurant",
      "type": "task",
      "participant": "courier_network",
      "lane": "bike_courier"
    },

    {
      "id": "scooter_take_meal",
      "name": "Take Meal to Customer",
      "type": "task",
      "participant": "courier_network",
      "lane": "scooter_courier"
    },
    {
      "id": "bike_take_meal",
      "name": "Take Meal to Customer",
      "type": "task",
      "participant": "courier_network",
      "lane": "bike_courier"
    }
  ],
  "events": [
    {
      "id": "order_received",
      "name": "Order Received",
      "type": "startEvent",
      "participant": "platform",
      "eventDefinition": "none"
    },
    {
      "id": "restaurant_order_received",
      "name": "Order Received (Restaurant)",
      "type": "startEvent",
      "participant": "restaurant",
      "eventDefinition": "message"
    },
    {
      "id": "courier_request_received",
      "name": "Courier Request Received",
      "type": "startEvent",
      "participant": "courier_network",
      "eventDefinition": "message"
    },

    {
      "id": "food_prepared",
      "name": "Food is Prepared",
      "type": "intermediateCatchEvent",
      "participant": "restaurant",
      "eventDefinition": "none"
    },
    {
      "id": "courier_arrived",
      "name": "Courier has Arrived",
      "type": "intermediateCatchEvent",
      "participant": "restaurant",
      "eventDefinition": "message"
    },

    {
      "id": "pickup_ready_received",
      "name": "Pickup Ready",
      "type": "intermediateCatchEvent",
      "participant": "courier_network",
      "eventDefinition": "message"
    },

    {
      "id": "delivery_confirmed",
      "name": "Delivery Confirmed",
      "type": "intermediateCatchEvent",
      "participant": "platform",
      "eventDefinition": "message"
    },
    {
      "id": "order_complete",
      "name": "Order Complete",
      "type": "endEvent",
      "participant": "platform",
      "eventDefinition": "none"
    }
  ],
  "gateways": [
    { "id": "split_parallel", "name": "", "type": "parallelGateway", "participant": "platform", "gatewayDirection": "diverging" },

    { "id": "distance_split_1", "name": "Distance < 5 km?", "type": "exclusiveGateway", "participant": "courier_network", "gatewayDirection": "diverging" },
    { "id": "courier_join_1", "name": "", "type": "exclusiveGateway", "participant": "courier_network", "gatewayDirection": "converging" },

    { "id": "join_parallel", "name": "", "type": "parallelGateway", "participant": "restaurant", "gatewayDirection": "converging" },

    { "id": "distance_split_2", "name": "Distance < 5 km?", "type": "exclusiveGateway", "participant": "courier_network", "gatewayDirection": "diverging" },
    { "id": "courier_join_2", "name": "", "type": "exclusiveGateway", "participant": "courier_network", "gatewayDirection": "converging" }
  ],
  "sequence_flows": [
    { "from": "order_received", "to": "split_parallel" },
    { "from": "split_parallel", "to": "forward_order" },
    { "from": "split_parallel", "to": "reach_out_courier" },

    { "from": "restaurant_order_received", "to": "prepare_food" },
    { "from": "prepare_food", "to": "food_prepared" },
    { "from": "food_prepared", "to": "join_parallel" },
    { "from": "courier_arrived", "to": "join_parallel" },

    { "from": "courier_request_received", "to": "distance_split_1" },
    { "from": "distance_split_1", "to": "bike_ride_to_restaurant", "condition": "< 5 km" },
    { "from": "distance_split_1", "to": "scooter_ride_to_restaurant", "condition": ">= 5 km" },
    { "from": "bike_ride_to_restaurant", "to": "courier_join_1" },
    { "from": "scooter_ride_to_restaurant", "to": "courier_join_1" },

    { "from": "pickup_ready_received", "to": "distance_split_2" },
    { "from": "distance_split_2", "to": "bike_take_meal", "condition": "< 5 km" },
    { "from": "distance_split_2", "to": "scooter_take_meal", "condition": ">= 5 km" },
    { "from": "bike_take_meal", "to": "courier_join_2" },
    { "from": "scooter_take_meal", "to": "courier_join_2" },

    { "from": "delivery_confirmed", "to": "order_complete" }
  ],
  "message_flows": [
    { "from": "forward_order", "to": "restaurant_order_received", "name": "Order details" },
    { "from": "reach_out_courier", "to": "courier_request_received", "name": "Courier request" },

    { "from": "courier_join_1", "to": "courier_arrived", "name": "Courier arrived" },

    { "from": "join_parallel", "to": "pickup_ready_received", "name": "Pickup ready" },

    { "from": "courier_join_2", "to": "delivery_confirmed", "name": "Order delivered" }
  ],
  "data": []
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
        return False, "Output is not a JSON object"

    if "nodes" in obj:
        return False, (
            "Output contains a 'nodes' array, which is not valid. "
            "You must use three separate arrays: 'tasks', 'events', and 'gateways'. "
            "Do not combine them into a single 'nodes' array."
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
            name = node.get("name", "")
            if str(name).strip() in ("...", "None", ""):
                return f"{label} '{nid}' has a placeholder or empty name"
            node_ids.add(nid)
        return None

    tasks    = obj.get("tasks", [])
    events   = obj.get("events", [])
    gateways = obj.get("gateways", [])

    if not isinstance(tasks, list) or len(tasks) == 0:
        return False, "tasks must be a non-empty array"
    if not isinstance(events, list):
        return False, "events must be an array (can be empty)"
    if not isinstance(gateways, list):
        return False, "gateways must be an array (can be empty)"

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
        max_tokens=4096
    )
    if isinstance(result, dict) and "choices" in result:
        return result["choices"][0]["message"]["content"]
    return str(result)


def extract_bpmn_few_shot(process_description, case_name, output_file=None, retries=2):
    # Warn if running on the few-shot example case itself
    if case_name == "case_6":
        print("WARNING: case_6 is the few-shot example. Results will be inflated — exclude from evaluation.")

    example_process = FEW_SHOT_EXAMPLE["process_description"]
    example_output  = json.dumps(FEW_SHOT_EXAMPLE["bpmn_json"], indent=2)

    prompt = f"""Extract a structured BPMN model from the process description below.

Here is an example of a process description and its correct BPMN JSON output:

--- EXAMPLE PROCESS ---
{example_process}

--- EXAMPLE OUTPUT ---
{example_output}

--- NOW EXTRACT ---
Process description: {process_description}

Apply the same structure to the process description above. Rules:
- Use concrete names from the description. Do not use placeholders like "...", "None", or empty strings.
- Every task, event, and gateway must reference a valid participant id declared in participants.
- sequence_flows must stay within a single pool. Cross-pool communication goes in message_flows.
- Every process must have at least one startEvent and one endEvent.
- Output exactly these top-level keys: participants, tasks, events, gateways, sequence_flows.
- Do NOT use a 'nodes' array. Tasks, events, and gateways are always separate arrays.
- Every gateway must have gatewayDirection of either "diverging" or "converging".
- Every event must have an eventDefinition."""

    try:
        model = load_model()
    except RuntimeError as e:
        print(e)
        return None

    print(f"Initial prompt:\n{prompt}\n")
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


if __name__ == "__main__":
    case_name = "case_7"

    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    CASES_DIR    = os.path.join(PROJECT_ROOT, "cases")
    OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(CASES_DIR, f"{case_name}.txt")
    out_file   = os.path.join(OUTPUT_DIR, f"{case_name}_few_shot_bpmn.json")

    print("--- Few-Shot Pipeline Started ---")
    print(f"Project Root is dit few shot?: {PROJECT_ROOT}")
    print(f"Reading from: {input_path}")
    print(f"Saving to:    {out_file}")
    print("---------------------------------")

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError

        with open(input_path, "r", encoding="utf-8") as f:
            process_text = f.read()

        print(f"🤖 LLM is analyzing '{case_name}' (few-shot)...")
        bpmn_json = extract_bpmn_few_shot(process_text, case_name=case_name, output_file=out_file, retries=2)

        if bpmn_json:
            print(f"Success! View the output in the sidebar under: outputs/{case_name}_few_shot_bpmn.json")
        else:
            print("Model extraction failed. Check the logs above.")

    except FileNotFoundError:
        print(f"Error: '{case_name}.txt' not found in {CASES_DIR}")