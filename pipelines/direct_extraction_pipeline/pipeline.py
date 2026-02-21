try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except Exception:
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
            filename="llama-2-7b-chat.Q2_K.gguf",
            n_ctx=2048,
            n_gpu_layers=-1,
            verbose=False
        )
    return llm


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
        }
    },
    "required": ["tasks", "sequence_flows"]
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
        prompt = f"""Extract all tasks, actors, and sequence flows from this process description.

Process description: {process_description}

Output a JSON object with:
- "tasks": array of objects with "id", "name", and "actor"
- "sequence_flows": array of objects with "from" and "to" (task ids)

Use concrete task names and actors from the description. Do not use placeholders."""
    
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
        max_tokens=1024 # dit kan hoger als de business case gigantisch is en de json is afgesneden
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
            max_tokens=1024
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
    case_name = "case_2" 
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