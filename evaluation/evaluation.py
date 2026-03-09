import json
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT     = os.path.dirname(SCRIPT_DIR)
GROUND_TRUTH_DIR = os.path.join(PROJECT_ROOT, "few_shot_cases")
OUTPUT_DIR       = os.path.join(PROJECT_ROOT, "pipelines", "direct_extraction_pipeline", "outputs")

# ── Cases and complexity strata ───────────────────────────────────────────────

COMPLEXITY = {
    #todo!!
}

APPROACHES = {
    "zero_shot":  "_zero_shot_bpmn.json",
    "few_shot":   "_few_shot_bpmn.json",
    "fine_tuned": "_fine_tuned_bpmn.json",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalise(name):
    return str(name).strip().lower()


def node_names(obj):
    nodes = (
        obj.get("tasks",    []) +
        obj.get("events",   []) +
        obj.get("gateways", [])
    )
    return {normalise(n["name"]) for n in nodes if isinstance(n, dict) and n.get("name")}


def task_names(obj):
    return {normalise(n["name"]) for n in obj.get("tasks", [])
            if isinstance(n, dict) and n.get("name")}


def event_names(obj):
    return {normalise(n["name"]) for n in obj.get("events", [])
            if isinstance(n, dict) and n.get("name")}


def gateway_names(obj):
    return {normalise(n["name"]) for n in obj.get("gateways", [])
            if isinstance(n, dict) and n.get("name")}


def id_to_name_map(obj):
    nodes = (
        obj.get("tasks",    []) +
        obj.get("events",   []) +
        obj.get("gateways", [])
    )
    return {
        n["id"]: normalise(n["name"])
        for n in nodes
        if isinstance(n, dict) and n.get("id") and n.get("name")
    }


def flow_pairs(obj):
    mapping = id_to_name_map(obj)
    pairs = set()
    for s in obj.get("sequence_flows", []):
        if not isinstance(s, dict):
            continue
        src = mapping.get(s.get("from"))
        tgt = mapping.get(s.get("to"))
        if src and tgt:
            pairs.add((src, tgt))
    return pairs


def message_flow_pairs(obj):
    mapping = id_to_name_map(obj)
    part_names = {p["id"]: normalise(p["name"])
                  for p in obj.get("participants", [])
                  if isinstance(p, dict) and p.get("id")}
    all_names = {**mapping, **part_names}
    pairs = set()
    for mf in obj.get("message_flows", []):
        if not isinstance(mf, dict):
            continue
        src = all_names.get(mf.get("from"))
        tgt = all_names.get(mf.get("to"))
        if src and tgt:
            pairs.add((src, tgt))
    return pairs


def participant_names(obj):
    return {
        normalise(p["name"])
        for p in obj.get("participants", [])
        if isinstance(p, dict) and p.get("name")
    }


def gateway_type_map(obj):
    return {
        normalise(g["name"]): g.get("type", "")
        for g in obj.get("gateways", [])
        if isinstance(g, dict) and g.get("name")
    }


def f1(predicted_set, ground_truth_set):
    if not ground_truth_set:
        return 0.0, 0.0, 0.0
    if not predicted_set:
        return 0.0, 0.0, 0.0
    tp        = len(predicted_set & ground_truth_set)
    precision = tp / len(predicted_set)
    recall    = tp / len(ground_truth_set)
    f1_score  = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return round(precision, 3), round(recall, 3), round(f1_score, 3)


# ── Per-case metric computation ───────────────────────────────────────────────

def evaluate_case(predicted, ground_truth):
    pred_nodes = node_names(predicted)
    gt_nodes   = node_names(ground_truth)
    node_p, node_r, node_f1 = f1(pred_nodes, gt_nodes)

    pred_flows = flow_pairs(predicted)
    gt_flows   = flow_pairs(ground_truth)
    flow_p, flow_r, flow_f1 = f1(pred_flows, gt_flows)

    pred_mflows = message_flow_pairs(predicted)
    gt_mflows   = message_flow_pairs(ground_truth)
    mflow_p, mflow_r, mflow_f1 = f1(pred_mflows, gt_mflows)

    pred_parts = participant_names(predicted)
    gt_parts   = participant_names(ground_truth)
    part_recall = (
        len(pred_parts & gt_parts) / len(gt_parts)
        if gt_parts else 0.0
    )

    pred_gw_types = gateway_type_map(predicted)
    gt_gw_types   = gateway_type_map(ground_truth)
    matched_gw    = set(pred_gw_types.keys()) & set(gt_gw_types.keys())
    if matched_gw:
        correct_types = sum(
            1 for name in matched_gw
            if pred_gw_types[name] == gt_gw_types[name]
        )
        gw_type_acc = round(correct_types / len(matched_gw), 3)
    else:
        gw_type_acc = None

    return {
        "node_precision":      node_p,
        "node_recall":         node_r,
        "node_f1":             node_f1,
        "flow_precision":      flow_p,
        "flow_recall":         flow_r,
        "flow_f1":             flow_f1,
        "mflow_precision":     mflow_p,
        "mflow_recall":        mflow_r,
        "mflow_f1":            mflow_f1,
        "participant_recall":  round(part_recall, 3),
        "gateway_type_acc":    gw_type_acc,
        "_pred_tasks":         task_names(predicted),
        "_gt_tasks":           task_names(ground_truth),
        "_pred_events":        event_names(predicted),
        "_gt_events":          event_names(ground_truth),
        "_pred_gateways":      gateway_names(predicted),
        "_gt_gateways":        gateway_names(ground_truth),
        "_pred_flows":         pred_flows,
        "_gt_flows":           gt_flows,
        "_pred_mflows":        pred_mflows,
        "_gt_mflows":          gt_mflows,
        "_pred_parts":         pred_parts,
        "_gt_parts":           gt_parts,
        "_pred_gw_types":      pred_gw_types,
        "_gt_gw_types":        gt_gw_types,
    }


# ── Printing helpers ──────────────────────────────────────────────────────────

def print_separator(char="─", width=72):
    print(char * width)


def fmt_set(s):
    return ", ".join(f"'{x}'" for x in sorted(s)) if s else "—"


def print_case_result(case_name, approach, metrics):
    print(f"\n  ┌─ [{case_name}]  approach: {approach}")

    # Scores
    print(f"  │  Nodes        P:{metrics['node_precision']:.3f}  "
          f"R:{metrics['node_recall']:.3f}  F1:{metrics['node_f1']:.3f}")
    print(f"  │  Seq flows    P:{metrics['flow_precision']:.3f}  "
          f"R:{metrics['flow_recall']:.3f}  F1:{metrics['flow_f1']:.3f}")
    print(f"  │  Msg flows    P:{metrics['mflow_precision']:.3f}  "
          f"R:{metrics['mflow_recall']:.3f}  F1:{metrics['mflow_f1']:.3f}")
    print(f"  │  Participants recall: {metrics['participant_recall']:.3f}")
    gw = metrics["gateway_type_acc"]
    print(f"  │  Gateway type accuracy: {'N/A' if gw is None else f'{gw:.3f}'}")

    # Participants
    missing_parts = metrics["_gt_parts"]   - metrics["_pred_parts"]
    extra_parts   = metrics["_pred_parts"] - metrics["_gt_parts"]
    print(f"  │")
    print(f"  │  PARTICIPANTS")
    print(f"  │    Expected : {fmt_set(metrics['_gt_parts'])}")
    print(f"  │    Got      : {fmt_set(metrics['_pred_parts'])}")
    if missing_parts:
        print(f"  │    ✗ Missing: {fmt_set(missing_parts)}")
    if extra_parts:
        print(f"  │    + Extra  : {fmt_set(extra_parts)}")
    if not missing_parts and not extra_parts:
        print(f"  │    ✓ All participants matched")

    # Tasks
    missing_tasks = metrics["_gt_tasks"]   - metrics["_pred_tasks"]
    extra_tasks   = metrics["_pred_tasks"] - metrics["_gt_tasks"]
    print(f"  │")
    print(f"  │  TASKS")
    if missing_tasks:
        print(f"  │    ✗ Missing: {fmt_set(missing_tasks)}")
    if extra_tasks:
        print(f"  │    + Extra  : {fmt_set(extra_tasks)}")
    if not missing_tasks and not extra_tasks:
        print(f"  │    ✓ All tasks matched")

    # Events
    missing_events = metrics["_gt_events"]   - metrics["_pred_events"]
    extra_events   = metrics["_pred_events"] - metrics["_gt_events"]
    print(f"  │")
    print(f"  │  EVENTS")
    if missing_events:
        print(f"  │    ✗ Missing: {fmt_set(missing_events)}")
    if extra_events:
        print(f"  │    + Extra  : {fmt_set(extra_events)}")
    if not missing_events and not extra_events:
        print(f"  │    ✓ All events matched")

    # Gateways
    missing_gw = metrics["_gt_gateways"]   - metrics["_pred_gateways"]
    extra_gw   = metrics["_pred_gateways"] - metrics["_gt_gateways"]
    print(f"  │")
    print(f"  │  GATEWAYS")
    if missing_gw:
        print(f"  │    ✗ Missing: {fmt_set(missing_gw)}")
    if extra_gw:
        print(f"  │    + Extra  : {fmt_set(extra_gw)}")
    matched_gw_names = (set(metrics["_pred_gw_types"].keys()) &
                        set(metrics["_gt_gw_types"].keys()))
    type_mismatches = [
        f"'{name}': expected {metrics['_gt_gw_types'][name]}, "
        f"got {metrics['_pred_gw_types'][name]}"
        for name in matched_gw_names
        if metrics["_pred_gw_types"][name] != metrics["_gt_gw_types"][name]
    ]
    for mismatch in type_mismatches:
        print(f"  │    ✗ Wrong type: {mismatch}")
    if not missing_gw and not extra_gw and not type_mismatches:
        print(f"  │    ✓ All gateways matched with correct types")

    # Sequence flows
    missing_flows = metrics["_gt_flows"]   - metrics["_pred_flows"]
    extra_flows   = metrics["_pred_flows"] - metrics["_gt_flows"]
    print(f"  │")
    print(f"  │  SEQUENCE FLOWS")
    if missing_flows:
        print(f"  │    ✗ Missing:")
        for src, tgt in sorted(missing_flows):
            print(f"  │        '{src}' → '{tgt}'")
    if extra_flows:
        print(f"  │    + Extra:")
        for src, tgt in sorted(extra_flows):
            print(f"  │        '{src}' → '{tgt}'")
    if not missing_flows and not extra_flows:
        print(f"  │    ✓ All sequence flows matched")

    # Message flows
    missing_mflows = metrics["_gt_mflows"]   - metrics["_pred_mflows"]
    extra_mflows   = metrics["_pred_mflows"] - metrics["_gt_mflows"]
    print(f"  │")
    print(f"  │  MESSAGE FLOWS")
    if missing_mflows:
        print(f"  │    ✗ Missing:")
        for src, tgt in sorted(missing_mflows):
            print(f"  │        '{src}' → '{tgt}'")
    if extra_mflows:
        print(f"  │    + Extra:")
        for src, tgt in sorted(extra_mflows):
            print(f"  │        '{src}' → '{tgt}'")
    if not missing_mflows and not extra_mflows:
        print(f"  │    ✓ All message flows matched")

    print(f"  └{'─' * 60}")


def avg(values):
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 3) if valid else None


def print_summary_table(all_results):
    print_separator("═")
    print("SUMMARY — Average metrics across all evaluated cases")
    print_separator("═")
    header = (f"{'Approach':<14} {'Node F1':>8} {'Flow F1':>8} "
              f"{'MFlow F1':>9} {'Part Rec':>9} {'GW Acc':>8}")
    print(header)
    print_separator()
    for approach, case_metrics in all_results.items():
        values = list(case_metrics.values())
        if not values:
            print(f"{approach:<14} {'—':>8} {'—':>8} {'—':>9} {'—':>9} {'—':>8}")
            continue
        print(f"{approach:<14} "
              f"{avg([m['node_f1']            for m in values]):>8} "
              f"{avg([m['flow_f1']            for m in values]):>8} "
              f"{avg([m['mflow_f1']           for m in values]):>9} "
              f"{avg([m['participant_recall'] for m in values]):>9} "
              f"{str(avg([m['gateway_type_acc'] for m in values])):>8}")
    print_separator()


def print_stratified_table(all_results):
    print_separator("═")
    print("TIER 3 — Complexity-Stratified Results")
    print_separator("═")
    levels = ["simple", "medium", "complex"]
    for approach, case_metrics in all_results.items():
        print(f"\n  Approach: {approach}")
        print(f"  {'Level':<10} {'Cases':>6} {'Node F1':>9} {'Flow F1':>9}")
        print("  " + "─" * 40)
        for level in levels:
            level_cases = [
                metrics for case_name, metrics in case_metrics.items()
                if COMPLEXITY.get(case_name) == level
            ]
            if not level_cases:
                print(f"  {level:<10} {'0':>6} {'—':>9} {'—':>9}")
                continue
            n_f1 = avg([m["node_f1"] for m in level_cases])
            f_f1 = avg([m["flow_f1"] for m in level_cases])
            print(f"  {level:<10} {len(level_cases):>6} {str(n_f1):>9} {str(f_f1):>9}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── CONFIGURATION ─────────────────────────────────────────────────────────
    CASES_TO_EVALUATE = [
        "case_2",

    ]
    # ── END CONFIGURATION ──────────────────────────────────────────────────────

    print_separator("═")
    print("BPMN EXTRACTION — EVALUATION RESULTS")
    print(f"Ground-truth dir : {GROUND_TRUTH_DIR}")
    print(f"Outputs dir      : {OUTPUT_DIR}")
    print_separator("═")

    all_results = {approach: {} for approach in APPROACHES}

    for case_name in CASES_TO_EVALUATE:
        gt_path = os.path.join(GROUND_TRUTH_DIR, f"{case_name}.json")

        if not os.path.exists(gt_path):
            print(f"\n  [SKIP] Ground truth not found for {case_name}: {gt_path}")
            continue

        ground_truth = load_json(gt_path)

        print_separator("─")
        print(f"CASE: {case_name}  (complexity: {COMPLEXITY.get(case_name, 'unknown')})")
        print_separator("─")

        for approach, suffix in APPROACHES.items():
            pred_path = os.path.join(OUTPUT_DIR, f"{case_name}{suffix}")

            if not os.path.exists(pred_path):
                print(f"\n  [SKIP] No output found for {case_name} / {approach}")
                continue

            try:
                predicted = load_json(pred_path)
            except json.JSONDecodeError as e:
                print(f"\n  [ERROR] Could not parse {pred_path}: {e}")
                continue

            metrics = evaluate_case(predicted, ground_truth)
            all_results[approach][case_name] = metrics
            print_case_result(case_name, approach, metrics)

    print()
    print_summary_table(all_results)
    print()
    print_stratified_table(all_results)
    print()
    print_separator("═")
    print("Evaluation complete.")
    print_separator("═")