import os
import sys
import json
import argparse
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict, deque

# ── Namespaces ────────────────────────────────────────────────────────────────
BPMN   = "http://www.omg.org/spec/BPMN/20100524/MODEL"
BPMNDI = "http://www.omg.org/spec/BPMN/20100524/DI"
DC     = "http://www.omg.org/spec/DD/20100524/DC"
DI     = "http://www.omg.org/spec/DD/20100524/DI"

def tag(ns, local):
    return f"{{{ns}}}{local}"

# ── Node visual sizes ─────────────────────────────────────────────────────────
NODE_W = {
    "startEvent": 36, "endEvent": 36,
    "intermediateCatchEvent": 36, "intermediateThrowEvent": 36,
    "exclusiveGateway": 50, "parallelGateway": 50,
    "inclusiveGateway": 50, "eventBasedGateway": 50,
}
NODE_H = NODE_W.copy()
DEFAULT_W, DEFAULT_H = 120, 80

# ── Layout constants ──────────────────────────────────────────────────────────
POOL_HEADER_W = 30       # vertical pool label strip width
LANE_HEADER_W = 30       # vertical lane label strip width
LANE_H        = 220      # ↑ was 160 — more vertical room per lane
POOL_H_BARE   = 250      # ↑ was 180 — taller bare pools
COL_W         = 260      # ↑ was 190 — wider columns = more horizontal spacing
START_X_PAD   = 120      # ↑ was 80  — more padding before first column
POOL_GAP      = 60       # ↑ was 40  — more space between pools
POOL_ORIGIN_X = 10       # left edge of all pools

EVENT_DEFS = {
    "message": "messageEventDefinition", "timer": "timerEventDefinition",
    "signal": "signalEventDefinition",   "error": "errorEventDefinition",
    "escalation": "escalationEventDefinition", "link": "linkEventDefinition",
    "conditional": "conditionalEventDefinition",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA NORMALISER
# ─────────────────────────────────────────────────────────────────────────────
# Supports three input formats:
#   1. New format:  pools[] + lanes[] (separate top-level arrays)
#   2. Old format:  participants[] with nested lanes[]
#   3. Internal:    actors[] (already normalised)
# ═══════════════════════════════════════════════════════════════════════════════

def normalise(json_data):
    data = json_data.copy()

    if "actors" in data:
        pass

    elif "participants" in data:
        data["actors"] = [
            {"id": p["id"], "name": p["name"],
             "type": p.get("type", "pool"), "lanes": p.get("lanes", [])}
            for p in data["participants"]
        ]

    elif "pools" in data:
        pool_lane_map = defaultdict(list)
        for lane in data.get("lanes", []):
            ref = lane.get("pool") or lane.get("poolRef") or lane.get("participant")
            if ref:
                pool_lane_map[ref].append({"id": lane["id"], "name": lane["name"]})
        data["actors"] = [
            {"id": p["id"], "name": p["name"], "type": "pool",
             "lanes": pool_lane_map.get(p["id"], [])}
            for p in data["pools"]
        ]
    else:
        data["actors"] = []

    pool_ids     = {a["id"] for a in data["actors"]}
    lane_to_pool = {ln["id"]: a["id"]
                    for a in data["actors"] for ln in a.get("lanes", [])}

    for node_list in ("tasks", "events", "gateways"):
        out = []
        for n in data.get(node_list, []):
            n       = n.copy()
            act_id  = n.get("actor") or n.get("participant") or n.get("pool")
            lane_id = n.get("lane")

            if act_id and act_id in pool_ids:
                n["actor"] = act_id
                if not (lane_id and lane_id in lane_to_pool):
                    n.pop("lane", None)
            elif lane_id:
                if lane_id in pool_ids:
                    n["actor"] = lane_id
                    n.pop("lane", None)
                elif lane_id in lane_to_pool:
                    n["actor"] = lane_to_pool[lane_id]
                    n["lane"]  = lane_id
                else:
                    n["actor"] = lane_id
            elif act_id:
                n["actor"] = act_id
            out.append(n)
        data[node_list] = out

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate(json_data):
    data   = normalise(json_data)
    errors, warnings = [], []

    pool_ids     = {a["id"] for a in data.get("actors", [])}
    lane_to_pool = {ln["id"]: a["id"]
                    for a in data.get("actors", []) for ln in a.get("lanes", [])}
    all_nodes    = _get_all_nodes(data)
    node_ids     = {n["id"] for n in all_nodes}

    for n in all_nodes:
        pid = n.get("actor")
        if not pid:
            errors.append(f"  [NODE '{n['id']}'] missing pool assignment.")
        elif pid not in pool_ids:
            errors.append(f"  [NODE '{n['id']}'] actor='{pid}' not declared.")
        lid = n.get("lane")
        if lid:
            if lid not in lane_to_pool:
                errors.append(f"  [NODE '{n['id']}'] lane='{lid}' not declared.")
            elif lane_to_pool[lid] != pid:
                errors.append(f"  [NODE '{n['id']}'] lane/pool mismatch.")

    node_pool = {n["id"]: n.get("actor") for n in all_nodes}

    for f in data.get("sequence_flows", []):
        for side in ("from", "to"):
            if f.get(side) not in node_ids:
                errors.append(f"  [SEQ FLOW {f.get('id')}] '{side}' "
                              f"node '{f.get(side)}' missing.")
        sp, tp = node_pool.get(f["from"]), node_pool.get(f["to"])
        if sp and tp and sp != tp:
            warnings.append(f"  [SEQ FLOW {f.get('id')}] crosses pools "
                            f"→ will become message flow.")

    for mf in data.get("message_flows", []):
        for side in ("from", "to"):
            if mf.get(side) not in node_ids:
                errors.append(f"  [MSG FLOW {mf.get('id')}] '{side}' "
                              f"node '{mf.get(side)}' missing.")

    if warnings:
        print("WARNINGS:\n" + "\n".join(warnings))
    if errors:
        print("ERRORS:\n" + "\n".join(errors))
        return False
    print("  Validation passed.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_all_nodes(data):
    nodes = []
    for n in data.get("events",   []): nodes.append({**n, "_cat": "event"})
    for n in data.get("tasks",    []): nodes.append({**n, "_cat": "task"})
    for n in data.get("gateways", []): nodes.append({**n, "_cat": "gateway"})
    return nodes

def _build_actor_node_map(data):
    m = {a["id"]: [] for a in data.get("actors", [])}
    for n in _get_all_nodes(data):
        if n.get("actor") in m:
            m[n["actor"]].append(n)
    return m

def _detect_cross_pool(data):
    np = {n["id"]: n.get("actor") for n in _get_all_nodes(data)}
    seq, promoted = [], []
    for f in data.get("sequence_flows", []):
        if np.get(f["from"]) != np.get(f["to"]):
            promoted.append(f)
        else:
            seq.append(f)
    return seq, promoted


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _find_back_edges(node_ids, seq_flows):
    """DFS colouring to find back-edges (edges that create cycles)."""
    id_set = set(node_ids)
    local  = [f for f in seq_flows if f["from"] in id_set and f["to"] in id_set]
    succ   = defaultdict(list)
    for f in local:
        succ[f["from"]].append(f["to"])

    WHITE, GRAY, BLACK = 0, 1, 2
    color      = {n: WHITE for n in node_ids}
    back_edges = set()

    sys.setrecursionlimit(10000)

    def dfs(u):
        color[u] = GRAY
        for v in succ[u]:
            if color[v] == GRAY:
                back_edges.add((u, v))
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK

    for n in node_ids:
        if color[n] == WHITE:
            dfs(n)

    return back_edges


def _assign_columns(node_ids, seq_flows):
    """
    Longest-path column assignment on the DAG formed by removing back-edges.
    Returns { node_id: col_index }.
    """
    if not node_ids:
        return {}

    id_set     = set(node_ids)
    back_edges = _find_back_edges(node_ids, seq_flows)

    forward = [f for f in seq_flows
               if f["from"] in id_set and f["to"] in id_set
               and (f["from"], f["to"]) not in back_edges]

    succ   = defaultdict(list)
    in_deg = defaultdict(int)
    for f in forward:
        succ[f["from"]].append(f["to"])
        in_deg[f["to"]] += 1

    col   = {n: 0 for n in node_ids}
    queue = deque(n for n in node_ids if in_deg[n] == 0)

    while queue:
        n = queue.popleft()
        for s in succ[n]:
            if col[n] + 1 > col[s]:
                col[s] = col[n] + 1
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    return col


def _hint_columns_from_message_flows(actor, nodes, col_map, all_message_flows,
                                     other_pool_cols):
    """
    For isolated pools (all nodes at col 0), order nodes by the column
    of the remote node they communicate with via message flows.
    """
    id_set    = {n["id"] for n in nodes}
    all_col_0 = all(col_map.get(nid, 0) == 0 for nid in id_set)
    if not all_col_0 or not other_pool_cols:
        return

    remote_col = {}
    for mf in all_message_flows:
        src, tgt = mf["from"], mf["to"]
        if src in id_set and tgt in other_pool_cols:
            remote_col[src] = max(remote_col.get(src, 0), other_pool_cols[tgt])
        if tgt in id_set and src in other_pool_cols:
            remote_col[tgt] = max(remote_col.get(tgt, 0), other_pool_cols[src])

    if not remote_col:
        return

    for nid in id_set:
        if nid in remote_col:
            col_map[nid] = remote_col[nid]


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _node_wh(n_type):
    return NODE_W.get(n_type, DEFAULT_W), NODE_H.get(n_type, DEFAULT_H)


def _place_cell(nodes, cx, band_y, band_h, node_layout):
    """
    Stack nodes vertically inside a lane band, centred on column cx.
    When multiple nodes share a column+lane cell they are distributed
    evenly with guaranteed minimum gap between them.
    """
    if not nodes:
        return

    n        = len(nodes)
    MIN_GAP  = 30   # minimum px gap between node edges
    sizes    = [_node_wh(node["type"]) for node in nodes]
    total_h  = sum(h for _, h in sizes) + MIN_GAP * (n - 1)

    # Centre the whole stack vertically inside the band
    start_y  = band_y + (band_h - total_h) / 2
    cursor_y = start_y

    for i, node in enumerate(nodes):
        w, h = sizes[i]
        node_layout[node["id"]] = {
            "x": int(cx - w / 2),
            "y": int(cursor_y),
            "w": w, "h": h,
        }
        cursor_y += h + MIN_GAP


def _compute_layout(data, seq_flows, message_flows):
    actors         = data.get("actors", [])
    actor_node_map = _build_actor_node_map(data)

    pool_layout  = {}
    lane_layout  = {}
    node_layout  = {}
    pool_col_map = {}
    cur_y        = 20

    # ── Pass 1: assign columns per pool ──────────────────────────────────────
    for actor in actors:
        pid   = actor["id"]
        nodes = actor_node_map.get(pid, [])
        if not nodes:
            pool_col_map[pid] = {}
            continue
        ids = [n["id"] for n in nodes]
        pool_col_map[pid] = _assign_columns(ids, seq_flows)

    # ── Pass 2: apply message-flow hints to isolated pools ────────────────────
    all_remote_cols = {}
    for pid, cm in pool_col_map.items():
        all_remote_cols.update(cm)

    for actor in actors:
        pid   = actor["id"]
        nodes = actor_node_map.get(pid, [])
        if not nodes:
            continue
        _hint_columns_from_message_flows(
            actor, nodes, pool_col_map[pid], message_flows, all_remote_cols)

    # ── Pass 3: compute pixel positions ──────────────────────────────────────
    for actor in actors:
        pid   = actor["id"]
        lanes = actor.get("lanes", [])
        nodes = actor_node_map.get(pid, [])
        cm    = pool_col_map[pid]

        if not nodes:
            pool_layout[pid] = {
                "x": POOL_ORIGIN_X, "y": cur_y,
                "w": 400,
                "h": len(lanes) * LANE_H if lanes else POOL_H_BARE,
            }
            cur_y += pool_layout[pid]["h"] + POOL_GAP
            continue

        n_cols    = max(cm.values(), default=0) + 1
        headers_w = POOL_HEADER_W + (LANE_HEADER_W if lanes else 0)

        # ── Dynamic lane height: grow lane if a column is overcrowded ─────────
        # Find max number of nodes sharing a single (lane, col) cell
        if lanes:
            cell_counts = defaultdict(int)
            for n in nodes:
                lid = n.get("lane", "__no_lane__")
                c   = cm.get(n["id"], 0)
                cell_counts[(lid, c)] += 1
            max_in_cell   = max(cell_counts.values(), default=1)
            MIN_GAP       = 30
            needed_h      = max_in_cell * DEFAULT_H + (max_in_cell - 1) * MIN_GAP + 60
            effective_lane_h = max(LANE_H, needed_h)
        else:
            effective_lane_h = LANE_H

        content_w = START_X_PAD + n_cols * COL_W + 60
        pool_w    = max(headers_w + content_w, 600)
        pool_h    = len(lanes) * effective_lane_h if lanes else POOL_H_BARE

        pool_layout[pid] = {
            "x": POOL_ORIGIN_X, "y": cur_y,
            "w": pool_w, "h": pool_h,
        }

        x_origin = POOL_ORIGIN_X + headers_w + START_X_PAD
        col_cx   = [x_origin + c * COL_W for c in range(n_cols)]

        if lanes:
            lane_y = cur_y
            for lane in lanes:
                lid = lane["id"]
                lane_layout[lid] = {
                    "x": POOL_ORIGIN_X + POOL_HEADER_W,
                    "y": lane_y,
                    "w": pool_w - POOL_HEADER_W,
                    "h": effective_lane_h,
                }
                lane_nodes = [n for n in nodes if n.get("lane") == lid]
                by_col     = defaultdict(list)
                for n in lane_nodes:
                    by_col[cm[n["id"]]].append(n)
                for c, cell in by_col.items():
                    _place_cell(cell, col_cx[c], lane_y, effective_lane_h, node_layout)
                lane_y += effective_lane_h

            # Unlaned nodes → centre in full pool height
            unlaned = [n for n in nodes if not n.get("lane")]
            by_col  = defaultdict(list)
            for n in unlaned:
                by_col[cm[n["id"]]].append(n)
            for c, cell in by_col.items():
                _place_cell(cell, col_cx[c], cur_y, pool_h, node_layout)

        else:
            by_col = defaultdict(list)
            for n in nodes:
                by_col[cm[n["id"]]].append(n)
            for c, cell in by_col.items():
                _place_cell(cell, col_cx[c], cur_y, pool_h, node_layout)

        cur_y += pool_h + POOL_GAP

    return pool_layout, lane_layout, node_layout


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

def _waypoints(src, tgt):
    """
    Forward edge  → exit right-centre, elbow if different y, enter left-centre.
    Back-edge     → arc over the top of both nodes.
    """
    sx = src["x"] + src["w"]
    sy = src["y"] + src["h"] // 2
    tx = tgt["x"]
    ty = tgt["y"] + tgt["h"] // 2

    if sx <= tx + 10:
        if abs(sy - ty) < 6:
            return [(sx, sy), (tx, ty)]
        mid_x = (sx + tx) // 2
        return [(sx, sy), (mid_x, sy), (mid_x, ty), (tx, ty)]
    else:
        top_y  = min(src["y"], tgt["y"]) - 50   # arc clears nodes with extra room
        src_cx = src["x"] + src["w"] // 2
        tgt_cx = tgt["x"] + tgt["w"] // 2
        return [
            (src_cx, src["y"]),
            (src_cx, top_y),
            (tgt_cx, top_y),
            (tgt_cx, tgt["y"]),
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# XML BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def create_bpmn_xml(json_data, output_path):
    data = normalise(json_data)

    ET.register_namespace("bpmn",   BPMN)
    ET.register_namespace("bpmndi", BPMNDI)
    ET.register_namespace("dc",     DC)
    ET.register_namespace("di",     DI)

    actors    = data.get("actors", [])
    seq_flows, promoted = _detect_cross_pool(data)

    message_flows = list(data.get("message_flows", [])) + [
        {**f, "name": f.get("condition", ""), "id": f"MsgFlow_p{i}"}
        for i, f in enumerate(promoted)
    ]

    has_collab = len(actors) > 1 or bool(message_flows)
    collab_id  = "Collaboration_1"

    # ── Root ──────────────────────────────────────────────────────────────────
    root = ET.Element(tag(BPMN, "definitions"), {
        "id": "Definitions_1",
        "targetNamespace": "http://bpmn.io/schema/bpmn",
        "exporter": "BPMN Pipeline", "exporterVersion": "2.0",
    })

    # ── Collaboration ─────────────────────────────────────────────────────────
    if has_collab:
        collab = ET.SubElement(root, tag(BPMN, "collaboration"), id=collab_id)
        for a in actors:
            ET.SubElement(collab, tag(BPMN, "participant"), {
                "id": f"actor_{a['id']}", "name": a["name"],
                "processRef": f"Process_{a['id']}",
            })
        for i, mf in enumerate(message_flows):
            ET.SubElement(collab, tag(BPMN, "messageFlow"), {
                "id":        mf.get("id", f"MsgFlow_{i}"),
                "name":      mf.get("name", ""),
                "sourceRef": mf["from"],
                "targetRef": mf["to"],
            })

    # ── Processes ─────────────────────────────────────────────────────────────
    actor_node_map = _build_actor_node_map(data)
    node_type_map  = {}

    for actor in actors:
        pid   = actor["id"]
        lanes = actor.get("lanes", [])
        proc  = ET.SubElement(root, tag(BPMN, "process"), {
            "id": f"Process_{pid}", "name": actor["name"], "isExecutable": "false",
        })

        if lanes:
            ls       = ET.SubElement(proc, tag(BPMN, "laneSet"), id=f"LaneSet_{pid}")
            ln_nodes = defaultdict(list)
            for n in actor_node_map.get(pid, []):
                if n.get("lane"):
                    ln_nodes[n["lane"]].append(n["id"])
            for lane in lanes:
                le = ET.SubElement(ls, tag(BPMN, "lane"),
                                   id=lane["id"], name=lane["name"])
                for nid in ln_nodes.get(lane["id"], []):
                    ET.SubElement(le, tag(BPMN, "flowNodeRef")).text = nid

        pool_node_ids = {n["id"] for n in actor_node_map.get(pid, [])}

        for n in actor_node_map.get(pid, []):
            node_type_map[n["id"]] = n["type"]
            elem = ET.SubElement(proc, tag(BPMN, n["type"]),
                                 id=n["id"], name=n["name"])
            if n["_cat"] == "event":
                ev = n.get("eventDefinition", "none")
                if ev and ev != "none" and ev in EVENT_DEFS:
                    ET.SubElement(elem, tag(BPMN, EVENT_DEFS[ev]),
                                  id=f"{n['id']}_def")
            if n["_cat"] == "gateway":
                elem.set("gatewayDirection",
                         n.get("gatewayDirection", "diverging"))

        for i, f in enumerate(seq_flows):
            if f["from"] not in pool_node_ids:
                continue
            attrs = {"id": f.get("id", f"Flow_{i}"),
                     "sourceRef": f["from"], "targetRef": f["to"]}
            if f.get("condition"):
                attrs["name"] = f["condition"]
            sf = ET.SubElement(proc, tag(BPMN, "sequenceFlow"), attrs)
            if f.get("condition"):
                ET.SubElement(sf, tag(BPMN, "conditionExpression")).text = \
                    f["condition"]

    # ── Layout ────────────────────────────────────────────────────────────────
    pool_layout, lane_layout, node_layout = \
        _compute_layout(data, seq_flows, message_flows)

    # ── BPMNDI ────────────────────────────────────────────────────────────────
    diagram   = ET.SubElement(root, tag(BPMNDI, "BPMNDiagram"), id="BPMNDiagram_1")
    plane_ref = collab_id if has_collab else \
                (f"Process_{actors[0]['id']}" if actors else "Process_1")
    plane = ET.SubElement(diagram, tag(BPMNDI, "BPMNPlane"),
                          id="BPMNPlane_1", bpmnElement=plane_ref)

    # Pool shapes
    if has_collab:
        for actor in actors:
            pl = pool_layout.get(actor["id"])
            if not pl:
                continue
            sh = ET.SubElement(plane, tag(BPMNDI, "BPMNShape"), {
                "id": f"actor_{actor['id']}_di",
                "bpmnElement": f"actor_{actor['id']}",
                "isHorizontal": "true",
            })
            ET.SubElement(sh, tag(DC, "Bounds"),
                          x=str(pl["x"]), y=str(pl["y"]),
                          width=str(pl["w"]), height=str(pl["h"]))

    # Lane shapes
    for actor in actors:
        for lane in actor.get("lanes", []):
            ll = lane_layout.get(lane["id"])
            if not ll:
                continue
            sh = ET.SubElement(plane, tag(BPMNDI, "BPMNShape"), {
                "id": f"{lane['id']}_di",
                "bpmnElement": lane["id"],
                "isHorizontal": "true",
            })
            ET.SubElement(sh, tag(DC, "Bounds"),
                          x=str(ll["x"]), y=str(ll["y"]),
                          width=str(ll["w"]), height=str(ll["h"]))

    # Node shapes
    for nid, layout in node_layout.items():
        n_type = node_type_map.get(nid, "")
        attrs  = {"id": f"{nid}_di", "bpmnElement": nid}
        if "Gateway" in n_type:
            attrs["isMarkerVisible"] = "true"
        sh = ET.SubElement(plane, tag(BPMNDI, "BPMNShape"), attrs)
        ET.SubElement(sh, tag(DC, "Bounds"),
                      x=str(layout["x"]), y=str(layout["y"]),
                      width=str(layout["w"]), height=str(layout["h"]))

    # Sequence flow edges
    for i, f in enumerate(seq_flows):
        src, tgt = f["from"], f["to"]
        if src not in node_layout or tgt not in node_layout:
            continue
        wps  = _waypoints(node_layout[src], node_layout[tgt])
        edge = ET.SubElement(plane, tag(BPMNDI, "BPMNEdge"), {
            "id": f"{f.get('id', f'Flow_{i}')}_di",
            "bpmnElement": f.get("id", f"Flow_{i}"),
        })
        for wx, wy in wps:
            ET.SubElement(edge, tag(DI, "waypoint"), x=str(wx), y=str(wy))

    # Message flow edges
    node_pool_map = {n["id"]: n.get("actor") for n in _get_all_nodes(data)}
    for i, mf in enumerate(message_flows):
        src, tgt = mf["from"], mf["to"]
        if src not in node_layout or tgt not in node_layout:
            continue
        sl  = node_layout[src]
        tl  = node_layout[tgt]
        sp  = node_pool_map.get(src)
        tp  = node_pool_map.get(tgt)
        spy = pool_layout[sp]["y"] if sp in pool_layout else 0
        tpy = pool_layout[tp]["y"] if tp in pool_layout else 0
        if spy < tpy:
            wx1, wy1 = sl["x"] + sl["w"] // 2, sl["y"] + sl["h"]
            wx2, wy2 = tl["x"] + tl["w"] // 2, tl["y"]
        else:
            wx1, wy1 = sl["x"] + sl["w"] // 2, sl["y"]
            wx2, wy2 = tl["x"] + tl["w"] // 2, tl["y"] + tl["h"]
        edge = ET.SubElement(plane, tag(BPMNDI, "BPMNEdge"), {
            "id": f"{mf.get('id', f'MsgFlow_{i}')}_di",
            "bpmnElement": mf.get("id", f"MsgFlow_{i}"),
        })
        ET.SubElement(edge, tag(DI, "waypoint"), x=str(wx1), y=str(wy1))
        ET.SubElement(edge, tag(DI, "waypoint"), x=str(wx2), y=str(wy2))

    # ── Serialise ─────────────────────────────────────────────────────────────
    raw    = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    pretty = minidom.parseString(raw).toprettyxml(indent="  ")
    lines  = [l for l in pretty.splitlines() if not l.startswith("<?xml")]
    final  = '<?xml version="1.0" encoding="UTF-8"?>\n' + "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final)
    print(f"  Written → {output_path}")


def _resolve_input_json(*, case_name, pipeline_name, prompting_strategy, input_json_override=None):
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    JSON_SOURCE_DIR = os.path.join(PROJECT_ROOT, "pipelines", pipeline_name, "outputs")

    if input_json_override:
        if not os.path.exists(input_json_override):
            raise FileNotFoundError("input not found")
        return input_json_override

    input_json = os.path.join(
        JSON_SOURCE_DIR,
        f"{case_name}_{prompting_strategy}_bpmn.json",
    )

    if not os.path.exists(input_json):
        raise FileNotFoundError("input not found")

    return input_json


def generate_case_bpmn_xml(
    *,
    case_name="case_29",
    pipeline_name="direct_extraction_pipeline",
    prompting_strategy="",
    input_json=None,
    output_bpmn=None,
):
    SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
    XML_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

    if output_bpmn is None:
        output_bpmn = os.path.join(XML_OUTPUT_DIR, f"{case_name}_{prompting_strategy}.bpmn")

    try:
        input_json_path = _resolve_input_json(
            case_name=case_name,
            pipeline_name=pipeline_name,
            prompting_strategy=prompting_strategy,
            input_json_override=input_json,
        )

        with open(input_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        print("--- BPMN XML Generator ---")
        print(f"Input: {input_json_path}")
        create_bpmn_xml(raw, output_bpmn)
        print(f"Written output: {output_bpmn}")
        return output_bpmn
    except FileNotFoundError:
        print("input not found")
        return None


if __name__ == "__main__":
    # ── CONFIGURATION ─────────────────────────────────────────────────────────
    case_name          = "case_24"
    pipeline_name      = "direct_extraction_pipeline"
    prompting_strategy = "zero_shot_phi4"
    # ──────────────────────────────────────────────────────────────────────────

    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    JSON_SOURCE_DIR = os.path.join(PROJECT_ROOT, "pipelines", pipeline_name, "outputs")
    XML_OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

    input_json  = os.path.join(JSON_SOURCE_DIR, f"{case_name}_{prompting_strategy}.json")
    output_bpmn = os.path.join(XML_OUTPUT_DIR,  f"{case_name}_{prompting_strategy}.bpmn")

    print("--- BPMN XML Generator ---")
    try:
        if not os.path.exists(input_json):
            raise FileNotFoundError(f"Input not found: {input_json}")
        with open(input_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        print("Validating JSON...")
        create_bpmn_xml(raw, output_bpmn)
        print("Done — upload to https://bpmn.io to view")
    except Exception:
        import traceback
        traceback.print_exc()