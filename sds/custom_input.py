"""Custom floorplan input parser and validator for interactive user-defined layouts."""

from __future__ import annotations
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from .block import Block
from .constraint_graph import ConstraintGraph
from .sequence_pair import sequence_pair_to_constraint_graphs


# Default sample custom floorplan (8-block modern SoC)
DEFAULT_CUSTOM_BLOCKS = [
    {"Name": "CPU_Core", "Type": "Soft", "Area": 160.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.33, "Max_AR": 3.0},
    {"Name": "GPU_Core", "Type": "Soft", "Area": 220.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.5, "Max_AR": 2.0},
    {"Name": "SRAM_Cache", "Type": "Hard", "Area": 100.0, "Fixed_Width": 10.0, "Fixed_Height": 10.0, "Min_AR": 1.0, "Max_AR": 1.0},
    {"Name": "DRAM_Ctrl", "Type": "Hard", "Area": 80.0, "Fixed_Width": 8.0, "Fixed_Height": 10.0, "Min_AR": 1.0, "Max_AR": 1.0},
    {"Name": "DSP_Engine", "Type": "Soft", "Area": 90.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.4, "Max_AR": 2.5},
    {"Name": "Crypto_Unit", "Type": "Soft", "Area": 50.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.5, "Max_AR": 2.0},
    {"Name": "PCIe_Phy", "Type": "Hard", "Area": 72.0, "Fixed_Width": 12.0, "Fixed_Height": 6.0, "Min_AR": 1.0, "Max_AR": 1.0},
    {"Name": "Power_Mgmt", "Type": "Soft", "Area": 40.0, "Fixed_Width": 0.0, "Fixed_Height": 0.0, "Min_AR": 0.5, "Max_AR": 2.0},
]

DEFAULT_CUSTOM_NETS = (
    "CPU_Core, SRAM_Cache\n"
    "CPU_Core, GPU_Core, DRAM_Ctrl\n"
    "GPU_Core, SRAM_Cache\n"
    "DSP_Engine, DRAM_Ctrl\n"
    "Crypto_Unit, PCIe_Phy\n"
    "CPU_Core, Power_Mgmt\n"
    "PCIe_Phy, DRAM_Ctrl"
)


def parse_blocks_from_dataframe(df: pd.DataFrame) -> Tuple[Dict[int, Block], Dict[str, int], float]:
    """Parse Block objects from an editable Pandas DataFrame.

    Returns:
        (blocks_dict, name_to_id_map, total_area)
    """
    blocks: Dict[int, Block] = {}
    name_to_id: Dict[str, int] = {}
    total_area = 0.0

    for idx, row in df.iterrows():
        block_id = idx + 1
        name = str(row.get("Name", f"B{block_id}")).strip()
        if not name:
            name = f"B{block_id}"

        b_type = str(row.get("Type", "Soft")).strip().capitalize()
        is_hard = (b_type == "Hard")

        area = float(row.get("Area", 100.0))
        fixed_w = float(row.get("Fixed_Width", 0.0))
        fixed_h = float(row.get("Fixed_Height", 0.0))

        min_ar = float(row.get("Min_AR", 1.0 / 3.0))
        max_ar = float(row.get("Max_AR", 3.0))
        if min_ar <= 0:
            min_ar = 0.33
        if max_ar < min_ar:
            max_ar = max(min_ar * 1.5, 3.0)

        if is_hard:
            if fixed_w <= 0 or fixed_h <= 0:
                # If dimensions missing, derive square from area
                fixed_w = math.sqrt(max(area, 1.0))
                fixed_h = fixed_w
            area = fixed_w * fixed_h
            b = Block(
                id=block_id,
                name=name,
                is_hard=True,
                w_fixed=fixed_w,
                h_fixed=fixed_h,
            )
        else:
            if area <= 0:
                area = 100.0
            b = Block(
                id=block_id,
                name=name,
                area=area,
                is_hard=False,
                aspect_ratio_range=(min_ar, max_ar),
            )

        blocks[block_id] = b
        name_to_id[name] = block_id
        name_to_id[str(block_id)] = block_id
        total_area += area

    return blocks, name_to_id, total_area


def parse_netlist_text(net_text: str, name_to_id: Dict[str, int]) -> List[List[int]]:
    """Parse netlist from multi-line text (comma or space separated block names/IDs).

    Format examples:
        CPU, ALU, RAM
        1, 2, 3
        GPU - DRAM_Ctrl - Cache
    """
    nets: List[List[int]] = []
    if not net_text or not net_text.strip():
        return nets

    lines = net_text.strip().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Split on commas, dashes, semicolons or spaces
        tokens = [t.strip() for t in re.split(r"[,;\-\s]+", line) if t.strip()]
        resolved_net = []
        for token in tokens:
            if token in name_to_id:
                resolved_net.append(name_to_id[token])
            elif token.isdigit() and int(token) in name_to_id.values():
                resolved_net.append(int(token))
        if len(resolved_net) >= 2:
            nets.append(resolved_net)

    return nets


def parse_custom_json(json_str: str) -> Tuple[pd.DataFrame, str]:
    """Parse JSON specification into DataFrame and Netlist text."""
    data = json.loads(json_str)
    blocks_data = data.get("blocks", [])
    nets_data = data.get("nets", [])

    rows = []
    for b in blocks_data:
        rows.append({
            "Name": b.get("name", ""),
            "Type": "Hard" if b.get("is_hard", False) else "Soft",
            "Area": float(b.get("area", 100.0)),
            "Fixed_Width": float(b.get("w_fixed", 0.0)),
            "Fixed_Height": float(b.get("h_fixed", 0.0)),
            "Min_AR": float(b.get("aspect_ratio_range", [0.33, 3.0])[0]),
            "Max_AR": float(b.get("aspect_ratio_range", [0.33, 3.0])[1]),
        })
    df = pd.DataFrame(rows)

    net_lines = []
    for net in nets_data:
        if isinstance(net, list):
            net_lines.append(", ".join(str(item) for item in net))
    nets_text = "\n".join(net_lines)

    return df, nets_text


def export_custom_to_json(df: pd.DataFrame, nets_text: str, name_to_id: Dict[str, int]) -> str:
    """Export current custom blocks and netlist to JSON string."""
    blocks_list = []
    for idx, row in df.iterrows():
        is_hard = str(row.get("Type", "Soft")).strip().capitalize() == "Hard"
        b_dict = {
            "id": idx + 1,
            "name": str(row.get("Name", f"B{idx+1}")),
            "is_hard": is_hard,
            "area": float(row.get("Area", 100.0)),
        }
        if is_hard:
            b_dict["w_fixed"] = float(row.get("Fixed_Width", 10.0))
            b_dict["h_fixed"] = float(row.get("Fixed_Height", 10.0))
        else:
            b_dict["aspect_ratio_range"] = [
                float(row.get("Min_AR", 0.33)),
                float(row.get("Max_AR", 3.0)),
            ]
        blocks_list.append(b_dict)

    nets_list = []
    for line in nets_text.strip().splitlines():
        tokens = [t.strip() for t in re.split(r"[,;\-\s]+", line) if t.strip()]
        if len(tokens) >= 2:
            nets_list.append(tokens)

    data = {
        "blocks": blocks_list,
        "nets": nets_list,
    }
    return json.dumps(data, indent=2)

