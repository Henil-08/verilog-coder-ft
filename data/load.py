"""Load OpenRTLSet from the Hugging Face Hub into a flat record schema."""

from pathlib import Path

import yaml
from datasets import Dataset, load_dataset

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize(row: dict) -> dict:
    return {
        "id": str(row["index"]),
        "ioheader": (row["ioheader"] or "").strip(),
        "code": (row["full_code"] or "").strip(),
        "description": (row["llm_response"] or "").strip(),
        "repo": row["Repo_url"] or "",
        "license": row["lic_name"] or "",
        "has_children": bool(row["children_index"]),
    }


def load_raw(cfg: dict) -> Dataset:
    """Return OpenRTLSet with columns: id, ioheader, code, description, repo, license, has_children."""
    ds = load_dataset(cfg["dataset"]["hf_id"], split=cfg["dataset"]["split"])
    return ds.map(_normalize, remove_columns=ds.column_names)
