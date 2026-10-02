"""JSON-schema validation of plan documents plus semantic checks the schema cannot express."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "plan.schema.json"


@lru_cache(maxsize=1)
def _validator():
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.Draft202012Validator(schema)


def validate(doc: dict) -> None:
    errs = sorted(_validator().iter_errors(doc), key=lambda e: list(e.path))
    if errs:
        msg = "\n".join(f"  {'/'.join(map(str, e.path))}: {e.message}" for e in errs[:20])
        raise ValueError(f"plan.json fails schema ({len(errs)} errors):\n{msg}")
    ids = set()
    for r in doc["rooms"]:
        ids.add(r["id"])
        ids.update(w["id"] for w in r["walls"])
    ids.update(o["id"] for o in doc["openings"])
    sids = {s["id"] for s in doc["surfaces"]}
    for o in doc["openings"]:
        for rid in o["rooms"]:
            if rid not in ids:
                raise ValueError(f"opening {o['id']} references unknown room {rid}")
    for coll in ("damage", "concealed_damage_flags", "scope"):
        for d in doc[coll]:
            if d["surface"] not in sids:
                raise ValueError(f"{coll} {d['id']} references unknown surface {d['surface']}")
    # every measurement interval must contain its value
    def walk(x):
        if isinstance(x, dict):
            if {"value", "ci95", "sigma"} <= x.keys():
                lo, hi = x["ci95"]
                if not (lo <= x["value"] <= hi):
                    raise ValueError(f"interval {x['ci95']} does not contain value {x['value']}")
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(doc)
