"""Damage regions, concealed-damage rules and scope line items (filled in by task 6)."""
from __future__ import annotations


def attach_damage(plan, ctx, log=print) -> None:
    plan.extra.setdefault("damage", [])
    plan.extra.setdefault("flags", [])
    plan.extra.setdefault("scope", [])
