"""Built-in figure renderers (import for registration side effects)."""

from __future__ import annotations

from surf_rag.viz.registry import register
from surf_rag.viz.renderers.router_pred_vs_oracle import render_router_pred_vs_oracle
from surf_rag.viz.renderers.router_pred_vs_oracle_intervals import (
    render_router_pred_vs_oracle_intervals,
)

register("router_pred_vs_oracle", render_router_pred_vs_oracle)
register("router_pred_vs_oracle_intervals", render_router_pred_vs_oracle_intervals)
