"""
FastAPI application for the Traffic Control OpenEnv environment.

Uses openenv-core's create_fastapi_app() for spec-compliant endpoints.
"""

from openenv.core.env_server import create_fastapi_app
from fastapi.responses import HTMLResponse

from ..models import TrafficAction, TrafficObservation
from .environment import TrafficEnvironment

# create_fastapi_app expects a CALLABLE (class), not an instance.
# It creates a new TrafficEnvironment for each request.
app = create_fastapi_app(
    env=TrafficEnvironment,
    action_cls=TrafficAction,
    observation_cls=TrafficObservation,
)


@app.get("/", response_class=HTMLResponse)
async def landing_page():
    return """<!DOCTYPE html>
<html><head><title>Traffic Control OpenEnv</title>
<style>
body{font-family:-apple-system,sans-serif;background:#0f172a;color:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
.c{background:#1e293b;padding:2.5rem;border-radius:1rem;max-width:580px;text-align:center;border:1px solid #334155}
.s{color:#22c55e;font-weight:bold}
.b{background:#3b82f6;padding:.2rem .6rem;border-radius:.4rem;font-size:.8rem;margin:.15rem;display:inline-block}
p{color:#94a3b8;line-height:1.6}
.e{text-align:left;background:#020617;padding:1rem;border-radius:.5rem;font-family:monospace;font-size:.85rem;margin-top:1rem;color:#60a5fa}
a{color:#60a5fa}
</style></head><body><div class="c">
<h1>🚦 Traffic Control OpenEnv</h1>
<p>Status: <span class="s">● ONLINE</span></p>
<p>Autonomous Traffic Signal Control environment for reinforcement learning agents.
Handles congestion management, emergency vehicle routing, and multi-intersection coordination.</p>
<div style="margin:1rem 0">
<span class="b">Easy: Single Intersection</span>
<span class="b">Medium: Corridor + EV</span>
<span class="b">Hard: Grid Incident</span>
</div>
<div class="e">
POST /reset &nbsp;→ Start new episode<br>
POST /step &nbsp; → Send action<br>
GET &nbsp;/state &nbsp;→ Current state<br>
GET &nbsp;/health → Health check
</div>
<p style="margin-top:1.5rem"><small>Built with <b>OpenEnv-core</b> | 
<a href="https://huggingface.co/spaces/Tarunts19/traffic-control-env">View on HF</a></small></p>
</div></body></html>"""
