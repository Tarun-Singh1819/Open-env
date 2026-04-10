---
title: Traffic Control OpenEnv
emoji: 🚦
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
---

# 🚦 Traffic Control OpenEnv Environment

> **Autonomous Traffic Signal Control with Inter-Agent Coordination Protocol**

An OpenEnv-compliant reinforcement learning environment where an LLM agent operates as a **Traffic Management Center**, controlling signal phases for a cluster of intersections while coordinating with neighboring heuristic-controlled agents through natural-language messages.

---

## Motivation

Traffic signal coordination is a genuine real-world problem:
- **400+ hours/year** are lost per commuter to congestion in major US cities
- Emergency vehicle delays caused by poorly-timed signals cost lives
- Multi-agent coordination (signals talking to each other) is an active research area

This environment lets you **train and evaluate LLM agents** on this task using the standard OpenEnv `reset()`/`step()`/`state()` API.

---

## Environment Description

### How It Works

The environment simulates a network of intersections with:
- **Cell-based road segments** (vehicles move cell-by-cell each 5-second timestep)
- **NEMA-compliant signal phases** (minimum green, yellow clearance, all-red)
- **Heuristic controllers** for non-agent intersections (fixed timing + emergency pre-emption)
- **Emergency vehicles** with priority routing through the network

### Hybrid Agent Architecture

```
┌─────────────┐     messages     ┌─────────────┐
│  LLM Agent  │ ◄──────────────► │  Heuristic   │
│  (I1, I2)   │   coordination   │  (I3, I4)    │
│  Your Code  │                  │  Fixed-Timing │
└─────────────┘                  └─────────────┘
```

The LLM controls a **subset** of intersections. The rest run on fixed-timing heuristics. The agent receives messages from neighbors and can send messages to influence their behavior.

---

## Action Space

```python
class TrafficAction(Action):
    commands: List[PhaseCommand]         # Signal phase changes
    messages: List[CoordinationMessage]  # Messages to neighbors
    reasoning: str                       # Agent's explanation
```

### Phase Commands
| Phase     | Green Directions |
|-----------|-----------------|
| NS_GREEN  | North, South    |
| NS_LEFT   | North, South (left turn) |
| EW_GREEN  | East, West      |
| EW_LEFT   | East, West (left turn) |

### Coordination Messages
| Field           | Type     | Description |
|-----------------|----------|-------------|
| from_node       | str      | Sender intersection ID |
| to_node         | str      | Target intersection ID |
| message_type    | enum     | traffic_alert, emergency_alert, congestion_warning, status_update |
| urgency         | enum     | routine, elevated, urgent, critical |
| content         | str      | Natural language situation description |
| suggested_action| str      | Optional recommended response |

---

## Observation Space

```python
class TrafficObservation(Observation):
    step_number: int
    time_seconds: int
    intersections: List[IntersectionObservation]  # All intersection states
    emergency_vehicles: List[EmergencyVehicleObs]  # Active EVs
    incoming_messages: List[CoordinationMessage]    # Messages from neighbors
    network_metrics: NetworkMetrics                 # Aggregate stats
    agent_cluster: List[str]                        # IDs you control
```

Each intersection provides:
- Current phase and elapsed time
- Queue lengths per approach direction
- Average/max wait times
- Throughput from last green phase

---

## Tasks

### 1. Single Intersection Rush Hour (Easy)
- **Network**: 1 intersection, no neighbors
- **Challenge**: Optimize NS vs EW green split with asymmetric demand
- **Max steps**: 60 (5 minutes simulated)
- **No coordination required**

### 2. Corridor Green Wave with Emergency (Medium)
- **Network**: 3 intersections in a line (you control 2, 1 is heuristic)
- **Challenge**: Create a "green wave" for smooth traffic flow, handle an emergency vehicle, coordinate with the heuristic neighbor
- **Max steps**: 120 (10 minutes simulated)
- **Coordination required**: Forward emergency alerts

### 3. Grid Network Incident Response (Hard)
- **Network**: 2×2 grid (you control 2, 2 are heuristic)
- **Challenge**: Handle road capacity reduction (incident), 2 competing emergency vehicles, demand shifts, bidirectional coordination
- **Max steps**: 180 (15 minutes simulated)
- **Full coordination required**: Multi-hop message chains

---

## Reward Function

The reward is computed per-step and provides **continuous signal** (not just sparse end-of-episode):

| Component | Weight | Signal |
|-----------|--------|--------|
| Traffic flow | ~40% | Throughput, queue reduction, wait time improvement |
| EV coordination | ~35% | Proactive phase changes, delay minimization |
| Message quality | ~15% | Relevant alerts sent, false alarm avoidance |
| Equity | ~10% | No single approach starved of green time |

---

## Grading (0.0 – 1.0)

Each task has a deterministic grader that evaluates:
1. **Delay improvement** vs fixed-timing baseline
2. **EV coordination chain** completeness (receive → react → forward)
3. **Message quality** (relevance, timing, accuracy)
4. **Equity** (max approach delay bounded)
5. **Gridlock prevention** (severe penalty for deadlocks)

---

## Setup & Usage

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Local Development

```bash
# Clone the repo
git clone https://github.com/Tarunts19/traffic-control-env.git
cd traffic-control-env

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r traffic_env/server/requirements.txt

# Run the server
uvicorn traffic_env.server.app:app --host 0.0.0.0 --port 7860

# Run baseline (rule-based agent)
python -m traffic_env.server.baseline_inference

# Run LLM inference
export HF_TOKEN="your-hf-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Docker

```bash
docker build -t traffic-control-env .
docker run -p 7860:7860 traffic-control-env
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset`  | POST | Start new episode |
| `/step`   | POST | Execute action |
| `/state`  | GET  | Current state |
| `/schema` | GET  | Action/Observation schemas |

---

## Baseline Scores

Rule-based agent (longest-queue-first + emergency override):

| Task | Steps | Reward | Avg Delay | Vehicles Cleared |
|------|-------|--------|-----------|-----------------|
| single_intersection | 60 | 8.64 | 12.92s | 48 |
| corridor_green_wave | 120 | 15.39 | 14.65s | 226 |
| grid_incident | 180 | 25.36 | 19.00s | 488 |

---

## Project Structure

```
traffic_control_env/
├── inference.py              # LLM inference script (mandatory format)
├── openenv.yaml              # OpenEnv spec manifest
├── Dockerfile                # Container build
├── README.md
└── traffic_env/
    ├── models.py             # Typed Action, Observation, State
    └── server/
        ├── app.py            # FastAPI server (uses create_fastapi_app)
        ├── environment.py    # Core OpenEnv environment
        ├── simulation.py     # Cell-based traffic simulation
        ├── intersection.py   # Signal phase management
        ├── road.py           # Road segments with vehicle movement
        ├── vehicle.py        # Vehicle and EmergencyVehicle classes
        ├── task.py           # Task definitions (easy/medium/hard)
        ├── grader.py         # Deterministic episode grading
        └── baseline_inference.py  # Rule-based baseline
```

---

## License

MIT

## Author

Tarun Singh — [Hugging Face](https://huggingface.co/Tarunts19)
