import traci
import sys
import json
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
SUMO_CONFIG = BASE_DIR / "simple.sumocfg.xml"
QTABLE_PATH = BASE_DIR / "qtable.json"
RESULTS_PATH = BASE_DIR / "results.png"
BASELINE_TRAVEL_PATH = BASE_DIR / "baseline_travel.json"
# === BASELINE MODE - NO PRIORITY (fixed traffic lights only) ===
BASELINE_MODE = True
print("\n=== BASELINE RUN — NO ADAPTIVE PRIORITY ===\n")

# ── Q-LEARNING CONFIG ─────────────────────────────────────────────────────
ALPHA   = 0.1
GAMMA   = 0.9
EPSILON = 0.2
ACTIONS = [0, 2, 4, 6]  # Single-approach green program indices
TL_ID   = "A0"

# ── VEHICLE TYPE SETS ─────────────────────────────────────────────────────
EMERGENCY_TYPES  = {"ambulance", "police", "fire_truck"}
PUBLIC_TRANSPORT = {"bus"}

# ── MULTI-FACTOR SCORING WEIGHTS ──────────────────────────────────────────
W_DIST    = 0.5
W_SPEED   = 0.3
W_QUEUE   = 0.2
MAX_DIST  = 80.0
MAX_SPEED = 16.67
MAX_QUEUE = 10.0

# ── EMERGENCY VEHICLE CONFIG ──────────────────────────────────────────────
# lane      : the lane the vehicle travels on toward A0
# phase     : traffic light phase that gives THIS vehicle a green light
# max_speed : vehicle's max speed (m/s) — used for score normalisation
# type      : label for display / metrics
EMERGENCY_VEHICLES = {
    "amb1":  {"lane": "left0A0_0",   "phase": 6, "max_speed": 16.67, "type": "ambulance"},
    "pol1":  {"lane": "right0A0_0",  "phase": 2, "max_speed": 19.44, "type": "police"},
    "fire1": {"lane": "top0A0_0",    "phase": 0, "max_speed": 14.0,  "type": "fire_truck"},
    "amb2":  {"lane": "bottom0A0_0", "phase": 4, "max_speed": 16.67, "type": "ambulance"},
    # Same arm/phase as fire1: arbitration between them is cosmetic only; priority still grants the correct phase for both.
    "pol2":  {"lane": "top0A0_0",    "phase": 0, "max_speed": 19.44, "type": "police"},
    # Same arm/phase as pol1: arbitration between them is cosmetic only; priority still grants the correct phase for both.
    "fire2": {"lane": "right0A0_0",  "phase": 2, "max_speed": 14.0,  "type": "fire_truck"},
}

# ── BASELINE TRAVEL TIMES (seconds, without priority) ─────────────────────
# Run once with priority disabled to fill these in accurately.
# Current values are estimates — update after your baseline run.
BASELINE_TRAVEL = {
    "amb1": 55.0,
    "pol1": 55.0,
    "fire1": 60.0,
    "amb2": 55.0,
    "pol2": 55.0,
    "fire2": 60.0,
}

# ── Q-TABLE ───────────────────────────────────────────────────────────────
q_table = {s: {a: 0.0 for a in ACTIONS} for s in ("low", "medium", "high")}

# ── LOAD EXISTING Q-TABLE (so it improves across runs) ────────────────────
try:
    with open(QTABLE_PATH) as f:
        loaded = json.load(f)
        for s in loaded:
            for a_str, val in loaded[s].items():
                action = int(a_str)
                if s in q_table and action in q_table[s]:
                    q_table[s][action] = val
    print("Loaded existing Q-table from qtable.json\n")
except FileNotFoundError:
    print("No existing Q-table found — starting fresh.\n")

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────
def get_priority_score(dist, speed, queue, max_speed=MAX_SPEED):
    """Compute normalised 0–1 priority score from 3 factors."""
    norm_dist  = max(0.0, (MAX_DIST - dist) / MAX_DIST)   # closer  → higher
    norm_speed = min(1.0, speed / max_speed)               # faster  → higher
    norm_queue = min(1.0, queue / MAX_QUEUE)               # more q  → higher
    return round(W_DIST * norm_dist + W_SPEED * norm_speed + W_QUEUE * norm_queue, 4)

def score_to_state(score):
    if score < 0.35:   return "low"
    elif score < 0.65: return "medium"
    else:              return "high"

def choose_action(state, forced_phase=None):
    """Epsilon-greedy — biased toward the correct phase for this vehicle."""
    if forced_phase is not None and random.random() > EPSILON:
        return forced_phase        # exploit: pick the right lane phase
    return random.choice(ACTIONS)  # explore

def update_q(state, action, reward, next_state):
    best_next = max(q_table[next_state].values())
    old = q_table[state][action]
    q_table[state][action] = old + ALPHA * (reward + GAMMA * best_next - old)

def get_queue(lane_id):
    try:    return traci.lane.getLastStepHaltingNumber(lane_id)
    except: return 0

# ── PERFORMANCE METRICS STORAGE ───────────────────────────────────────────
metrics = {
    vid: {
        "depart_step":          None,
        "exit_step":            None,
        "priority_activations": 0,
        "retrigger_count":      0,
        "peak_score":           0.0,
        "score_sum":            0.0,
        "score_count":          0,
    }
    for vid in EMERGENCY_VEHICLES
}

# ── LAUNCH SUMO ───────────────────────────────────────────────────────────
print("=" * 60)
print("  Adaptive Emergency Priority System — VANET + Q-Learning")
print("  Team: Amritha Zerin D, Harshini G, Kayalvizhi M")
print("=" * 60)
print("Starting SUMO... press PLAY in the GUI toolbar.\n")

try:
    traci.start([
        "sumo-gui",
        "-c", str(SUMO_CONFIG),
        "--delay", "100",
    ])
    print("TraCI connected!\n")
except Exception as e:
    print("Failed to connect:", e)
    input("Press Enter to close...")
    sys.exit(1)

# ── SIMULATION STATE ──────────────────────────────────────────────────────
step         = 0
priority_active = {vid: False for vid in EMERGENCY_VEHICLES}
last_state      = {vid: "low" for vid in EMERGENCY_VEHICLES}
last_action     = {vid: cfg["phase"] for vid, cfg in EMERGENCY_VEHICLES.items()}
score_logs      = {vid: [] for vid in EMERGENCY_VEHICLES}
step_logs       = {vid: [] for vid in EMERGENCY_VEHICLES}
approach_wait_sums = {"top0A0_0": 0.0, "right0A0_0": 0.0, "bottom0A0_0": 0.0, "left0A0_0": 0.0}
approach_wait_counts = {"top0A0_0": 0, "right0A0_0": 0, "bottom0A0_0": 0, "left0A0_0": 0}

# V2V broadcast tracking
v2v_broadcast_count = 0
V2V_RADIUS = 150.0  # metres

print(f"{'Step':>6} | {'Vehicle':>8} | {'Score':>6} | {'State':>7} | "
      f"{'Action':>6} | {'Q0':>7} | {'Q2':>7} | {'Q4':>7} | {'Q6':>7} | {'Event':>12}")
print("-" * 104)

# ── MAIN SIMULATION LOOP ──────────────────────────────────────────────────
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    step += 1
    vehicles = traci.vehicle.getIDList()
    for lane_id in approach_wait_sums:
        approach_wait_sums[lane_id] += traci.lane.getWaitingTime(lane_id)
        approach_wait_counts[lane_id] += 1

    # ── VEHICLE TYPE CENSUS (every 100 steps) ─────────────────────────────
    if step % 100 == 0:
        print(f"\n  [Step {step}] Vehicle census ({len(vehicles)} on road):")
        for veh in vehicles:
            vtype = traci.vehicle.getTypeID(veh)
            if vtype in EMERGENCY_TYPES:
                cat = f"EMERGENCY ({vtype})"
            elif vtype in PUBLIC_TRANSPORT:
                cat = f"PUBLIC TRANSPORT ({vtype})"
            else:
                cat = "PRIVATE CAR"
            print(f"    {veh:>12} → {cat}")
        print()

    # ── V2V BROADCAST CHECK ───────────────────────────────────────────────
    # Count how many vehicles are within V2V radius of any emergency vehicle
    for vid in EMERGENCY_VEHICLES:
        if vid in vehicles:
            try:
                ex, ey = traci.vehicle.getPosition(vid)
                nearby = 0
                for other in vehicles:
                    if other == vid:
                        continue
                    ox, oy = traci.vehicle.getPosition(other)
                    dist_v2v = ((ex - ox)**2 + (ey - oy)**2) ** 0.5
                    if dist_v2v <= V2V_RADIUS:
                        nearby += 1
                if nearby > 0:
                    v2v_broadcast_count += nearby
            except:
                pass

    # ── PROCESS EACH EMERGENCY VEHICLE ────────────────────────────────────
    for vid, cfg in EMERGENCY_VEHICLES.items():
        if vid in vehicles:

            # Record first appearance (actual depart into network)
            if metrics[vid]["depart_step"] is None:
                metrics[vid]["depart_step"] = step

            # Gather position / speed
            try:
                lane_id  = traci.vehicle.getLaneID(vid)
                lane_pos = traci.vehicle.getLanePosition(vid)
                lane_len = traci.lane.getLength(lane_id)
                if lane_id.startswith(":"):
                    dist = 0.0
                else:
                    dist = max(0.0, lane_len - lane_pos)
                speed    = traci.vehicle.getSpeed(vid)
            except:
                continue

            queue = get_queue(cfg["lane"])
            score = get_priority_score(dist, speed, queue, cfg["max_speed"])
            state = score_to_state(score)

            # Log score for graph and aggregate metrics
            score_logs[vid].append(score)
            step_logs[vid].append(step)
            metrics[vid]["score_sum"]   += score
            metrics[vid]["score_count"] += 1
            if score > metrics[vid]["peak_score"]:
                metrics[vid]["peak_score"] = score

            # Q-learning: reward previous action
            if priority_active[vid]:
                phase_bonus = 2.0 if last_action[vid] == cfg["phase"] else 0.0
                reward = score * 10 + phase_bonus
                update_q(last_state[vid], last_action[vid], reward, state)

            # Choose action (biased toward correct phase)
            action = choose_action(state, forced_phase=cfg["phase"])

                        # ── PRIORITY TRIGGER + RE-TRIGGER ─────────────────────────────
            if BASELINE_MODE:
                # Baseline: do NOTHING → let SUMO static traffic light run
                pass
            else:
                if dist <= MAX_DIST:
                    if not priority_active[vid]:
                        # First trigger
                        traci.trafficlight.setPhase(TL_ID, action)
                        traci.trafficlight.setPhaseDuration(TL_ID, 30)
                        priority_active[vid] = True
                        metrics[vid]["priority_activations"] += 1
                        print(f"{step:>6} | {vid:>8} | {score:>6.3f} | {state:>7} | "
                              f"{action:>6} | {q_table[state][0]:>7.2f} | {q_table[state][2]:>7.2f} | "
                              f"{q_table[state][4]:>7.2f} | {q_table[state][6]:>7.2f} | {'PRIORITY ON':>12}")

                    elif step % 50 == 0:
                        # Re-trigger every 50 steps
                        traci.trafficlight.setPhase(TL_ID, action)
                        traci.trafficlight.setPhaseDuration(TL_ID, 30)
                        metrics[vid]["retrigger_count"] += 1
                        print(f"{step:>6} | {vid:>8} | {score:>6.3f} | {state:>7} | "
                              f"{action:>6} | {q_table[state][0]:>7.2f} | {q_table[state][2]:>7.2f} | "
                              f"{q_table[state][4]:>7.2f} | {q_table[state][6]:>7.2f} | {'RE-TRIGGER':>12}")

                elif dist > MAX_DIST and priority_active[vid]:
                    priority_active[vid] = False
                    print(f"{step:>6} | {vid:>8} | {score:>6.3f} | {state:>7} | "
                          f"{'--':>6} | {q_table[state][0]:>7.2f} | {q_table[state][2]:>7.2f} | "
                          f"{q_table[state][4]:>7.2f} | {q_table[state][6]:>7.2f} | {'CLEARED':>12}")

            last_state[vid]  = state
            last_action[vid] = action

        else:
            # Vehicle has left the simulation
            if priority_active[vid]:
                priority_active[vid] = False
                print(f"{step:>6} | {vid:>8} | {'--':>6} | {'--':>7} | "
                      f"{'--':>6} | {'--':>7} | {'--':>7} | {'--':>7} | {'PASSED':>12}")
            if metrics[vid]["depart_step"] and metrics[vid]["exit_step"] is None:
                metrics[vid]["exit_step"] = step

# ── MEASURED BASELINE TRAVEL (for run_emergency.py comparison) ────────────
_measured = {}
for vid, m in metrics.items():
    if m["depart_step"] and m["exit_step"]:
        _measured[vid] = round((m["exit_step"] - m["depart_step"]) * 0.1, 1)
if _measured:
    out = dict(_measured)
    if len(_measured) < len(EMERGENCY_VEHICLES):
        print(
            f"WARNING: baseline_travel incomplete ({len(_measured)}/{len(EMERGENCY_VEHICLES)} EMVs); "
            f"merging with existing {BASELINE_TRAVEL_PATH.name} if present.\n"
        )
        if BASELINE_TRAVEL_PATH.exists():
            try:
                with open(BASELINE_TRAVEL_PATH, encoding="utf-8") as f:
                    prev = json.load(f)
                if isinstance(prev, dict):
                    out = {**prev, **out}
            except Exception:
                pass
    with open(BASELINE_TRAVEL_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nMeasured baseline travel times saved to {BASELINE_TRAVEL_PATH.name}")
    print("  (run_emergency.py loads this automatically for fair comparison.)\n")

# ── SAVE Q-TABLE ──────────────────────────────────────────────────────────
with open(QTABLE_PATH, "w") as f:
    json.dump({s: {str(a): v for a, v in acts.items()} for s, acts in q_table.items()}, f, indent=2)
print("\nQ-table saved to qtable.json")
print("Final Q-table:")
for s, actions in q_table.items():
    print(
        f"  {s:>7}: phase0={actions[0]:.4f}  phase2={actions[2]:.4f}  "
        f"phase4={actions[4]:.4f}  phase6={actions[6]:.4f}"
    )

# ── PERFORMANCE METRICS REPORT ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PERFORMANCE METRICS REPORT")
print("=" * 60)

for vid, m in metrics.items():
    cfg = EMERGENCY_VEHICLES[vid]
    print(f"\n  {vid.upper()} — {cfg['type'].upper()}")
    print(f"  {'─' * 40}")

    if m["depart_step"] and m["exit_step"]:
        travel_steps = m["exit_step"] - m["depart_step"]
        travel_time  = travel_steps * 0.1          # step-length = 0.1s
        avg_score    = (m["score_sum"] / m["score_count"]
                        if m["score_count"] > 0 else 0.0)

        print(f"    Depart step              : {m['depart_step']}")
        print(f"    Exit step                : {m['exit_step']}")
        print(f"    Travel time (fixed signal, no adaptive priority) : {travel_time:.1f} s")
        # Do not print "time saved" here — that compares adaptive vs reference in run_emergency.py only.
        # BASELINE_TRAVEL in memory is stale vs this run; reference file is updated above.
        print(f"    Reference for adaptive run : {BASELINE_TRAVEL_PATH.name} (this run's times)")

        print(f"    Priority activations     : {m['priority_activations']}")
        print(f"    Re-trigger count         : {m['retrigger_count']}")
        print(f"    Peak priority score      : {m['peak_score']:.4f}")
        print(f"    Avg priority score       : {avg_score:.4f}")

    else:
        print(f"    Vehicle did not complete its route in this run.")

print(f"\n  V2V BROADCASTS")
print(f"  {'─' * 40}")
print(f"    Total V2V messages sent  : {v2v_broadcast_count}")
print(f"    V2V radius used          : {V2V_RADIUS} m")

avg_waits = []
for lane_id, total_wait in approach_wait_sums.items():
    n = max(approach_wait_counts[lane_id], 1)
    avg_waits.append(total_wait / n)
if sum(x * x for x in avg_waits) > 0:
    jain_index = (sum(avg_waits) ** 2) / (len(avg_waits) * sum(x * x for x in avg_waits))
else:
    jain_index = 1.0
print(f"\n  FAIRNESS (Jain index (lane-mean) over approach waiting times)")
print(f"  {'─' * 40}")
print(f"    Jain fairness index      : {jain_index:.4f}")
print(f"    (1.0 is best fairness)")

# ── SAVE GRAPH ────────────────────────────────────────────────────────────
n = len(EMERGENCY_VEHICLES)
fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), squeeze=False)

colors = ["red", "blue", "darkorange"]
for i, (vid, cfg) in enumerate(EMERGENCY_VEHICLES.items()):
    ax  = axes[i][0]
    col = colors[i % len(colors)]
    if score_logs[vid]:
        ax.plot(step_logs[vid], score_logs[vid],
                color=col, linewidth=2, label=f"{vid} Priority Score")
        ax.axhline(0.65, color="orange", linestyle="--",
                   linewidth=1.5, label="High threshold (0.65)")
        ax.axhline(0.35, color="green",  linestyle="--",
                   linewidth=1.5, label="Medium threshold (0.35)")
        ax.fill_between(step_logs[vid], score_logs[vid], 0,
                        alpha=0.08, color=col)
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Priority Score (0–1)")
        ax.set_title(f"{vid.upper()} ({cfg['type']}) — Priority Score over Time\n"
                     f"Peak: {metrics[vid]['peak_score']:.3f}  |  "
                     f"Avg: {(metrics[vid]['score_sum']/max(metrics[vid]['score_count'],1)):.3f}")
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title(f"{vid.upper()} ({cfg['type']}) — No data recorded")

plt.tight_layout()
plt.savefig(RESULTS_PATH, dpi=150)
print("\nGraph saved to results.png")

traci.close()
print("\nSimulation complete!")
input("Press Enter to close...")