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

# ── Q-LEARNING CONFIG ─────────────────────────────────────────────────────
ALPHA   = 0.1
GAMMA   = 0.9
EPSILON = 0.2
# Single-approach green program:
# 0=top, 2=right, 4=bottom, 6=left
ACTIONS = [0, 2, 4, 6]
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
LOCK_STEPS = 35
RETRIGGER_INTERVAL = 120
SCORE_TIE_THRESHOLD = 0.04
MAX_CONSECUTIVE_HOLDS = 2
STARVATION_STEPS = 120
# Type weight for arbitration only (does not change displayed priority score)
EMV_TYPE_WEIGHT = {"fire_truck": 0.20, "ambulance": 0.06, "police": 0.03}
# Keep same winner across lock expiry if still competitive (reduces EMV flip-flop)
STICKY_ARB_EPS = 0.07

# ── EMERGENCY VEHICLE CONFIG ──────────────────────────────────────────────
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

# ── BASELINE TRAVEL TIMES (seconds) ────────────────────────────────────────
# Defaults used until you run baseline.py (writes baseline_travel.json).
_DEFAULT_BASELINE_TRAVEL = {
    "amb1": 55.0,
    "pol1": 55.0,
    "fire1": 60.0,
    "amb2": 55.0,
    "pol2": 55.0,
    "fire2": 60.0,
}
BASELINE_TRAVEL = dict(_DEFAULT_BASELINE_TRAVEL)
try:
    with open(BASELINE_TRAVEL_PATH, encoding="utf-8") as f:
        loaded_bt = json.load(f)
    for k, v in loaded_bt.items():
        if k in BASELINE_TRAVEL and isinstance(v, (int, float)):
            BASELINE_TRAVEL[k] = float(v)
    print(f"Loaded baseline travel times from {BASELINE_TRAVEL_PATH.name}\n")
except FileNotFoundError:
    print(
        f"No {BASELINE_TRAVEL_PATH.name} yet — run: python baseline.py\n"
        "Using embedded defaults for comparison until then.\n"
    )

# ── Q-TABLE ───────────────────────────────────────────────────────────────
q_table = {s: {a: 0.0 for a in ACTIONS} for s in ("low", "medium", "high")}

# ── LOAD EXISTING Q-TABLE ─────────────────────────────────────────────────
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
    norm_dist  = max(0.0, (MAX_DIST - dist) / MAX_DIST)
    norm_speed = min(1.0, speed / max_speed)
    norm_queue = min(1.0, queue / MAX_QUEUE)
    return round(W_DIST * norm_dist + W_SPEED * norm_speed + W_QUEUE * norm_queue, 4)

def score_to_state(score):
    if score < 0.35:   return "low"
    elif score < 0.65: return "medium"
    else:              return "high"

def choose_action(state, forced_phase=None):
    if forced_phase is not None and random.random() > EPSILON:
        return forced_phase
    return random.choice(ACTIONS)

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
        "depart_step": None, "exit_step": None,
        "priority_activations": 0, "retrigger_count": 0,
        "peak_score": 0.0, "score_sum": 0.0, "score_count": 0,
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
    # Avoid --time-to-teleport: low values cause mass teleporting under congestion.
    traci.start([
        "sumo-gui",
        "-c", str(SUMO_CONFIG),
        "--delay", "100",
    ])
    print("TraCI connected!\n")
except Exception as e:
    print(f"Failed to start SUMO / TraCI: {e}")
    print(f"Config: {SUMO_CONFIG}")
    input("Press Enter to close...")
    sys.exit(1)

# ── SIMULATION STATE ──────────────────────────────────────────────────────
step = 0
priority_active = {vid: False for vid in EMERGENCY_VEHICLES}
last_state      = {vid: "low" for vid in EMERGENCY_VEHICLES}
last_action     = {vid: cfg["phase"] for vid, cfg in EMERGENCY_VEHICLES.items()}
score_logs      = {vid: [] for vid in EMERGENCY_VEHICLES}
step_logs       = {vid: [] for vid in EMERGENCY_VEHICLES}
approach_wait_sums = {"top0A0_0": 0.0, "right0A0_0": 0.0, "bottom0A0_0": 0.0, "left0A0_0": 0.0}
approach_wait_counts = {"top0A0_0": 0, "right0A0_0": 0, "bottom0A0_0": 0, "left0A0_0": 0}
in_range_entry_step = {vid: None for vid in EMERGENCY_VEHICLES}
# One counted "activation" per 80m approach episode (avoids inflated counts when winner alternates)
priority_counted_this_approach = {vid: False for vid in EMERGENCY_VEHICLES}
arbitration_active_vid = None
arbitration_phase = None
arbitration_until_step = -1
arbitration_last_set_step = -1
consecutive_hold_count = 0

v2v_broadcast_count = 0
V2V_RADIUS = 150.0

print(f"{'Step':>6} | {'Vehicle':>8} | {'Score':>6} | {'State':>7} | "
      f"{'Action':>6} | {'Q0':>7} | {'Q2':>7} | {'Q4':>7} | {'Q6':>7} | {'Event':>12}")
print("-" * 104)

had_emv_candidates_prev = False

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
    step += 1
    vehicles = traci.vehicle.getIDList()
    for lane_id in approach_wait_sums:
        approach_wait_sums[lane_id] += traci.lane.getWaitingTime(lane_id)
        approach_wait_counts[lane_id] += 1

    if step % 100 == 0:
        print(f"\n  [Step {step}] Vehicle census ({len(vehicles)} on road):")
        for veh in vehicles:
            vtype = traci.vehicle.getTypeID(veh)
            cat = f"EMERGENCY ({vtype})" if vtype in EMERGENCY_TYPES else \
                  f"PUBLIC TRANSPORT ({vtype})" if vtype in PUBLIC_TRANSPORT else "PRIVATE CAR"
            print(f"    {veh:>12} → {cat}")
        print()

    # V2V broadcast count
    for vid in EMERGENCY_VEHICLES:
        if vid in vehicles:
            try:
                ex, ey = traci.vehicle.getPosition(vid)
                nearby = sum(1 for other in vehicles if other != vid and
                             ((ex - traci.vehicle.getPosition(other)[0])**2 +
                              (ey - traci.vehicle.getPosition(other)[1])**2)**0.5 <= V2V_RADIUS)
                if nearby > 0:
                    v2v_broadcast_count += nearby
            except:
                pass

    candidates = {}
    for vid, cfg in EMERGENCY_VEHICLES.items():
        if vid in vehicles:
            if metrics[vid]["depart_step"] is None:
                metrics[vid]["depart_step"] = step

            try:
                lane_id  = traci.vehicle.getLaneID(vid)
                lane_pos = traci.vehicle.getLanePosition(vid)
                lane_len = traci.lane.getLength(lane_id)
                # Internal junction lanes (`:`) have tiny length; treat as at junction for range check
                if lane_id.startswith(":"):
                    dist = 0.0
                else:
                    dist = max(0.0, lane_len - lane_pos)
                speed = traci.vehicle.getSpeed(vid)
            except Exception:
                continue

            queue = get_queue(cfg["lane"])
            score = get_priority_score(dist, speed, queue, cfg["max_speed"])
            state = score_to_state(score)

            score_logs[vid].append(score)
            step_logs[vid].append(step)
            metrics[vid]["score_sum"] += score
            metrics[vid]["score_count"] += 1
            if score > metrics[vid]["peak_score"]:
                metrics[vid]["peak_score"] = score

            if priority_active[vid]:
                phase_bonus = 2.0 if last_action[vid] == cfg["phase"] else 0.0
                reward = score * 10 + phase_bonus
                update_q(last_state[vid], last_action[vid], reward, state)

            action = choose_action(state, forced_phase=cfg["phase"])

            if dist <= MAX_DIST:
                if in_range_entry_step[vid] is None:
                    in_range_entry_step[vid] = step
                candidates[vid] = {"cfg": cfg, "score": score, "state": state, "action": action}
            elif dist > MAX_DIST and priority_active[vid] and vid != arbitration_active_vid:
                priority_active[vid] = False
                in_range_entry_step[vid] = None
                priority_counted_this_approach[vid] = False
                print(f"{step:>6} | {vid:>8} | {score:>6.3f} | {state:>7} | "
                      f"{'--':>6} | {q_table[state][0]:>7.2f} | {q_table[state][2]:>7.2f} | "
                      f"{q_table[state][4]:>7.2f} | {q_table[state][6]:>7.2f} | {'CLEARED':>12}")
            elif dist > MAX_DIST:
                in_range_entry_step[vid] = None
                priority_counted_this_approach[vid] = False

            last_state[vid]  = state
            last_action[vid] = action

        else:
            if priority_active[vid]:
                priority_active[vid] = False
                in_range_entry_step[vid] = None
                priority_counted_this_approach[vid] = False
                print(f"{step:>6} | {vid:>8} | {'--':>6} | {'--':>7} | "
                      f"{'--':>6} | {'--':>7} | {'--':>7} | {'--':>7} | {'--':>7} | {'PASSED':>12}")
            if metrics[vid]["depart_step"] and metrics[vid]["exit_step"] is None:
                metrics[vid]["exit_step"] = step

    if arbitration_active_vid and arbitration_active_vid not in candidates:
        priority_active[arbitration_active_vid] = False
        in_range_entry_step[arbitration_active_vid] = None
        priority_counted_this_approach[arbitration_active_vid] = False
        arbitration_active_vid = None
        arbitration_phase = None
        arbitration_until_step = -1
        arbitration_last_set_step = -1
        consecutive_hold_count = 0

    if candidates:
        previous_winner = arbitration_active_vid
        waiting_steps = {
            v: step - (in_range_entry_step[v] if in_range_entry_step[v] is not None else step)
            for v in candidates
        }
        def arb_score(vid):
            c = candidates[vid]
            w = EMV_TYPE_WEIGHT.get(c["cfg"]["type"], 0.0)
            return c["score"] + w

        starved_vids = [v for v, waited in waiting_steps.items() if waited >= STARVATION_STEPS]

        if starved_vids:
            winner_vid = max(starved_vids, key=lambda v: (waiting_steps[v], arb_score(v)))
        elif arbitration_active_vid in candidates and step < arbitration_until_step:
            winner_vid = arbitration_active_vid
        else:
            # Demo-safe tie-break:
            # 1) highest score wins
            # 2) for near-equal scores, favor longer waiting in-range EMV
            top_arb = max(arb_score(v) for v in candidates)
            near_ties = [v for v in candidates if top_arb - arb_score(v) <= SCORE_TIE_THRESHOLD]
            winner_vid = max(
                near_ties,
                key=lambda v: (
                    waiting_steps[v],
                    arb_score(v),
                ),
            )
            # After lock expires: keep current winner if still near-best (stops rapid EMV switching)
            if (
                arbitration_active_vid in candidates
                and step >= arbitration_until_step
                and consecutive_hold_count < MAX_CONSECUTIVE_HOLDS
            ):
                top_arb2 = max(arb_score(v) for v in candidates)
                if arb_score(arbitration_active_vid) >= top_arb2 - STICKY_ARB_EPS:
                    winner_vid = arbitration_active_vid

        if (
            arbitration_active_vid in candidates
            and winner_vid == arbitration_active_vid
            and len(candidates) > 1
            and consecutive_hold_count >= MAX_CONSECUTIVE_HOLDS
        ):
            alternatives = [v for v in candidates if v != arbitration_active_vid]
            winner_vid = max(alternatives, key=lambda v: (waiting_steps[v], arb_score(v)))

        winner = candidates[winner_vid]
        if arbitration_active_vid and arbitration_active_vid != winner_vid and priority_active[arbitration_active_vid]:
            priority_active[arbitration_active_vid] = False

        # Only one emergency vehicle controls signal in a step.
        for vid in candidates:
            if vid != winner_vid:
                priority_active[vid] = False

        if not priority_active[winner_vid]:
            traci.trafficlight.setPhase(TL_ID, winner["action"])
            traci.trafficlight.setPhaseDuration(TL_ID, LOCK_STEPS)
            if previous_winner == winner_vid:
                consecutive_hold_count += 1
            else:
                consecutive_hold_count = 1
            priority_active[winner_vid] = True
            if not priority_counted_this_approach[winner_vid]:
                metrics[winner_vid]["priority_activations"] += 1
                priority_counted_this_approach[winner_vid] = True
            arbitration_last_set_step = step
            print(f"{step:>6} | {winner_vid:>8} | {winner['score']:>6.3f} | {winner['state']:>7} | "
                  f"{winner['action']:>6} | {q_table[winner['state']][0]:>7.2f} | {q_table[winner['state']][2]:>7.2f} | "
                  f"{q_table[winner['state']][4]:>7.2f} | {q_table[winner['state']][6]:>7.2f} | {'PRIORITY ON':>12}")
        elif arbitration_last_set_step >= 0 and step - arbitration_last_set_step >= RETRIGGER_INTERVAL:
            traci.trafficlight.setPhase(TL_ID, winner["action"])
            traci.trafficlight.setPhaseDuration(TL_ID, LOCK_STEPS)
            metrics[winner_vid]["retrigger_count"] += 1
            arbitration_last_set_step = step
            print(f"{step:>6} | {winner_vid:>8} | {winner['score']:>6.3f} | {winner['state']:>7} | "
                  f"{winner['action']:>6} | {q_table[winner['state']][0]:>7.2f} | {q_table[winner['state']][2]:>7.2f} | "
                  f"{q_table[winner['state']][4]:>7.2f} | {q_table[winner['state']][6]:>7.2f} | {'RE-TRIGGER':>12}")

        arbitration_active_vid = winner_vid
        arbitration_phase = winner["action"]
        arbitration_until_step = step + LOCK_STEPS

    # When no EMV is in the 80m zone, hand the junction back to the static program.
    # Without this, the last forced phase stays forever and gridlock + teleport spam get worse.
    n_cand = len(candidates)
    if had_emv_candidates_prev and n_cand == 0:
        try:
            traci.trafficlight.setProgram(TL_ID, "0")
        except Exception:
            pass
    had_emv_candidates_prev = n_cand > 0

# ── SAVE Q-TABLE & REPORT ─────────────────────────────────────────────────
with open(QTABLE_PATH, "w") as f:
    json.dump({s: {str(a): v for a, v in acts.items()} for s, acts in q_table.items()}, f, indent=2)

print("\nQ-table saved to qtable.json")
print("Final Q-table:")
for s, actions in q_table.items():
    print(
        f"  {s:>7}: phase0={actions[0]:.4f}  phase2={actions[2]:.4f}  "
        f"phase4={actions[4]:.4f}  phase6={actions[6]:.4f}"
    )

print("\n" + "=" * 60)
print("  PERFORMANCE METRICS REPORT")
print("=" * 60)

for vid, m in metrics.items():
    cfg = EMERGENCY_VEHICLES[vid]
    print(f"\n  {vid.upper()} — {cfg['type'].upper()}")
    print(f"  {'─' * 40}")

    if m["depart_step"] and m["exit_step"]:
        travel_steps = m["exit_step"] - m["depart_step"]
        travel_time  = travel_steps * 0.1
        baseline     = BASELINE_TRAVEL.get(vid)
        avg_score    = (m["score_sum"] / m["score_count"] if m["score_count"] > 0 else 0.0)

        print(f"    Depart step              : {m['depart_step']}")
        print(f"    Exit step                : {m['exit_step']}")
        print(f"    Travel time (with prio)  : {travel_time:.1f} s")

        if baseline is not None:
            saved = baseline - travel_time
            # Float noise (e.g. JSON baseline vs step-derived time) must not show -0.0 as "SLOWER"
            same_tol = 0.05
            if abs(saved) < same_tol:
                status = "≈ SAME as reference"
                saved_disp = 0.0
                pct_disp = 0.0
            elif saved > 0:
                status = "✓ FASTER"
                saved_disp = saved
                pct_disp = (saved / baseline * 100) if baseline > 0 else 0.0
            else:
                status = "✗ SLOWER — check re-trigger"
                saved_disp = saved
                pct_disp = (saved / baseline * 100) if baseline > 0 else 0.0
            print(f"    Travel time (baseline)   : {baseline:.1f} s")
            print(f"    Time saved               : {saved_disp:.1f} s ({pct_disp:.1f}%)  {status}")

        print(f"    Priority activations     : {m['priority_activations']}")
        print(f"    Re-trigger count         : {m['retrigger_count']}")
        print(f"    Peak priority score      : {m['peak_score']:.4f}")
        print(f"    Avg priority score       : {avg_score:.4f}")

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
fig, axes = plt.subplots(n, 1, figsize=(12, 4*n), squeeze=False)
colors = ["red", "blue", "darkorange", "purple", "teal", "brown"]
for i, (vid, cfg) in enumerate(EMERGENCY_VEHICLES.items()):
    ax = axes[i][0]
    col = colors[i % len(colors)]
    if score_logs[vid]:
        avg_score = metrics[vid]["score_sum"] / max(metrics[vid]["score_count"], 1)
        ax.plot(step_logs[vid], score_logs[vid], color=col, linewidth=2, label=f"{vid} Priority Score")
        ax.axhline(0.65, color="orange", linestyle="--", label="High (0.65)")
        ax.axhline(0.35, color="green",  linestyle="--", label="Medium (0.35)")
        ax.fill_between(step_logs[vid], score_logs[vid], 0, alpha=0.08, color=col)
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Priority Score (0–1)")
        ax.set_title(f"{vid.upper()} ({cfg['type']}) — Priority Score over Time\n"
                     f"Peak: {metrics[vid]['peak_score']:.3f} | Avg: {avg_score:.3f}")
        ax.legend(loc="upper left")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_PATH, dpi=150)
print("\nGraph saved to results.png")

traci.close()
print("\nSimulation complete!")
input("Press Enter to close...")