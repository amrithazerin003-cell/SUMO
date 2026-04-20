# Adaptive Emergency Vehicle Priority System

Project name: Adaptive Emergency Vehicle Priority System. Team: Amritha Zerin D, Harshini G, Kayalvizhi M. Tech: SUMO + Python TraCI + Q-Learning + VANET. Single 4-arm intersection junction A0, 6 EMVs in 2 waves.

## Prerequisites

- SUMO ([Download link](https://sumo.dlr.de/docs/Downloads.php))
- Python 3.8+
- `pip install traci matplotlib`
- Set `SUMO_HOME` environment variable

## Setup

```bash
git clone https://github.com/amrithazerin003-cell/SUMO.git
cd SUMO
```

## How to Run

1. `python baseline.py`  
   Press Play in SUMO GUI. This writes `baseline_travel.json`.
2. `python run_emergency.py`  
   Press Play in SUMO GUI. Results are printed in terminal, graph saved to `results.png`.

Note: always run `baseline.py` before `run_emergency.py` on a new machine.

## Results

| EMV | Baseline -> With Priority | Improvement |
|---|---|---|
| AMB1 | 84.1s->21.2s | 74.8% |
| POL1 | 18.1s->8.1s | 55.2% |
| FIRE1 | 53.6s->11.8s | 78.0% |
| AMB2 | 59.1s->10.8s | 81.7% |
| POL2 | 67.9s->7.8s | 88.5% |
| FIRE2 | 56.0s->10.8s | 80.7% |

Jain fairness: 0.9123  
V2V messages: 3349

## File Structure

| File | Purpose |
|---|---|
| `run_emergency.py` | Adaptive emergency-priority simulation using Q-learning |
| `baseline.py` | Baseline (fixed-signal) simulation and baseline travel capture |
| `routes.rou.xml` | Vehicle types, background flows, and EMV/bus departures/routes |
| `simple.net.xml` | 4-arm intersection network and traffic light logic |
| `simple.sumocfg.xml` | SUMO simulation configuration |
| `baseline_travel.json` | Baseline travel-time reference values |
| `qtable.json` | Learned Q-table values |
| `results.png` | Output graph from emergency-priority run |
