# Decentralized Perishable Inventory Model with Transshipment and Emergency Orders

This repository contains a small simulation and report package for studying a two-node perishable inventory system under demand surges. The core question is:

> When node A is hit by a surge, is it better to wait for normal warehouse replenishment, use node B transshipment, or place an emergency warehouse order?

The current version models:

- two facilities: `A` and `B`
- three usable inventory ages: `new`, `medium`, `old`
- FEFO issuing (`first-expire-first-out`)
- regular warehouse replenishment with lead times
- one-period delayed transshipment from `B -> A`
- one-period delayed emergency warehouse orders to `A`
- two cost regimes:
  - `Underage > Overage`
  - `Overage > Underage`
- continuous review as the main operating baseline
- periodic review as a benchmark reference


## Files

- `final_report_sim.py`  
  Standalone Python simulation used to generate the core policy comparisons.

- `final_report.tex`  
  Main LaTeX report describing the model, assumptions, equations, and results.

- `final_report.pdf`  
  Compiled PDF of the main report.


## Main Modeling Idea

Node `A` is the stressed node. Node `B` may help as a donor node if it has releasable stock.

The model compares three surge responses:

1. `Warehouse only`  
   No extra emergency action. A waits for normal replenishment.

2. `Use node B`  
   B transships inventory to A if shipping cost is lower than the cost of making A wait.

3. `Emergency order`  
   A places an expedited warehouse order if the emergency unit cost is lower than the cost of making A wait.

Both transshipment and emergency orders take **one full period** to arrive. That means they do **not** fix the shortage in the same period as the surge. They change the recovery path in the next period.

## Cost Logic

The cost structure includes:

- lost-profit / waiting cost for backlog
- regular ordering cost
- holding cost
- expiry cost
- transshipment cost
- fresh-unit transshipment surcharge
- emergency warehouse order cost

The emergency-order unit cost is currently set to:

```text
c^em = 6.0
```

The base transshipment unit cost is:

```text
c^tr = 1.2
```

Fresh transshipment carries an extra surcharge:

```text
c^new = 2.0
```

## Key Assumptions

- planning horizon: `15` periods
- surge occurs in `period 7`
- node `A` demand is doubled during the surge
- transshipment lead time: `1` period
- emergency order lead time: `1` period
- emergency arrivals enter as `new` inventory
- transshipment cap: `70` units per period
- random seed fixed at `42`

## Requirements

To run the simulator:

- Python 3.10+ recommended
- `numpy`
- `pandas`

To compile the LaTeX reports:

- `tectonic` or another LaTeX engine with the required packages

## How to Run

Run the simulation:

```bash
python3 final_report_sim.py
```

This prints two summary tables:

- baseline comparison
- surge response comparison

Compile the main report:

```bash
tectonic final_report.tex
```


## What the Results Show

The main conclusion of the current version is:

- continuous review is the right baseline for testing emergency response
- node `B` transshipment is the preferred first response when B has releasable stock
- emergency warehouse ordering is a fallback

Under the current assumptions:

- in the `Underage > Overage` regime, transshipment and emergency ordering produce the same service improvement, but transshipment is cheaper
- in the `Overage > Underage` regime, transshipment still helps, but emergency ordering is too expensive to activate

## Intended Use

This repository is set up for:

- simulation-based policy comparison
- LaTeX report writing
- presentation preparation
- sensitivity analysis on costs, lead times, donor reserve, and surge assumptions

## Notes

This is a research / report repository, not a packaged software library. The code is written to support transparent analysis and reproducible report tables rather than to serve as a general-purpose inventory optimization package.
