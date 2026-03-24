# ⚙️ Four-Bar Linkage Path Synthesizer

A **Streamlit web app** that synthesises a four-bar linkage mechanism whose coupler point traces a path through any 4 user-defined positions, then animates the mechanism.

---

## What it does

Given **4 prescribed coupler-point positions** (A, B, C, D), the app:

1. Runs a numerical optimisation (scipy `least_squares` with multiple random restarts) to find the 9 linkage parameters
2. Displays the results in a clean table
3. Renders a **GIF animation** of the mechanism in motion, with the coupler point tracing the path live

### Linkage topology

```
E ──[EG]── G ──[GH]── H ──[HF]── F
                │
                I  ← coupler tracer point
```

| Symbol | Meaning |
|--------|---------|
| **E, F** | Fixed ground pivots (rigid supports — can be anywhere) |
| **G, H** | Moving revolute joints on the cranks |
| **I** | Coupler tracer point — passes through A, B, C, D |
| **EG** | Input crank length |
| **HF** | Output crank length |
| **GH** | Coupler link length |
| **GI, IH** | Distances from coupler joints to tracer point |

### Inputs and outputs

| | Count | Description |
|--|-------|-------------|
| **Inputs** | 8 | x, y coordinates of A, B, C, D |
| **Outputs** | 9 | x_E, y_E, x_F, y_F, EG, GI, IH, HG, HF |

---

## Live Demo

> Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) — see deployment section below.

---

## Screenshots

| Input & Results | Animation |
|---|---|
| Enter 4 points in the sidebar, get 9 output values | Animated GIF of the linkage in motion |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/fourbar-linkage-synthesizer.git
cd fourbar-linkage-synthesizer
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
fourbar-linkage-synthesizer/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## How the Synthesis Works

### Problem formulation

The four-bar linkage has **9 unknown parameters**:

```
[x_E, y_E, x_F, y_F, EG, HF, GH, px, py]
```

Each prescribed point introduces **1 hidden variable** (the crank angle θ at that position), contributing 2 equations and 1 unknown — net **1 equation per point**.

With 4 points → 4 equations for 9 unknowns → **underdetermined** (infinitely many solutions). The optimiser finds one good solution from this family.

### Optimisation strategy

The full parameter vector has **13 variables**:

```
[x_E, y_E, x_F, y_F, EG, HF, GH, px, py, θ_A, θ_B, θ_C, θ_D]
```

**Residuals** (8 equations):

```
res_k = I(θ_k) − P_k    for k = A, B, C, D
```

where `I(θ_k)` is the forward-kinematics position of the coupler point at crank angle `θ_k`.

`scipy.optimize.least_squares` (Trust Region Reflective) minimises `Σ res²` with **300 random restarts** to escape local minima.

### Forward kinematics

Given crank angle θ₂:

1. **G** = E + EG · [cos θ₂, sin θ₂]
2. Solve triangle G–H–F (law of cosines) to find **H**
3. Coupler frame angle θ₃ = atan2(H−G)
4. **I** = G + R(θ₃) · [px, py]

---

## Deployment on Streamlit Community Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — done!

---

## Dependencies

```
streamlit
numpy
scipy
matplotlib
pillow
pandas
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Example

Default input points (pixels):

| Point | x | y |
|-------|---|---|
| A | 200 | 180 |
| B | 370 | 140 |
| C | 490 | 230 |
| D | 310 | 310 |

Expected output (approximate — many valid solutions exist):

| Parameter | Value |
|-----------|-------|
| x_E | ~100 |
| y_E | ~400 |
| x_F | ~400 |
| y_F | ~400 |
| EG | ~80 |
| GI | ~70 |
| IH | ~120 |
| HG | ~150 |
| HF | ~110 |
| RMS error | < 1e-6 |

---

## Notes

- **Non-Grashof linkages** (where the ground link is too long relative to the cranks) cannot fully rotate and will show a partial coupler curve — this is a physical constraint, not a bug.
- The solution is **not unique** — many different linkages can pass through the same 4 points. The optimiser returns one valid member of the solution family.
- Increasing **restarts** improves solution quality but increases computation time.

---

## License

MIT License — free to use, modify, and distribute.
