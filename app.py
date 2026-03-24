"""
Four-Bar Linkage Path Synthesis — Streamlit App
================================================
Enter 4 coupler-point positions (A, B, C, D).
The app synthesises a four-bar linkage and shows an animation of it moving.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import least_squares
import streamlit as st
import io
import tempfile
import os

# ─────────────────────────────────────────────────────────────────────────────
# Kinematics
# ─────────────────────────────────────────────────────────────────────────────

def forward_kinematics(theta2, E, F, EG, HF, GH, px, py):
    G = E + EG * np.array([np.cos(theta2), np.sin(theta2)])
    d_vec = F - G
    d = np.linalg.norm(d_vec)
    if d < 1e-9 or d > GH + HF or d < abs(GH - HF):
        return None
    a = (GH**2 - HF**2 + d**2) / (2.0 * d)
    h2 = GH**2 - a**2
    if h2 < 0:
        return None
    h = np.sqrt(h2)
    mid  = G + a * d_vec / d
    perp = np.array([-d_vec[1], d_vec[0]]) / d
    H    = mid - h * perp
    theta3 = np.arctan2(H[1] - G[1], H[0] - G[0])
    I = G + np.array([
        px * np.cos(theta3) - py * np.sin(theta3),
        px * np.sin(theta3) + py * np.cos(theta3),
    ])
    return G, H, I


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis
# ─────────────────────────────────────────────────────────────────────────────

def residuals(params, prescribed_pts):
    E  = np.array([params[0], params[1]])
    F  = np.array([params[2], params[3]])
    EG, HF, GH = params[4], params[5], params[6]
    px, py     = params[7], params[8]
    res = []
    for P, theta in zip(prescribed_pts, params[9:13]):
        r = forward_kinematics(theta, E, F, EG, HF, GH, px, py)
        if r is None:
            res.extend([1e4, 1e4])
        else:
            res.extend([r[2][0] - P[0], r[2][1] - P[1]])
    return np.array(res)


def synthesize(points, n_restarts=300, seed=42):
    rng    = np.random.default_rng(seed)
    pts    = np.array(points, dtype=float)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    avg_r  = np.mean(np.linalg.norm(pts - [cx, cy], axis=1))
    spread = max(avg_r, 60)
    canvas = max(spread * 6, 800)

    lo = [cx-canvas, cy-canvas, cx-canvas, cy-canvas,
          5, 5, 5, -canvas, -canvas,
          -np.pi, -np.pi, -np.pi, -np.pi]
    hi = [cx+canvas, cy+canvas, cx+canvas, cy+canvas,
          canvas, canvas, canvas, canvas, canvas,
          np.pi, np.pi, np.pi, np.pi]

    best_result, best_cost = None, np.inf

    for _ in range(n_restarts):
        angle   = rng.uniform(0, 2 * np.pi)
        dist    = rng.uniform(0.3, 2.5) * spread
        sp      = rng.uniform(0.3, 1.5) * spread
        py_sign = rng.choice([-1, 1])
        x0 = [
            cx + dist * np.cos(angle) - sp, cy + dist * np.sin(angle),
            cx + dist * np.cos(angle) + sp, cy + dist * np.sin(angle),
            rng.uniform(0.2, 1.5) * spread,
            rng.uniform(0.2, 1.5) * spread,
            rng.uniform(0.4, 2.0) * spread,
            rng.uniform(-0.6, 0.6) * spread,
            py_sign * rng.uniform(0.2, 1.5) * spread,
            *rng.uniform(-np.pi, np.pi, 4),
        ]
        try:
            res = least_squares(
                residuals, x0, args=(pts,), bounds=(lo, hi),
                method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=8000,
            )
            if res.cost < best_cost:
                best_cost, best_result = res.cost, res
        except Exception:
            pass

    if best_result is None:
        return None

    p = best_result.x
    E  = np.array([p[0], p[1]])
    F  = np.array([p[2], p[3]])
    EG, HF, GH = p[4], p[5], p[6]
    px_, py_   = p[7], p[8]
    thetas     = p[9:13]
    GI = np.sqrt(px_**2 + py_**2)
    IH = np.sqrt((px_ - GH)**2 + py_**2)
    rms = np.sqrt(best_cost * 2 / 4)

    return {
        'E': E, 'F': F,
        'EG': EG, 'HF': HF, 'GH': GH,
        'px': px_, 'py': py_,
        'GI': GI, 'IH': IH,
        'thetas': thetas,
        'rms_error': rms,
        'params': p,
    }


def verify(points, sol):
    errors = []
    for P, theta in zip(np.array(points), sol['thetas']):
        r = forward_kinematics(
            theta, sol['E'], sol['F'],
            sol['EG'], sol['HF'], sol['GH'],
            sol['px'], sol['py'],
        )
        errors.append(np.inf if r is None else float(np.linalg.norm(r[2] - P)))
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Animation builder
# ─────────────────────────────────────────────────────────────────────────────

def build_animation(points, sol, n_frames=120, fps=30):
    pts    = np.array(points)
    E, F   = sol['E'], sol['F']
    EG, HF, GH = sol['EG'], sol['HF'], sol['GH']
    px_, py_   = sol['px'], sol['py']

    # Collect the full coupler curve
    curve_pts = []
    for i in range(720):
        r = forward_kinematics(i * np.pi / 360, E, F, EG, HF, GH, px_, py_)
        if r:
            curve_pts.append(r[2])

    # Collect valid animation frames (sweep full crank rotation)
    all_thetas = np.linspace(0, 2 * np.pi, n_frames + 1)[:-1]
    frames = []
    for theta in all_thetas:
        r = forward_kinematics(theta, E, F, EG, HF, GH, px_, py_)
        if r:
            frames.append((theta, r[0], r[1], r[2]))   # theta, G, H, I

    if not frames:
        return None

    # Axis limits with padding
    all_x = [E[0], F[0]] + [p[0] for p in pts]
    all_y = [E[1], F[1]] + [p[1] for p in pts]
    if curve_pts:
        all_x += [p[0] for p in curve_pts]
        all_y += [p[1] for p in curve_pts]
    for _, G, H, I in frames:
        all_x += [G[0], H[0], I[0]]
        all_y += [G[1], H[1], I[1]]
    pad = (max(all_x) - min(all_x) + max(all_y) - min(all_y)) * 0.08 + 20
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title('Four-Bar Linkage — Coupler Path Synthesis', fontsize=13, pad=10)
    ax.grid(True, alpha=0.18, linestyle='--', color='gray')
    ax.set_facecolor('#f9f9f9')

    ORANGE = '#E85D24'
    BLUE   = '#3B8BD4'
    GREEN  = '#27B47A'
    PURPLE = '#9B6BD4'
    AMBER  = '#F2A623'
    GRAY   = '#666666'

    # Static elements
    # Full coupler curve (dashed)
    if curve_pts:
        cx_arr = [p[0] for p in curve_pts]
        cy_arr = [p[1] for p in curve_pts]
        ax.plot(cx_arr, cy_arr, color=AMBER, lw=1.5, ls='--', alpha=0.55,
                label='Coupler curve', zorder=2)

    # Ground link E–F
    ax.plot([E[0], F[0]], [E[1], F[1]], color=GRAY, lw=2, ls='-',
            alpha=0.4, zorder=2)

    # Fixed pivots E and F
    for pt, lbl in [(E, 'E'), (F, 'F')]:
        ax.scatter(*pt, color=ORANGE, s=150, zorder=9, edgecolors='white', linewidths=1.5)
        ax.annotate(lbl, pt, textcoords='offset points', xytext=(8, 6),
                    fontsize=12, fontweight='bold', color=ORANGE)

    # Target points A–D
    for k, P in enumerate(pts):
        ax.scatter(*P, color=PURPLE, s=130, zorder=10,
                   edgecolors='white', linewidths=1.5)
        ax.annotate('ABCD'[k], P, textcoords='offset points', xytext=(8, 6),
                    fontsize=13, fontweight='bold', color=PURPLE)

    # Dynamic elements (updated each frame)
    line_EG,  = ax.plot([], [], color=ORANGE, lw=3,   zorder=5, solid_capstyle='round')
    line_GH,  = ax.plot([], [], color=GRAY,   lw=2.5, zorder=5, solid_capstyle='round')
    line_HF,  = ax.plot([], [], color=ORANGE, lw=3,   zorder=5, solid_capstyle='round')
    line_GI,  = ax.plot([], [], color=GREEN,  lw=1.5, ls=':',   zorder=4)
    line_HI,  = ax.plot([], [], color=GREEN,  lw=1.5, ls=':',   zorder=4)
    trail_line,= ax.plot([], [], color=AMBER,  lw=2.2, zorder=3, alpha=0.9)

    dot_G  = ax.scatter([], [], color=BLUE,   s=70,  zorder=8, edgecolors='white', linewidths=1.2)
    dot_H  = ax.scatter([], [], color=BLUE,   s=70,  zorder=8, edgecolors='white', linewidths=1.2)
    dot_I  = ax.scatter([], [], color=GREEN,  s=100, zorder=8, edgecolors='white', linewidths=1.5)

    lbl_G  = ax.text(0, 0, 'G', fontsize=9,  color=BLUE,  fontweight='bold', zorder=11)
    lbl_H  = ax.text(0, 0, 'H', fontsize=9,  color=BLUE,  fontweight='bold', zorder=11)
    lbl_I  = ax.text(0, 0, 'I', fontsize=10, color=GREEN, fontweight='bold', zorder=11)

    trail_x, trail_y = [], []

    # Legend
    import matplotlib.patches as mpatches
    legend_items = [
        mpatches.Patch(color=PURPLE, label='Target points A–D'),
        mpatches.Patch(color=ORANGE, label='Fixed pivots E, F / cranks EG, HF'),
        mpatches.Patch(color=GRAY,   label='Coupler GH'),
        mpatches.Patch(color=GREEN,  label='Coupler point I'),
        mpatches.Patch(color=AMBER,  label='Coupler curve'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=8,
              framealpha=0.85, edgecolor='#ccc')

    def init():
        line_EG.set_data([], []); line_GH.set_data([], [])
        line_HF.set_data([], []); line_GI.set_data([], [])
        line_HI.set_data([], []); trail_line.set_data([], [])
        dot_G.set_offsets(np.empty((0, 2)))
        dot_H.set_offsets(np.empty((0, 2)))
        dot_I.set_offsets(np.empty((0, 2)))
        return (line_EG, line_GH, line_HF, line_GI, line_HI,
                trail_line, dot_G, dot_H, dot_I, lbl_G, lbl_H, lbl_I)

    def update(frame_idx):
        _, G, H, I = frames[frame_idx]

        # Links
        line_EG.set_data([E[0], G[0]], [E[1], G[1]])
        line_GH.set_data([G[0], H[0]], [G[1], H[1]])
        line_HF.set_data([H[0], F[0]], [H[1], F[1]])
        line_GI.set_data([G[0], I[0]], [G[1], I[1]])
        line_HI.set_data([H[0], I[0]], [H[1], I[1]])

        # Trail
        trail_x.append(I[0]); trail_y.append(I[1])
        trail_line.set_data(trail_x, trail_y)

        # Dots
        dot_G.set_offsets([[G[0], G[1]]])
        dot_H.set_offsets([[H[0], H[1]]])
        dot_I.set_offsets([[I[0], I[1]]])

        # Labels (offset slightly)
        lbl_G.set_position((G[0] + 6, G[1] + 6))
        lbl_H.set_position((H[0] + 6, H[1] + 6))
        lbl_I.set_position((I[0] + 7, I[1] + 7))

        return (line_EG, line_GH, line_HF, line_GI, line_HI,
                trail_line, dot_G, dot_H, dot_I, lbl_G, lbl_H, lbl_I)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, interval=1000 // fps,
        blit=True,
    )

    # Save as GIF via temp file (pillow writer requires a real file path)
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tf:
        tmp_path = tf.name
    try:
        ani.save(tmp_path, writer='pillow', fps=fps)
        plt.close(fig)
        with open(tmp_path, 'rb') as f:
            buf = io.BytesIO(f.read())
        buf.seek(0)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Four-Bar Linkage Synthesizer",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Four-Bar Linkage Path Synthesis")
st.markdown(
    "Enter **4 coupler-point positions** (A, B, C, D). "
    "The app finds a four-bar linkage whose tracer point **I** passes through all four points, "
    "then animates the mechanism."
)

# ── Sidebar: Input ────────────────────────────────────────────────────────────
st.sidebar.header("📌 Input — 4 Prescribed Points")
st.sidebar.markdown("Enter the x, y coordinates for each point (any unit).")

defaults = {
    'A': (200.0, 180.0),
    'B': (370.0, 140.0),
    'C': (490.0, 230.0),
    'D': (310.0, 310.0),
}

points = []
for label, (dx, dy) in defaults.items():
    st.sidebar.subheader(f"Point {label}")
    col1, col2 = st.sidebar.columns(2)
    x = col1.number_input(f"x_{label}", value=dx, step=1.0,
                           format="%.2f", key=f"x{label}")
    y = col2.number_input(f"y_{label}", value=dy, step=1.0,
                           format="%.2f", key=f"y{label}")
    points.append([x, y])

st.sidebar.divider()
st.sidebar.subheader("⚙️ Solver Settings")
n_restarts = st.sidebar.slider(
    "Restarts (more = better, slower)",
    min_value=50, max_value=500, value=200, step=50,
)
fps = st.sidebar.slider("Animation FPS", min_value=10, max_value=60, value=30, step=5)
n_frames = st.sidebar.slider("Animation frames", min_value=60, max_value=240, value=120, step=20)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999,
                                value=42, step=1)

run = st.sidebar.button("▶  Synthesize & Animate", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.subheader("Input Points")
    import pandas as pd
    df = pd.DataFrame(points, columns=["x", "y"],
                       index=["A", "B", "C", "D"])
    st.dataframe(df, use_container_width=True)

    st.markdown("""
**Linkage topology:**
```
E ──[EG]── G ──[GH]── H ──[HF]── F
                │
                I  ← tracer / coupler point
```
- **E, F** — fixed ground pivots
- **G, H** — moving pivots (revolute joints)
- **I**    — coupler tracer point (must pass through A, B, C, D)
""")

with col_right:
    if not run:
        st.info("👈  Set the four point coordinates in the sidebar, then click **Synthesize & Animate**.")

if run:
    with st.spinner("Synthesising linkage… (this may take 10–30 s depending on restarts)"):
        sol = synthesize(points, n_restarts=int(n_restarts), seed=int(seed))

    if sol is None:
        st.error("❌ Optimisation failed. Try different point positions or increase restarts.")
        st.stop()

    errors = verify(points, sol)
    rms    = sol['rms_error']

    # ── Results table ─────────────────────────────────────────────────────────
    with col_left:
        st.subheader("Results — 9 Output Values")
        res_data = {
            "Parameter": ["x_E", "y_E", "x_F", "y_F", "EG", "GI", "IH", "HG", "HF"],
            "Value":     [
                sol['E'][0], sol['E'][1],
                sol['F'][0], sol['F'][1],
                sol['EG'],   sol['GI'],   sol['IH'],
                sol['GH'],   sol['HF'],
            ],
        }
        res_df = pd.DataFrame(res_data)
        res_df["Value"] = res_df["Value"].map(lambda v: f"{v:.5f}")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.subheader("Verification Errors")
        err_df = pd.DataFrame({
            "Point":    list("ABCD"),
            "Error":    [f"{e:.2e}" for e in errors],
            "Status":   ["✅ OK" if e < 1.0 else "⚠️ High" for e in errors],
        })
        st.dataframe(err_df, use_container_width=True, hide_index=True)

        status_color = "green" if rms < 1.0 else "orange"
        st.markdown(
            f"**RMS error:** :{status_color}[{rms:.4e}] (same units as input)"
        )

        st.subheader("Crank Angles at A–D")
        angle_df = pd.DataFrame({
            "Point":      list("ABCD"),
            "Angle (rad)": [f"{t:.5f}" for t in sol['thetas']],
            "Angle (deg)": [f"{np.degrees(t):.3f}°" for t in sol['thetas']],
        })
        st.dataframe(angle_df, use_container_width=True, hide_index=True)

    # ── Animation ─────────────────────────────────────────────────────────────
    with col_right:
        st.subheader("Animation")
        with st.spinner("Rendering animation…"):
            gif_buf = build_animation(
                points, sol,
                n_frames=int(n_frames),
                fps=int(fps),
            )
        if gif_buf is None:
            st.warning("⚠️ Could not render animation — linkage may not have a full rotation range.")
        else:
            st.image(gif_buf, caption="Four-bar linkage in motion — coupler point I traces the path",
                     use_container_width=True)
            st.download_button(
                label="⬇️  Download GIF",
                data=gif_buf,
                file_name="fourbar_linkage.gif",
                mime="image/gif",
            )

    st.divider()
    with st.expander("📐 Full parameter details"):
        st.json({
            "E": sol['E'].tolist(),
            "F": sol['F'].tolist(),
            "EG": float(sol['EG']),
            "HF": float(sol['HF']),
            "GH": float(sol['GH']),
            "GI": float(sol['GI']),
            "IH": float(sol['IH']),
            "px": float(sol['px']),
            "py": float(sol['py']),
            "thetas_deg": [float(np.degrees(t)) for t in sol['thetas']],
            "rms_error": float(rms),
        })
