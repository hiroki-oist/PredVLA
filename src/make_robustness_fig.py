"""LIBERO-CTRL robustness figure for the PredVLA project page.

Numbers are the published per-axis success rates from the LIBERO-CTRL repository
(README table "The same numbers as a table", results/paper/), Sawada & Kasahara,
arXiv:2609.15940. n = 400 rollouts per policy x axis x severity; nominal = the policy's own
2,000 unperturbed rollouts. The simultaneous ("all six at once") condition is omitted here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AXES = ["camera", "lighting", "robot initial pose", "sensor", "actuation", "language"]
LEVELS = ["nominal", "L1", "L2", "L3"]
POL = {  # name: (params, nominal, {axis: [L1, L2, L3]})
    "PredVLA (0.68 M)":       (0.68e6, 79.3, dict(camera=[73.0, 63.7, 63.0], lighting=[78.2, 77.5, 76.8],
                               robot=[73.0, 52.8, 25.8], sensor=[79.2, 78.0, 67.2], actuation=[80.0, 75.2, 62.7],
                               language=[1.8, 0.0, 0.2])),
    "π₀.₅ (4.14 B)":          (4.14e9, 96.2, dict(camera=[91.5, 80.8, 56.0], lighting=[98.2, 98.0, 95.5],
                               robot=[94.0, 87.5, 66.8], sensor=[96.8, 97.0, 87.5], actuation=[98.2, 96.2, 91.5],
                               language=[86.2, 79.8, 80.5])),
    "OpenVLA-OFT (7.54 B)":   (7.54e9, 96.7, dict(camera=[94.0, 87.5, 69.8], lighting=[96.2, 95.2, 95.2],
                               robot=[89.8, 75.2, 44.8], sensor=[97.5, 97.0, 87.2], actuation=[96.2, 95.2, 92.0],
                               language=[90.8, 90.0, 89.8])),
    "SmolVLA (450 M)":        (450e6, 76.3, dict(camera=[63.7, 44.8, 24.8], lighting=[75.2, 73.5, 70.8],
                               robot=[63.2, 46.0, 24.8], sensor=[76.0, 75.5, 43.5], actuation=[74.0, 72.2, 62.7],
                               language=[38.8, 28.7, 21.5])),
    "VLA-JEPA (2.77 B)":      (2.77e9, 97.6, dict(camera=[95.2, 85.5, 66.8], lighting=[98.2, 98.2, 95.2],
                               robot=[98.2, 92.5, 72.8], sensor=[95.0, 72.0, 36.5], actuation=[98.5, 96.0, 92.2],
                               language=[96.0, 94.8, 93.5])),
}
KEY = dict(zip(AXES, ["camera", "lighting", "robot", "sensor", "actuation", "language"]))
STYLE = {"PredVLA (0.68 M)": ("#C8472F", "o", 3.0, 1.0),
         "π₀.₅ (4.14 B)": ("#2F6DB5", "s", 2.0, 0.9),
         "OpenVLA-OFT (7.54 B)": ("#5A8BC9", "D", 2.0, 0.9),
         "SmolVLA (450 M)": ("#8C94A3", "^", 2.0, 0.9),
         "VLA-JEPA (2.77 B)": ("#2B2F36", "v", 2.0, 0.9)}
BG = "#FFFFFF"
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 12,
                     "axes.edgecolor": "#6B7280", "axes.labelcolor": "#2B2F36",
                     "xtick.color": "#2B2F36", "ytick.color": "#2B2F36",
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(15, 9.2), dpi=150)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.78], hspace=0.55, wspace=0.28,
                      left=0.055, right=0.985, top=0.875, bottom=0.09)
# ---- top: absolute success per axis ----
for i, ax_name in enumerate(AXES):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(BG)
    for name, (npar, nom, d) in POL.items():
        col, mk, lw, al = STYLE[name]
        y = [nom] + d[KEY[ax_name]]
        ax.plot(range(4), y, color=col, marker=mk, ms=5 if name.startswith("PredVLA") else 4, lw=lw, alpha=al,
                zorder=3 if name.startswith("PredVLA") else 2, label=name)
    ax.set_xticks(range(4)); ax.set_xticklabels(["nom.", "L1", "L2", "L3"])
    ax.set_ylim(0, 102); ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.set_title(ax_name, fontsize=13, color="#2B2F36", pad=6)
    if i == 0:
        ax.set_ylabel("success rate [%]")
    else:
        ax.tick_params(labelleft=False)
fig.text(0.055, 0.955, "Success rate under controlled single-axis perturbations (LIBERO-CTRL, severity L1 < L2 < L3)",
         fontsize=16, fontweight="bold", color="#2B2F36")
fig.text(0.055, 0.925, "n = 400 paired rollouts per policy × axis × level; “nom.” = the policy’s own 2,000 unperturbed rollouts. "
         "Simultaneous (all-six) condition omitted.", fontsize=11, color="#6B7280")
# ---- bottom: retention at L3 relative to own nominal ----
axb = fig.add_subplot(gs[1, :])
axb.set_facecolor(BG)
names = list(POL)
x = np.arange(len(AXES)); wdt = 0.16
for j, name in enumerate(names):
    npar, nom, d = POL[name]
    ret = [100.0 * d[KEY[a]][2] / nom for a in AXES]
    col, mk, lw, al = STYLE[name]
    bars = axb.bar(x + (j - 2) * wdt, ret, width=wdt, color=col, alpha=0.95 if name.startswith("PredVLA") else 0.8,
                   label=name, zorder=3)
axb.axhline(100, color="#D5D8DE", lw=1, zorder=1)
axb.set_xticks(x); axb.set_xticklabels(AXES, fontsize=12)
axb.set_ylim(0, 112); axb.set_ylabel("retained at L3\n[% of own nominal]")
axb.grid(axis="y", color="#E5E7EB", lw=0.8)
axb.set_title("Robustness relative to each policy’s own nominal score: success at L3 ÷ nominal success",
              fontsize=13, loc="left", color="#2B2F36", pad=8)
axb.legend(ncol=5, frameon=False, fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
fig.savefig("fig_robustness.png", facecolor=BG)
print("saved fig_robustness.png")
# 検算に使う数値
for name, (npar, nom, d) in POL.items():
    vis = np.mean([d[KEY[a]][2] / nom for a in AXES[:5]]) * 100
    print(f"{name:24s} mean L3 retention over 5 non-language axes: {vis:5.1f} %   language L3: {d['language'][2]:5.1f}")
