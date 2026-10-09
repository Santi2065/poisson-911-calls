"""Regenerates the figures and the numbers shown in the README from 911_times.csv.

    pip install numpy scipy pandas matplotlib
    python docs/figures/make_figures.py
"""
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import gamma, norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for f in Path("/usr/share/fonts/lm").glob("lm*10-*.otf"):  # Latin Modern, if installed
    font_manager.fontManager.addfont(str(f))
plt.style.use(HERE / "paper.mplstyle")
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def save(fig, name):
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


df = pd.read_csv(ROOT / "911_times.csv")
BANDS = {"mañana": "Morning (06–12)", "tarde": "Afternoon (12–18)",
         "noche": "Evening (18–24)", "madrugada": "Night (00–06)"}
YEAR = 365.25 * 86400
ALPHA0, BETA0 = 1.0, 1.0  # Gamma prior used in code.ipynb
times = {b: df[b].dropna().to_numpy() for b in BANDS}
gaps = {b: np.diff(t) for b, t in times.items()}
n = {b: len(x) for b, x in gaps.items()}
T = {b: x.sum() for b, x in gaps.items()}
lam = {b: n[b] / T[b] for b in BANDS}
lam_bayes = {b: (ALPHA0 + n[b]) / (BETA0 + T[b]) for b in BANDS}

# ---- Figure 1: counting process and inter-arrival times
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
for (b, lab), c in zip(BANDS.items(), C):
    t = times[b]
    ax1.plot(t / YEAR, np.arange(len(t)) / 1e3, color=c, lw=1, label=lab)
    x = gaps[b]
    x = x[x <= 3000]
    edges = np.arange(-30, 3001, 60)  # timestamps are mostly whole minutes: centre bins on k*60 s
    h, _ = np.histogram(x, bins=edges, density=True)
    ax2.step(edges[:-1] + 30, h, where="mid", color=c, lw=0.9)
ax1.set_xlabel("calendar time $t$ [years]")
ax1.set_ylabel("$N(t)$ [thousands of calls]")
ax1.set_title("(a) Counting process")
ax1.legend(loc="upper left")
ax2.set_yscale("log")
ax2.set_xlabel("time between calls $x$ [s]")
ax2.set_ylabel("density [1/s]")
ax2.set_title("(b) Inter-arrival times, $x \\leq 3000$ s")
fig.tight_layout()
save(fig, "fig1-poisson.svg")

# ---- Figure 2: posterior of the rate and differential entropy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
for (b, lab), c in zip(BANDS.items(), C):
    post = gamma(ALPHA0 + n[b], scale=1 / (BETA0 + T[b]))
    lo, hi = post.ppf([1e-5, 1 - 1e-5])
    grid = np.linspace(lo, hi, 400)
    ax1.fill_between(grid * 1e3, post.pdf(grid) / 1e3, color=c, alpha=0.25, lw=0)
    ax1.plot(grid * 1e3, post.pdf(grid) / 1e3, color=c, lw=1)
    ax1.text(lam_bayes[b] * 1e3, post.pdf(lam_bayes[b]) / 1e3 * 1.04, lab.split(" ")[0],
             ha="center", va="bottom", fontsize=8, color=c)
ax1.set_xlabel("$\\lambda$ [calls per 1000 s]")
ax1.set_ylabel("posterior density")
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.12)
ax1.set_title("(a) Posterior $\\mathrm{Gamma}(\\alpha_0 + n, \\beta_0 + T)$")
order = sorted(BANDS, key=lambda b: lam[b])
for i, b in enumerate(order):
    ax2.plot(1 - np.log(lam[b]), i, "o", color=C[list(BANDS).index(b)], ms=5)
ax2.set_yticks(range(4), [BANDS[b].split(" ")[0] for b in order])
ax2.set_ylim(-0.6, 3.6)
ax2.set_xlabel("$\\hat h = 1 - \\ln \\hat\\lambda$ [nats]")
ax2.set_title("(b) Differential entropy")
ax2.tick_params(right=False)
fig.tight_layout()
save(fig, "fig2-rates.svg")

# ---- Figure 3: TOST, smallest margin delta at which equivalence is declared (alpha = 0.05)
Z95 = norm.ppf(0.95)


def tost(x1, x2, delta):
    l1, l2 = len(x1) / x1.sum(), len(x2) / x2.sum()
    se = np.sqrt(l1 ** 2 / len(x1) + l2 ** 2 / len(x2))  # Var(MLE) = lambda^2 / n
    za, zb = (l1 - l2 + delta) / se, (l1 - l2 - delta) / se
    return za, zb, abs(l1 - l2) + Z95 * se


rows = []
for b1, b2 in combinations(BANDS, 2):
    za, zb, dmin = tost(gaps[b1], gaps[b2], 0.002)
    rows.append((f"{BANDS[b1].split(' ')[0]} vs {BANDS[b2].split(' ')[0]}", za, zb, dmin, "between"))
for b in BANDS:
    x = gaps[b]
    za, zb, dmin = tost(x[:len(x) // 2], x[len(x) // 2:], 0.002)
    rows.append((f"{BANDS[b].split(' ')[0]}: 1st vs 2nd half", za, zb, dmin, "within"))
fig, ax = plt.subplots(figsize=(7.2, 3.0))
for i, (lab, _, _, dmin, kind) in enumerate(rows):
    ax.plot([5e-5, dmin], [i, i], color="#bdbdbd", lw=0.8)
    ax.plot(dmin, i, "o", color=C[0] if kind == "between" else C[2], ms=4.5)
ax.axvline(0.002, color=C[1], ls="--", lw=1)
ax.text(0.0023, 0, "$\\delta = 0.002$ s$^{-1}$\n(margin set by\nthe assignment)", ha="left", va="top",
        fontsize=8, color=C[1])
ax.set_xscale("log")
ax.set_xlim(5e-5, 1e-2)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.invert_yaxis()
ax.set_xlabel("smallest margin $\\delta^*$ for which TOST declares equivalence [s$^{-1}$]")
ax.tick_params(right=False)
save(fig, "fig3-tost.svg")

# ---- numbers for the README tables
print("| Band | n | sum x [s] | lambda_MLE | lambda_Bayes | 95% CrI | h |")
for b, lab in BANDS.items():
    ci = gamma(ALPHA0 + n[b], scale=1 / (BETA0 + T[b])).ppf([0.025, 0.975])
    print(f"| {lab} | {n[b]:,} | {T[b]:,.0f} | {lam[b]:.9f} | {lam_bayes[b]:.9f} | "
          f"[{ci[0]:.6f}, {ci[1]:.6f}] | {1 - np.log(lam[b]):.4f} |")
for b in BANDS:
    x = gaps[b]
    print(f"{b}: span {T[b] / YEAR:.2f} y, median gap {np.median(x):.0f} s, share <=3000 s "
          f"{np.mean(x <= 3000):.4f}, zero gaps {np.mean(x == 0):.3f}, gaps 60000-70000 s "
          f"{np.sum((x >= 6e4) & (x < 7e4))}, mean of gaps <=3000 s {x[x <= 3000].mean():.1f} s")
for lab, za, zb, dmin, kind in rows:
    print(f"| {lab} | {za:.1f} | {zb:.1f} | {dmin:.2e} |")
