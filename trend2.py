# -*- coding: utf-8 -*-
"""难度趋势对照：grid / prime 两种任务族 × 抗塌缩开关，多 seed 摊表。

判据（跑之前写死）：
  「题越来越难」= ev_LB 后段均值显著高于前段（B 在 A 的题上越来越差）。
  同时必须报 ev_x_std：如果 ev_LB 升了但 x_std 也冲到饱和区，那是假离散，不算变难。

用法
    python trend2.py --task grid --div-w 0
    python trend2.py --task grid --div-w 1 --div-lo 0.15 --div-hi 0.55
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import replace

import abctree as m


def summarize(ev_path):
    rows = list(csv.DictReader(open(ev_path, encoding="utf-8")))
    k = max(len(rows) // 6, 3)
    head, tail = rows[:k], rows[-k:]

    def mean(rs, key):
        return sum(float(r[key]) for r in rs) / len(rs)

    return {
        "LB_h": mean(head, "ev_LB"), "LB_t": mean(tail, "ev_LB"),
        "LC_h": mean(head, "ev_LC"), "LC_t": mean(tail, "ev_LC"),
        "gap_h": mean(head, "ev_gap"), "gap_t": mean(tail, "ev_gap"),
        "xs_h": mean(head, "ev_x_std"), "xs_t": mean(tail, "ev_x_std"),
        "xa_h": mean(head, "ev_x_abs"), "xa_t": mean(tail, "ev_x_abs"),
        "pB_h": mean(head, "ev_passB"), "pB_t": mean(tail, "ev_passB"),
        "pC_h": mean(head, "ev_passC"), "pC_t": mean(tail, "ev_passC"),
        "onlyB_t": mean(tail, "ev_onlyB"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="grid", choices=["grid", "prime"])
    ap.add_argument("--reward", default="diff", choices=["diff", "gate", "ramp"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr-a", type=float, default=1e-3)
    ap.add_argument("--lr-c", type=float, default=2e-4)
    ap.add_argument("--div-w", type=float, default=0.0)
    ap.add_argument("--div-lo", type=float, default=0.15)
    ap.add_argument("--div-hi", type=float, default=0.55)
    ap.add_argument("--n-max", type=int, default=1000)
    ap.add_argument("--solver-hidden", type=int, default=0)
    ap.add_argument("--solver-layers", type=int, default=2)
    ap.add_argument("--fixed-rule", action="store_true")
    ap.add_argument("--lam0", type=float, default=0.0)
    ap.add_argument("--lam1", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    tag = a.tag or (f"{a.task}_div{a.div_w:g}_sw{a.solver_hidden}x{a.solver_layers}"
                    f"{'_fr' if a.fixed_rule else ''}_{a.reward}")
    base = m.Config(task=a.task, steps=a.steps, lr_A=a.lr_a, lr_C=a.lr_c,
                    reward=a.reward, ramp_lam0=a.lam0, ramp_lam1=a.lam1,
                    div_w=a.div_w, div_lo=a.div_lo,
                    div_hi=a.div_hi, n_max=a.n_max,
                    solver_hidden=a.solver_hidden, solver_layers=a.solver_layers,
                    fixed_rule=a.fixed_rule, device="cuda")
    npar = sum(p.numel() for p in m.Solver(base).parameters())
    print(f"### task={a.task} div_w={a.div_w:g} solver={a.solver_hidden or 'same'}"
          f"x{a.solver_layers}  params={npar}", flush=True)

    out = {}
    for seed in a.seeds:
        d = f"runs/t2_{tag}_s{seed}"
        print(f"-- seed={seed}", flush=True)
        m.run_training(replace(base, seed=seed, out=d), verbose=False)
        out[seed] = summarize(os.path.join(d, "evals.csv"))

    cs = ["LB_h", "LB_t", "LC_h", "LC_t", "gap_h", "gap_t",
          "pB_h", "pB_t", "pC_h", "pC_t",
          "xs_h", "xs_t", "xa_h", "xa_t", "onlyB_t"]
    print("\nseed " + " ".join(f"{k:>9s}" for k in cs))
    for seed, r in out.items():
        print(f"{seed:4d} " + " ".join(f"{r[k]:9.4f}" for k in cs))

    rs = list(out.values())
    print(f"\nmean ± std (n={len(rs)})")
    for k in cs:
        mu = sum(r[k] for r in rs) / len(rs)
        sd = (sum((r[k] - mu) ** 2 for r in rs) / len(rs)) ** 0.5
        print(f"   {k:>8s} = {mu:9.4f} ± {sd:.4f}")
    dl = sum(r["LB_t"] - r["LB_h"] for r in rs) / len(rs)
    dx = sum(r["xs_t"] - r["xs_h"] for r in rs) / len(rs)
    n_up = sum(1 for r in rs if r["LB_t"] > r["LB_h"])
    print(f"\n ev_LB 变化 = {dl:+.5f}  ({n_up}/{len(rs)} 个 seed 上升)  "
          f"{'⇒ 题变难' if dl > 0 else '⇒ 题变简单'}")
    print(f" x_std 变化 = {dx:+.5f}")


if __name__ == "__main__":
    main()
