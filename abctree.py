# -*- coding: utf-8 -*-
"""
ABCtree —— 任务生成器 A 与双求解器 B/C 的三方对抗训练

结构
    A（上层，无外部输入）: 可学习码本 z_k + 噪声 eps -> 共享主干 -> 两个 head
        输出一份任务 (x, y*)
        x   任务必备参数，交给 B、C
        y*  该任务的预期输出，不交给 B、C
    B、C（下层）: 逐参数同形状的 MLP，初始参数与学习率不同，各自回归 y*

奖励
    r_B = -L_B ,  r_C = -L_C ,  r_A = r_B - r_C = L_C - L_B
    A 最大化 r_A，等价于最小化 L_B - L_C，即生成「B 已经会、C 还不会」的任务

两段反向
    段一  A 断联：任务 detach 后喂给 B、C，各自按经典监督信号反传、更新自己的参数
    段二  B、C 冻结：只对 A 的任务参数 x 求 ∂L_B/∂x 与 ∂L_C/∂x，合并为
          (∂L_B - ∂L_C)/∂x 后回传到 A；B、C 的参数不在该图内，故一个字节都不动

    为什么 y* 默认 detach：若让梯度经 y* 回传，A 可以把自己的答案直接设成 B 当前的
    输出，于是 L_B 恒为 0、L_C 保持最大，r_A 立刻被这个退化解刷满，A 塌缩成
    「照抄 B」而不再构造任务。默认只经 x 回传；要观察这个退化解时用 --grad-via-label。

判据（compare 模式）
    实验组（A 训练） vs 对照组（A 冻结）：
        随机任务下 B、C 的 gap 只来自两者 lr / 初始化的差别，是基线；
        若实验组的 gap 显著大于基线，说明 A 主动找到了 B 的优势区而非碰巧。

用法
    python abctree.py --mode selfcheck
    python abctree.py --mode train  --steps 3000 --out runs/train
    python abctree.py --mode compare --steps 3000 --seeds 0 1 2
    python abctree.py --mode plot --out runs/train
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    d_code: int = 8            # 码本维度
    d_noise: int = 8           # 每步噪声维度，给 A 提供连续的任务流
    d_hidden: int = 64
    solver_hidden: int = 0     # B/C 的宽度；0 = 与 d_hidden 相同
    solver_layers: int = 2     # B/C 的隐层数（1 = 直接 输入->隐层->输出）
    fixed_rule: bool = False   # True: y* 由固定随机规则从 x 决定，A 只能选 x 的位置
    d_x: int = 8               # 任务必备参数维度（grid 模式）
    d_y: int = 8               # 任务预期输出维度
    task: str = "grid"         # grid: A 输出 8 维任务参数 | prime: A 输出 a,b,c，c 为 B/C 输入
    n_max: int = 1000          # prime 模式：a,b ∈ [2, n_max]，c ∈ [2, n_max^2]
    prime_w: float = 1.0       # prime 惩罚权重
    prime_proj: bool = False   # True: a,b 硬投影到最近质数后再交给 B/C
    reward: str = "diff"       # diff | gate | ramp
    gate_tau: float = 0.02     # gate/ramp：L < gate_tau 视为「通关」
    gate_temp: float = 0.005   # 通关判据的软化温度
    ramp_lam0: float = 0.0     # ramp：C 通关惩罚的初始权重
    ramp_lam1: float = 2.0     # ramp：C 通关惩罚的末态权重（线性升到它）
    div_w: float = 0.0         # 抗塌缩权重（0 = 关）
    div_lo: float = 0.15       # 任务参数逐维 std 的下限（低于它算塌缩）
    div_hi: float = 0.55       # 上限（高于它是滑到 tanh 饱和，属假离散）
    n_codes: int = 64
    batch: int = 64
    steps: int = 3000
    lr_A: float = 1e-3
    lr_B: float = 1e-3
    lr_C: float = 2e-4
    seed: int = 0              # A 用 seed，B 用 seed+1，C 用 seed+2
    grad_clip: float = 1.0
    grad_via_label: bool = False   # 退化解开关，见文件头
    freeze_A: bool = False
    success_thr: float = 0.02  # L < success_thr 视为「这一题会做」
    fail_thr: float = 0.06     # L > fail_thr 视为「这一题不会做」
    log_every: int = 25
    eval_every: int = 100
    device: str = "cuda"
    out: str = "runs/train"


# ---------------------------------------------------------------- 模型

_PRIME_CACHE = {}


def prime_table(n_max: int, device):
    """[2, n_max] 内全部质数，升序。"""
    key = (int(n_max), str(device))
    if key not in _PRIME_CACHE:
        sieve = torch.ones(n_max + 1, dtype=torch.bool)
        sieve[:2] = False
        for i in range(2, int(n_max ** 0.5) + 1):
            if bool(sieve[i]):
                sieve[i * i::i] = False
        _PRIME_CACHE[key] = torch.nonzero(sieve).squeeze(1).to(device)
    return _PRIME_CACHE[key]


def nearest_prime_gap(v: torch.Tensor, primes: torch.Tensor):
    """v: (N,) 连续值 -> (到最近质数的距离, 那个质数)。

    距离函数是锯齿状的（a.e. 梯度 ±1），不是零梯度：它把 a 往最近的质数上拽，
    但落在质数上之后沿格点没有任何驱动力（这一步是本实验的关键性质，不是缺陷）。
    """
    d = (v.unsqueeze(-1) - primes.view(1, -1).to(v.dtype)).abs()
    m, idx = d.min(dim=-1)
    return m, primes[idx]


def solver_in_dim(cfg: Config) -> int:
    return 1 if cfg.task == "prime" else cfg.d_x


class TaskGenerator(nn.Module):
    """A：无外部输入，输出任务。

    grid  : (x ∈ R^d_x, y* ∈ R^d_y)，x 交给 B、C
    prime : 三个数 (a, b, c) + y*，a,b ∈ [2,n_max] 要求为质数、c 要求等于 a·b，
            c 是交给 B、C 的唯一输入
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_codes = cfg.n_codes
        self.d_noise = cfg.d_noise
        self.task = cfg.task
        self.n_max = cfg.n_max
        self.proj = cfg.prime_proj
        self.codes = nn.Parameter(torch.randn(cfg.n_codes, cfg.d_code) * 0.5)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.d_code + cfg.d_noise, cfg.d_hidden), nn.Tanh(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden), nn.Tanh(),
        )
        if cfg.task == "prime":
            self.head_n = nn.Linear(cfg.d_hidden, 3)
        else:
            self.head_x = nn.Linear(cfg.d_hidden, cfg.d_x)
        self.fixed_rule = cfg.fixed_rule
        if cfg.task == "prime" or not cfg.fixed_rule:
            self.head_y = nn.Linear(cfg.d_hidden, cfg.d_y)
        if cfg.fixed_rule:
            # 固定规则：y* = tanh(x @ W_rule)。A 只能选 x 落在哪，不能改目标的形状。
            # 这是切断「A 把 y* 压到低维流形以迁就 B」那条退路的唯一办法。
            W = torch.randn(cfg.d_x, cfg.d_y)
            nn.init.orthogonal_(W)
            self.register_buffer("W_rule", W)

    def forward(self, idx: torch.Tensor, eps: torch.Tensor):
        h = self.trunk(torch.cat([self.codes[idx], eps], dim=-1))
        if self.task != "prime":
            x = torch.tanh(self.head_x(h))
            if self.fixed_rule:
                return x, torch.tanh(x @ self.W_rule), h, None
            return x, torch.tanh(self.head_y(h)), h, None

        y = torch.tanh(self.head_y(h))

        r = torch.sigmoid(self.head_n(h))
        a = 2.0 + (self.n_max - 2.0) * r[:, 0]
        b = 2.0 + (self.n_max - 2.0) * r[:, 1]
        c = 2.0 + (self.n_max ** 2 - 2.0) * r[:, 2]
        if self.proj:
            # 硬投影 + 直通估计器：前向取最近质数（保证 a,b 真是质数、c 真是信号数），
            # 反向按恒等回传。直接 c = p_a * p_b 会把 A 的梯度路径整条切断
            # （质数表是常量，查表结果 requires_grad=False，实测报
            #  "One of the differentiated Tensors does not require grad"）。
            pr = prime_table(self.n_max, a.device)
            _, pa = nearest_prime_gap(a, pr)
            _, pb = nearest_prime_gap(b, pr)
            a = a + (pa.to(a.dtype) - a).detach()
            b = b + (pb.to(b.dtype) - b).detach()
            c = a * b
        # 送给 B、C 的是 c / n_max^2：与 c 信息完全相同（线性单调），但 c 本身到 1e6，
        # 直接进 64 宽 MLP 会让首层 tanh 全饱和、梯度精确为 0（实测过）。
        return (c / self.n_max ** 2).unsqueeze(-1), y, h, (a, b, c)

    def sample(self, bs: int, gen: torch.Generator | None = None, device=None):
        dev = device if device is not None else self.codes.device
        idx = torch.randint(0, self.n_codes, (bs,), generator=gen).to(dev)
        eps = torch.randn(bs, self.d_noise, generator=gen).to(dev)
        return self(idx, eps)


class Solver(nn.Module):
    """B / C：同一份定义，不同 seed 与不同 lr。"""

    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.solver_hidden or cfg.d_hidden
        blocks = []
        d = solver_in_dim(cfg)
        for _ in range(max(cfg.solver_layers, 0)):
            blocks += [nn.Linear(d, h), nn.Tanh()]
            d = h
        blocks.append(nn.Linear(d, cfg.d_y))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def build(cfg: Config, device):
    torch.manual_seed(cfg.seed)
    A = TaskGenerator(cfg).to(device)
    torch.manual_seed(cfg.seed + 1)
    B = Solver(cfg).to(device)
    torch.manual_seed(cfg.seed + 2)
    C = Solver(cfg).to(device)
    return A, B, C


# ---------------------------------------------------------------- 单步

def prime_penalty(A, aux):
    """A 的可满足性惩罚：a,b 偏离最近质数的距离 + c 与 a·b 的偏差。

    两项都按各自输出区间的宽度归一，所以都在 O(1) 量级、不随 n_max 变化。
    """
    a, b, c = aux
    pr = prime_table(A.n_max, a.device)
    ga, _ = nearest_prime_gap(a, pr)
    gb, _ = nearest_prime_gap(b, pr)
    dc = (c - a.to(c.dtype) * b.to(c.dtype)).abs()
    P = (ga + gb) / A.n_max + dc / (A.n_max ** 2)
    return P, ga, gb, dc


def train_step(cfg, A, B, C, optA, optB, optC, gen, frac=0.0):
    dev = A.codes.device
    prime = cfg.task == "prime"

    # 段一：A 断联，B、C 各按经典监督更新
    x1, y1, _, _ = A.sample(cfg.batch, gen, dev)
    x1, y1 = x1.detach(), y1.detach()
    LB1 = F.mse_loss(B(x1), y1)
    LC1 = F.mse_loss(C(x1), y1)
    optB.zero_grad(set_to_none=True)
    LB1.backward()
    optB.step()
    optC.zero_grad(set_to_none=True)
    LC1.backward()
    optC.step()

    rec = {"LB1": LB1.item(), "LC1": LC1.item()}

    # 段二：冻结 B、C，梯度合并后回传 A（+ 约束惩罚）
    if optA is not None:
        x2, y2, _, aux = A.sample(cfg.batch, gen, dev)
        tgt = y2 if cfg.grad_via_label else y2.detach()
        lb_i = F.mse_loss(B(x2), tgt, reduction="none").mean(1)
        lc_i = F.mse_loss(C(x2), tgt, reduction="none").mean(1)
        LB2, LC2 = lb_i.mean(), lc_i.mean()
        gB = torch.autograd.grad(LB2, x2, retain_graph=True)[0]
        gC = torch.autograd.grad(LC2, x2, retain_graph=True)[0]
        if cfg.reward == "ramp":
            # r_A = passB · (1 − λ(t)·passC)，λ 随时间线性升。
            # 乘性写法把「优先保证 B 通关」变成结构约束：B 不通关时 r_A ≈ 0，
            # 无论 C 多失败都拿不到分；C 通关的惩罚则越来越重，A 必须持续找新盲区。
            lam = cfg.ramp_lam0 + (cfg.ramp_lam1 - cfg.ramp_lam0) * min(max(frac, 0.0), 1.0)
            pB = torch.sigmoid((cfg.gate_tau - lb_i) / cfg.gate_temp)
            pC = torch.sigmoid((cfg.gate_tau - lc_i) / cfg.gate_temp)
            r = pB * (1.0 - lam * pC)
            loss_A = -r.mean()
            rec["reward"] = float(r.detach().mean())
            rec["gate"] = float(pB.detach().mean())
            rec["passB"] = float(pB.detach().mean())
            rec["passC"] = float(pC.detach().mean())
            rec["lam"] = lam
        elif cfg.reward == "gate":
            # 你的推理：C 变强就该把 A 推到更难的题。这要求 A 不能靠「把题变简单」拿分，
            # 所以不用差值，改成「只在 B 会做的题上计 C 的失败」：
            #   r_A = L_C · sigmoid((tau - L_B)/temp)
            # 平凡题 L_C≈0 拿不到分；超纲题门≈0 也拿不到分；只有 B 会、C 不会的带内得分。
            gate = torch.sigmoid((cfg.gate_tau - lb_i) / cfg.gate_temp)
            loss_A = -(lc_i * gate).mean()
            rec["reward"] = float((lc_i * gate).detach().mean())
            rec["gate"] = gate.mean().item()
        else:
            loss_A = LB2 - LC2
            rec["reward"] = (LC2 - LB2).item()
            rec["gate"] = 1.0
        if prime:
            P, ga, gb, dc = prime_penalty(A, aux)
            loss_A = loss_A + cfg.prime_w * P.mean()
            rec.update(P=P.mean().item(), ga=ga.mean().item(), gb=gb.mean().item(),
                       dc=dc.mean().item(),
                       a=aux[0].mean().item(), b=aux[1].mean().item(),
                       c=aux[2].mean().item(),
                       sat=float(((ga < 0.5) & (gb < 0.5)
                                  & (dc < 1e-3 * cfg.n_max ** 2)).float().mean()))
        # 通关率无论哪种奖励都记一笔，便于横向比较
        if "passB" not in rec:
            rec["passB"] = float(torch.sigmoid((cfg.gate_tau - lb_i)
                                               / cfg.gate_temp).detach().mean())
            rec["passC"] = float(torch.sigmoid((cfg.gate_tau - lc_i)
                                               / cfg.gate_temp).detach().mean())
            rec["lam"] = 0.0
        if cfg.div_w > 0:
            # 抗塌缩：逐维 std 低于 div_lo 算塌缩，高于 div_hi 是滑到 tanh 饱和区
            # （饱和区的「离散」是假的：梯度消失，学不到东西）。
            sdim = x2.std(0)
            dterm = F.relu(cfg.div_lo - sdim).mean() + F.relu(sdim - cfg.div_hi).mean()
            loss_A = loss_A + cfg.div_w * dterm
            rec["div"] = float(dterm.detach())
        else:
            rec["div"] = 0.0
        optA.zero_grad(set_to_none=True)
        loss_A.backward()
        gn = torch.nn.utils.clip_grad_norm_(A.parameters(), cfg.grad_clip)
        optA.step()
        rec.update(
            gB=gB.norm().item(),
            gC=gC.norm().item(),
            cos=float(F.cosine_similarity(gB.flatten(), gC.flatten(), dim=0)),
            grad_norm=float(gn),
        )
    else:
        with torch.no_grad():
            x2, y2, _, aux = A.sample(cfg.batch, gen, dev)
            LB2 = F.mse_loss(B(x2), y2)
            LC2 = F.mse_loss(C(x2), y2)
            if prime:
                P, ga, gb, dc = prime_penalty(A, aux)
                rec.update(P=P.mean().item(), ga=ga.mean().item(), gb=gb.mean().item(),
                           dc=dc.mean().item(), a=aux[0].mean().item(),
                           b=aux[1].mean().item(), c=aux[2].mean().item(),
                           sat=float(((ga < 0.5) & (gb < 0.5)
                                      & (dc < 1e-3 * cfg.n_max ** 2)).float().mean()))
        rec.update(gB=0.0, gC=0.0, cos=0.0, grad_norm=0.0, reward=0.0, gate=0.0,
                   div=0.0, passB=0.0, passC=0.0, lam=0.0)

    rec.update(LB=LB2.item(), LC=LC2.item(), gap=(LC2 - LB2).item(),
               x_std=x2.std(0).mean().item(), x_abs=x2.abs().mean().item(),
               y_std=y2.std(0).mean().item(), y_abs=y2.abs().mean().item())
    return rec


def evaluate(cfg, A, B, C, ev_idx, ev_eps):
    with torch.no_grad():
        x, y, _, aux = A(ev_idx, ev_eps)
        lb = F.mse_loss(B(x), y, reduction="none").mean(1)
        lc = F.mse_loss(C(x), y, reduction="none").mean(1)
    only_b = ((lb < cfg.success_thr) & (lc > cfg.fail_thr)).float().mean().item()
    pB = torch.sigmoid((cfg.gate_tau - lb) / cfg.gate_temp)
    pC = torch.sigmoid((cfg.gate_tau - lc) / cfg.gate_temp)
    out = {
        "ev_LB": lb.mean().item(),
        "ev_LC": lc.mean().item(),
        "ev_gap": (lc - lb).mean().item(),
        "ev_onlyB": only_b,
        "ev_passB": pB.mean().item(),
        "ev_passC": pC.mean().item(),
        "ev_x_std": x.std(0).mean().item(),
        "ev_x_abs": x.abs().mean().item(),
        "ev_y_std": y.std(0).mean().item(),
    }
    # 按 |x| 分箱：把「难度上升」和「任务被推到 tanh 边角」分开
    xa = x.abs().mean(dim=1)
    lo, hi = xa < 0.7, xa >= 0.7
    out["ev_LB_lo"] = lb[lo].mean().item() if bool(lo.any()) else float("nan")
    out["ev_LB_hi"] = lb[hi].mean().item() if bool(hi.any()) else float("nan")
    out["ev_n_lo"] = float(lo.float().mean())
    out["ev_n_hi"] = float(hi.float().mean())
    if cfg.task == "prime":
        a, b, c = aux
        pr = prime_table(cfg.n_max, a.device)
        ga, _ = nearest_prime_gap(a, pr)
        gb, _ = nearest_prime_gap(b, pr)
        dc = (c - a * b).abs()
        # 输入域宽度：c 覆盖了多宽的范围（对数尺度）、占了多少个 log 桶
        lc_ = torch.log10(c.clamp_min(2.0))
        hist = torch.histc(lc_, bins=40, min=0.0, max=6.0)
        out.update(
            ev_sat=float(((ga < 0.5) & (gb < 0.5)
                          & (dc < 1e-3 * cfg.n_max ** 2)).float().mean()),
            ev_a=a.mean().item(), ev_b=b.mean().item(), ev_c=c.mean().item(),
            ev_c_med=c.median().item(),
            ev_c_lo=c.min().item(), ev_c_hi=c.max().item(),
            ev_c_log_width=float(lc_.max() - lc_.min()),
            ev_c_bins=int((hist > 0).sum()),
            ev_y_spread=float(y.std(0).mean()),
        )
    return out


# ---------------------------------------------------------------- 训练

def run_training(cfg: Config, verbose: bool = True, return_models: bool = False):
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    A, B, C = build(cfg, dev)
    optA = None if cfg.freeze_A else torch.optim.Adam(A.parameters(), lr=cfg.lr_A)
    optB = torch.optim.Adam(B.parameters(), lr=cfg.lr_B)
    optC = torch.optim.Adam(C.parameters(), lr=cfg.lr_C)

    ge = torch.Generator().manual_seed(20241024)
    ev_idx = torch.randint(0, cfg.n_codes, (512,), generator=ge).to(dev)
    ev_eps = torch.randn(512, cfg.d_noise, generator=ge).to(dev)

    gen = torch.Generator().manual_seed(cfg.seed + 1000)
    rows, evals = [], []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        rec = train_step(cfg, A, B, C, optA, optB, optC, gen,
                         frac=(step - 1) / max(cfg.steps - 1, 1))
        rec["step"] = step
        if step % cfg.log_every == 0 or step == 1:
            rows.append(rec)
            if verbose and step % (cfg.log_every * 4) == 0:
                print(f"  step {step:5d}  LB={rec['LB']:.4f} LC={rec['LC']:.4f} "
                      f"gap={rec['gap']:+.4f} xstd={rec['x_std']:.3f}")
        if step % cfg.eval_every == 0 or step == cfg.steps:
            ev = evaluate(cfg, A, B, C, ev_idx, ev_eps)
            ev["step"] = step
            evals.append(ev)
    wall = time.time() - t0

    final = evaluate(cfg, A, B, C, ev_idx, ev_eps)
    final.update(
        freeze_A=cfg.freeze_A,
        seed=cfg.seed,
        lr_B=cfg.lr_B,
        lr_C=cfg.lr_C,
        steps=cfg.steps,
        wall_sec=round(wall, 2),
        steps_per_sec=round(cfg.steps / wall, 1),
        codes_std=A.codes.detach().std(0).mean().item(),
    )

    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(cfg.out, "evals.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(evals[0].keys()))
        w.writeheader()
        w.writerows(evals)
    with open(os.path.join(cfg.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    with open(os.path.join(cfg.out, "final.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    if return_models:
        return final, A, B, C
    return final


# ---------------------------------------------------------------- 自检

def _fd_check(cfg: Config):
    """两层有限差分校验（float64）。

    校验一：合并公式本身。对任务参数 x 独立做差分，比较 ∂(L_B-L_C)/∂x 与
            autograd 得到的 gB - gC，确认「一正一反相加」写对了。
    校验二：A 的参数梯度。由于 y* 在设计上是 detach 的标签，差分时必须把 y* 钉在
            基点的取值上（否则差分会把 y* 也随 θ 变化的那一部分算进去，与实现不同）。
            这个函数测的正是 A 实际收到的那条梯度路径。
    """
    torch.manual_seed(7)
    A = TaskGenerator(cfg).double()
    torch.manual_seed(8)
    B = Solver(cfg).double()
    torch.manual_seed(9)
    C = Solver(cfg).double()
    ge = torch.Generator().manual_seed(2024)
    idx = torch.randint(0, cfg.n_codes, (32,), generator=ge)
    eps = torch.randn(32, cfg.d_noise, generator=ge)

    x, y, _, _ = A(idx, eps)
    y_fix = y.detach()

    # ---- 校验一：x 层的合并符号
    LB = F.mse_loss(B(x), y_fix)
    LC = F.mse_loss(C(x), y_fix)
    gB = torch.autograd.grad(LB, x, retain_graph=True)[0]
    gC = torch.autograd.grad(LC, x, retain_graph=True)[0]
    merged = (gB - gC)

    h = 1e-6
    # 差分得到的是 batch 平均损失的导数，与 merged 在 batch 维取平均后比较
    fd_x = torch.zeros(x.shape[1], dtype=merged.dtype)
    for j in range(x.shape[1]):
        with torch.no_grad():
            xp = x.clone(); xp[:, j] += h
            lp = F.mse_loss(B(xp), y_fix) - F.mse_loss(C(xp), y_fix)
            xm = x.clone(); xm[:, j] -= h
            lm = F.mse_loss(B(xm), y_fix) - F.mse_loss(C(xm), y_fix)
        fd_x[j] = (lp - lm) / (2 * h)
    # 差分同时扰动 batch 内每一行，故一阶量是各样本梯度之和
    an_x = merged.sum(0)
    rel_x = (an_x - fd_x).abs().max().item() / max(fd_x.abs().max().item(), 1e-12)

    # ---- 校验二：A 的参数梯度（y* 钉在基点）
    def obj():
        xa, _, _, _ = A(idx, eps)
        return F.mse_loss(B(xa), y_fix) - F.mse_loss(C(xa), y_fix)

    A.zero_grad(set_to_none=True)
    obj().backward()
    p = A.trunk[0].weight
    an = p.grad.view(-1)[0].item()
    with torch.no_grad():
        p.view(-1)[0] += h
        lp = obj().item()
        p.view(-1)[0] -= 2 * h
        lm = obj().item()
        p.view(-1)[0] += h
    fd = (lp - lm) / (2 * h)
    rel_a = abs(an - fd) / max(abs(fd), 1e-12)
    return rel_x, rel_a, an, fd


def _fd_check_penalty(cfg: Config):
    """质数惩罚的梯度校验（float64）。

    距离函数是锯齿状的，微小扰动若恰好跨过两个质数的中点，差分本身就不稳定。
    所以先用两个步长确认 fd 稳定，再与解析梯度比；不稳定时只要求梯度非零。
    """
    torch.manual_seed(21)
    A = TaskGenerator(cfg).double()
    ge = torch.Generator().manual_seed(77)
    idx = torch.randint(0, cfg.n_codes, (16,), generator=ge)
    eps = torch.randn(16, cfg.d_noise, generator=ge)

    def obj():
        _, _, _, aux = A(idx, eps)
        return prime_penalty(A, aux)[0].sum()

    A.zero_grad(set_to_none=True)
    obj().backward()
    p = A.head_n.weight
    an = p.grad.view(-1)[3].item()

    def fd_at(h):
        with torch.no_grad():
            p.view(-1)[3] += h
            lp = obj().item()
            p.view(-1)[3] -= 2 * h
            lm = obj().item()
            p.view(-1)[3] += h
        return (lp - lm) / (2 * h)

    fd1, fd2 = fd_at(1e-7), fd_at(2.5e-8)
    stable = abs(fd1 - fd2) < 1e-3 * max(abs(fd1), 1e-9)
    rel = abs(an - fd1) / max(abs(fd1), 1e-12)
    return rel, fd1, an, stable


def selfcheck(cfg: Config) -> bool:
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ok_all = True

    def report(name, cond, detail=""):
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    print("== 1. 形状 / 超参 ==")
    A, B, C = build(cfg, dev)
    sb = [p.shape for p in B.parameters()]
    sc = [p.shape for p in C.parameters()]
    report("B、C 逐参数同形状", sb == sc, f"{len(sb)} tensors")
    diff = max((pb - pc).abs().max().item() for pb, pc in zip(B.parameters(), C.parameters()))
    report("B、C 初始参数不同", diff > 1e-3, f"max|Δ|={diff:.4f}")
    report("B、C 学习率不同", cfg.lr_B != cfg.lr_C, f"{cfg.lr_B} vs {cfg.lr_C}")
    report("B 输入维度正确，标签不泄漏", B.net[0].in_features == solver_in_dim(cfg),
           f"{B.net[0].in_features}")

    print("== 2. A 的初始任务分布 ==")
    gen = torch.Generator().manual_seed(11)
    x, y, h, aux = A.sample(256, gen, dev)
    xs, ys = x.std(0).mean().item(), y.std(0).mean().item()
    if cfg.task == "prime":
        c = aux[2]
        rel_spread = xs / max(abs(x).mean().item(), 1e-9)
        c_logw = math.log10(max(c.max().item() / c.min().item(), 1.0))
        report("任务参数不塌缩（相对离散度）", rel_spread > 0.01,
               f"std/mean={rel_spread:.4f}  log10(c)宽={c_logw:.3f}")
    else:
        report("任务参数不塌缩", xs > 0.03, f"x_std={xs:.4f}")
    report("预期输出不塌缩", ys > 0.03, f"y_std={ys:.4f}")
    if cfg.task == "prime":
        a, b, c = aux
        report("a,b 落在 [2,n_max]、c 落在 [2,n_max^2]",
               bool((a >= 2).all() and (a <= cfg.n_max).all()
                    and (c >= 2).all() and (c <= cfg.n_max ** 2).all()),
               f"a∈[{a.min():.1f},{a.max():.1f}] c∈[{c.min():.0f},{c.max():.0f}]")
        P, ga, gb, dc = prime_penalty(A, aux)
        if cfg.prime_proj:
            report("硬投影下惩罚按构造为 0", P.abs().max().item() < 1e-12,
                   f"max|P|={P.abs().max():.2e}")
        else:
            report("初始惩罚非零且有限", P.mean().item() > 0 and torch.isfinite(P).all(),
                   f"P={P.mean():.4f}")
    else:
        report("head 未饱和", abs(x).mean().item() < 0.95, f"mean|x|={abs(x).mean():.3f}")

    print("== 3. 段一断联：B、C 更新时 A 不接收梯度 ==")
    optA = torch.optim.Adam(A.parameters(), lr=cfg.lr_A)
    optB = torch.optim.Adam(B.parameters(), lr=cfg.lr_B)
    optC = torch.optim.Adam(C.parameters(), lr=cfg.lr_C)
    A.zero_grad(set_to_none=True)
    B.zero_grad(set_to_none=True)
    C.zero_grad(set_to_none=True)
    pB0 = B.net[0].weight.detach().clone()
    pC0 = C.net[0].weight.detach().clone()
    # 只跑段一
    x1, y1, _, _ = A.sample(cfg.batch, gen, dev)
    x1, y1 = x1.detach(), y1.detach()
    F.mse_loss(B(x1), y1).backward()
    optB.step()
    C.zero_grad(set_to_none=True)
    F.mse_loss(C(x1), y1).backward()
    optC.step()
    report("A 全部参数梯度为空（段一未连图）",
           all(p.grad is None for p in A.parameters()))
    report("B 参数已更新", not torch.equal(pB0, B.net[0].weight))
    report("C 参数已更新", not torch.equal(pC0, C.net[0].weight))

    print("== 4. 段二冻结：只更新 A，B、C 一个字节不动 ==")
    before = [(p.clone(), None if p.grad is None else p.grad.clone()) for p in B.parameters()]
    beforeC = [(p.clone(), None if p.grad is None else p.grad.clone()) for p in C.parameters()]
    x2, y2, _, _ = A.sample(cfg.batch, gen, dev)
    LB2 = F.mse_loss(B(x2), y2.detach())
    LC2 = F.mse_loss(C(x2), y2.detach())
    gB = torch.autograd.grad(LB2, x2, retain_graph=True)[0]
    gC = torch.autograd.grad(LC2, x2, retain_graph=True)[0]
    optA.zero_grad(set_to_none=True)
    x2.backward(gB - gC)
    optA.step()
    same = all(torch.equal(p, b) for p, (b, _) in zip(B.parameters(), before))
    same_c = all(torch.equal(p, b) for p, (b, _) in zip(C.parameters(), beforeC))
    grads_untouched = all(
        (p.grad is None and g is None) or (p.grad is not None and g is not None and torch.equal(p.grad, g))
        for p, (_, g) in list(zip(B.parameters(), before)) + list(zip(C.parameters(), beforeC)))
    report("B 参数逐位不变", same)
    report("C 参数逐位不变", same_c)
    report("B、C 梯度缓冲未被追加", grads_untouched)
    report("C 梯度源非零 ‖gC‖>0", gC.norm().item() > 0, f"{gC.norm().item():.4f}")
    report("A 收到梯度", any(p.grad is not None and p.grad.abs().sum() > 0 for p in A.parameters()))

    print("== 5. 梯度反转符号（有限差分，float64）==")
    rel_x, rel_a, an, fd = _fd_check(cfg)
    report("∂(L_B-L_C)/∂x 与差分一致（合并符号）", rel_x < 1e-6, f"rel={rel_x:.2e}")
    if cfg.prime_proj:
        # STE 的定义就是「前向投影、反向恒等」，两者必然与真实差分不一致，
        # 比 FD 没有意义。这里直接验恒等性。
        r_leaf = torch.tensor([500.0], dtype=torch.float64, requires_grad=True)
        pr = prime_table(cfg.n_max, torch.device("cpu"))
        _, p_near = nearest_prime_gap(r_leaf, pr)
        ste = r_leaf + (p_near.double() - r_leaf).detach()
        ste.sum().backward()
        report("STE：前向取质数、反向恒等",
               abs(ste.item() - p_near.item()) < 1e-12 and r_leaf.grad.item() == 1.0,
               f"fwd {r_leaf.item():.1f}->{ste.item():.1f}, grad={r_leaf.grad.item():.1f}")
        report("∂(L_B-L_C)/∂θ_A 非零（STE 路径）", abs(an) > 0, f"analytic={an:.3e}")
    else:
        report("∂(L_B-L_C)/∂θ_A 与差分一致", rel_a < 1e-5 or abs(an - fd) < 1e-10,
               f"analytic={an:.8f} fd={fd:.8f} rel={rel_a:.2e} |err|={abs(an - fd):.1e}")

    if cfg.task == "prime":
        print("== 5b. 质数惩罚的梯度与零点 ==")
        rel_p, fd_p, an_p, stable = _fd_check_penalty(cfg)
        if stable:
            report("∂P/∂θ_A 与差分一致", rel_p < 1e-4,
                   f"analytic={an_p:.6e} fd={fd_p:.6e} rel={rel_p:.2e}")
        else:
            report("∂P/∂θ_A 非零（差分跨过锯齿拐点，不比值）", abs(an_p) > 0,
                   f"analytic={an_p:.6e} fd={fd_p:.6e}")

        class _Stub:
            n_max = cfg.n_max
        pr = prime_table(cfg.n_max, torch.device("cpu"))
        a0 = pr[10:14].double()
        b0 = pr[20:24].double()
        P0, _, _, _ = prime_penalty(_Stub, (a0, b0, a0 * b0))
        report("合法解（a,b 为质数、c=a·b）处惩罚为 0",
               P0.abs().max().item() < 1e-12, f"max|P|={P0.abs().max():.2e}")
        Pa, _, _, _ = prime_penalty(_Stub, (a0 + 2.5, b0, a0 * b0))
        report("a 偏离质数时惩罚 > 0", Pa.min().item() > 0, f"min|P|={Pa.min():.4f}")
        Pc, _, _, _ = prime_penalty(_Stub, (a0, b0, a0 * b0 * 1.5))
        report("c ≠ a·b 时惩罚 > 0", Pc.min().item() > 0, f"min|P|={Pc.min():.4f}")

    print("== 6. 数值稳定性（20 步）==")
    A2, B2, C2 = build(replace(cfg, seed=99), dev)
    oA = torch.optim.Adam(A2.parameters(), lr=cfg.lr_A)
    oB = torch.optim.Adam(B2.parameters(), lr=cfg.lr_B)
    oC = torch.optim.Adam(C2.parameters(), lr=cfg.lr_C)
    g2 = torch.Generator().manual_seed(5)
    finite = True
    for _ in range(20):
        r = train_step(cfg, A2, B2, C2, oA, oB, oC, g2)
        finite = finite and all(math.isfinite(v) for v in r.values())
        finite = finite and all(torch.isfinite(p).all() for p in A2.parameters())
    report("20 步无 NaN/Inf", finite)

    print("\n" + ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return ok_all


# ---------------------------------------------------------------- 对照实验

def cmd_compare(cfg: Config, seeds):
    base = replace(cfg, out="")
    groups = {"A_frozen": True, "A_trained": False}
    table = {}
    for name, frozen in groups.items():
        res = []
        for s in seeds:
            tag = f"runs/cmp_{name}_s{s}"
            c = replace(base, freeze_A=frozen, seed=s, out=tag)
            print(f"-- {name} seed={s}")
            res.append(run_training(c, verbose=False))
        table[name] = res

    keys = ["ev_LB", "ev_LC", "ev_gap", "ev_onlyB", "ev_x_abs", "steps_per_sec"]
    if cfg.task == "prime":
        keys = ["ev_LB", "ev_gap", "ev_sat", "ev_c_med", "ev_c_log_width",
                "ev_c_bins", "ev_y_spread"]
    print("\n组别             seed  " + "  ".join(f"{k:>10s}" for k in keys))
    summary = {}
    for name, res in table.items():
        for r in res:
            print(f"{name:14s} {r['seed']:5d}  " +
                  "  ".join(f"{r[k]:10.4f}" for k in keys))
        summary[name] = {
            k: {
                "mean": sum(r[k] for r in res) / len(res),
                "std": (sum((r[k] - sum(q[k] for q in res) / len(res)) ** 2 for r in res)
                        / len(res)) ** 0.5,
            }
            for k in keys
        }
    print("\n汇总（mean ± std, n=%d）" % len(seeds))
    for name, s in summary.items():
        print(f"{name:14s} " + "  ".join(
            f"{k}={s[k]['mean']:.4f}±{s[k]['std']:.4f}" for k in ["ev_LB", "ev_LC", "ev_gap", "ev_onlyB"]))
    os.makedirs("runs", exist_ok=True)
    tag = f"lrB{cfg.lr_B:g}_lrC{cfg.lr_C:g}"
    with open(f"runs/compare_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    d = summary["A_trained"]["ev_gap"]["mean"] - summary["A_frozen"]["ev_gap"]["mean"]
    print(f"\nΔgap = gap(A_trained) - gap(A_frozen) = {d:+.4f}")
    return summary


# ---------------------------------------------------------------- 探针：C 到底追不追得上

def cmd_probe(cfg: Config, ft_steps: int, n_task: int = 4096):
    """训练结束后冻结 A 与 B，只让 C 在 A 的最终任务分布上继续学。

    这是区分「只有 B 能完成」和「C 只是跑得慢」的决定性实验：
        gap 在续训中快速归零  -> A 找到的是 C 的学习率追不上的漂移速度，属训练动态差异
        gap 稳住不动          -> 这批任务对 C 是结构性的，才谈得上只有 B 能完成
    """
    print(f"阶段一：常规训练 {cfg.steps} 步")
    final, A, B, C = run_training(replace(cfg, out=cfg.out + "_stage1"), verbose=False,
                                  return_models=True)
    dev = A.codes.device
    gf = torch.Generator().manual_seed(777)
    xs, ys, _, _ = A.sample(n_task, gf, dev)
    xs, ys = xs.detach(), ys.detach()

    def measure():
        with torch.no_grad():
            lb = F.mse_loss(B(xs), ys, reduction="none").mean(1)
            lc = F.mse_loss(C(xs), ys, reduction="none").mean(1)
        onlyb = ((lb < cfg.success_thr) & (lc > cfg.fail_thr)).float().mean().item()
        return lb.mean().item(), lc.mean().item(), onlyb

    lb0, lc0, ob0 = measure()
    gap0 = lc0 - lb0
    print(f"续训前   L_B={lb0:.4f}  L_C={lc0:.4f}  gap={gap0:+.4f}  onlyB={ob0:.3f}")

    optC = torch.optim.Adam(C.parameters(), lr=cfg.lr_C)
    rows = []
    for step in range(1, ft_steps + 1):
        idx = torch.randint(0, n_task, (cfg.batch,), generator=gf).to(dev)
        optC.zero_grad(set_to_none=True)
        F.mse_loss(C(xs[idx]), ys[idx]).backward()
        optC.step()
        if step % max(ft_steps // 8, 1) == 0 or step == ft_steps:
            lb, lc, ob = measure()
            rows.append((step, lb, lc, lc - lb, ob))
            print(f"  ft {step:5d}  L_B={lb:.4f}  L_C={lc:.4f}  gap={lc-lb:+.4f}  onlyB={ob:.3f}")

    lb1, lc1, ob1 = rows[-1][1], rows[-1][2], rows[-1][4]
    gap1 = lc1 - lb1
    print(f"\n续训 {ft_steps} 步后 gap 从 {gap0:+.4f} 变为 {gap1:+.4f}"
          f"（残留 {100 * gap1 / gap0 if abs(gap0) > 1e-9 else float('nan'):.0f}%）")
    out = {
        "gap_before": gap0, "gap_after": gap1,
        "LC_before": lc0, "LC_after": lc1,
        "onlyB_before": ob0, "onlyB_after": ob1,
        "ft_steps": ft_steps, "n_task": n_task,
        "curve": rows,
        "x_abs": xs.abs().mean().item(),
    }
    os.makedirs(cfg.out, exist_ok=True)
    with open(os.path.join(cfg.out, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# ---------------------------------------------------------------- 画图

def cmd_plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(out, "metrics.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(out, "evals.csv"), encoding="utf-8") as f:
        evs = list(csv.DictReader(f))
    st = [int(r["step"]) for r in rows]

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(st, [float(r["LB"]) for r in rows], label="L_B")
    ax[0, 0].plot(st, [float(r["LC"]) for r in rows], label="L_C")
    ax[0, 0].set_title("loss (train tasks)"); ax[0, 0].legend(); ax[0, 0].set_yscale("log")

    ax[0, 1].plot(st, [float(r["gap"]) for r in rows], color="crimson")
    ax[0, 1].axhline(0, ls="--", c="gray"); ax[0, 1].set_title("r_A = L_C - L_B")

    ax[1, 0].plot(st, [float(r["x_std"]) for r in rows], label="std(task x)")
    ax[1, 0].plot(st, [float(r["x_abs"]) for r in rows], label="mean|x| (saturation)")
    ax[1, 0].plot(st, [float(r["y_std"]) for r in rows], label="std(task y*)")
    ax[1, 0].set_title("task spread"); ax[1, 0].legend()

    es = [int(r["step"]) for r in evs]
    ax[1, 1].plot(es, [float(r["ev_LB"]) for r in evs], label="eval L_B")
    ax[1, 1].plot(es, [float(r["ev_LC"]) for r in evs], label="eval L_C")
    ax[1, 1].plot(es, [float(r["ev_onlyB"]) for r in evs], label="only-B rate")
    ax[1, 1].set_title("eval"); ax[1, 1].legend()
    for a in ax.ravel():
        a.set_xlabel("step"); a.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, "curves.png")
    fig.savefig(p, dpi=130)
    print("saved", p)


# ---------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="selfcheck",
                   choices=["selfcheck", "train", "compare", "probe", "plot"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--ft-steps", type=int, default=3000)
    p.add_argument("--task", default="grid", choices=["grid", "prime"])
    p.add_argument("--n-max", type=int, default=1000)
    p.add_argument("--prime-w", type=float, default=1.0)
    p.add_argument("--prime-proj", action="store_true")
    p.add_argument("--reward", default="diff", choices=["diff", "gate", "ramp"])
    p.add_argument("--lam0", type=float, default=0.0)
    p.add_argument("--lam1", type=float, default=2.0)
    p.add_argument("--gate-tau", type=float, default=0.02)
    p.add_argument("--gate-temp", type=float, default=0.005)
    p.add_argument("--div-w", type=float, default=0.0)
    p.add_argument("--div-lo", type=float, default=0.15)
    p.add_argument("--div-hi", type=float, default=0.55)
    p.add_argument("--solver-hidden", type=int, default=0)
    p.add_argument("--solver-layers", type=int, default=2)
    p.add_argument("--fixed-rule", action="store_true")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr-A", type=float, default=1e-3)
    p.add_argument("--lr-B", type=float, default=1e-3)
    p.add_argument("--lr-C", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--grad-via-label", action="store_true")
    p.add_argument("--freeze-A", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/train")
    a = p.parse_args()

    cfg = Config(steps=a.steps, batch=a.batch, lr_A=a.lr_A, lr_B=a.lr_B, lr_C=a.lr_C,
                 seed=a.seed, grad_via_label=a.grad_via_label, freeze_A=a.freeze_A,
                 task=a.task, n_max=a.n_max, prime_w=a.prime_w, prime_proj=a.prime_proj,
                 reward=a.reward, gate_tau=a.gate_tau, gate_temp=a.gate_temp,
                 ramp_lam0=a.lam0, ramp_lam1=a.lam1,
                 div_w=a.div_w, div_lo=a.div_lo, div_hi=a.div_hi,
                 solver_hidden=a.solver_hidden, solver_layers=a.solver_layers,
                 fixed_rule=a.fixed_rule,
                 device=a.device, out=a.out)
    if a.mode == "selfcheck":
        ok = selfcheck(cfg)
        raise SystemExit(0 if ok else 1)
    if a.mode == "train":
        print(f"training {cfg.steps} steps (freeze_A={cfg.freeze_A})")
        r = run_training(cfg)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                          for k, v in r.items()}, indent=2, ensure_ascii=False))
        return
    if a.mode == "compare":
        cmd_compare(cfg, a.seeds)
        return
    if a.mode == "probe":
        cmd_probe(cfg, a.ft_steps)
        return
    if a.mode == "plot":
        cmd_plot(a.out)


if __name__ == "__main__":
    main()
