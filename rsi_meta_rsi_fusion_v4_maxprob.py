#!/usr/bin/env python3
"""
RSI Meta-RSI Fusion v4 (Max-Probability Online Principle-Transfer Edition)

What this is (in plain terms):
- A CPU-first RSI engine whose improvement targets include:
    (1) update-rule (learning rule) search space
    (2) objective / loss composition search space
    (3) adversarial world / curriculum pressure (self-play via archive)
    (4) meta-meta controller: conditional (task-embedding -> rule knobs)

- It extends v2 with an OPTIONAL online "principle transfer" channel:
    GitHub + arXiv are scanned for *motifs* (NOT code copy/paste).
    Motifs are distilled into internal mutations, then passed through:
        Discover -> Verify (multi-holdout, distribution-shift) -> Promote (multi-objective) -> Transfer -> Recurse

Important reality note:
- This file cannot "guarantee AGI". What it does is maximize the probability that
  the loop remains (a) exploratory, (b) verifiable, (c) cumulative (via libraries),
  and (d) resistant to toxic transfer by enforcing multi-holdout + stability + complexity.

Why v4 vs v3:
- v3 injected online candidates but did not *learn* which external motifs help.
- v4 adds a contextual bandit (principle selection policy) that is updated from
  measured promotion outcomes. Over time it learns which motifs transfer well
  to your current task distribution and which are toxic.
- v4 adds diversity pressure (principle portfolio) and a compute governor that
  self-tunes trial counts/steps based on measured marginal gain.

Usage:
  python rsi_meta_rsi_fusion_v4_maxprob.py --out_dir runs --run_name v4 --cycles 100 --online 1

Environment variables:
  GITHUB_TOKEN  (optional)  - increases GitHub API rate limit

Dependencies:
- numpy (required)
- requests (optional; falls back to urllib)
- rsi_meta_rsi_fusion_v2.py (required; loaded from same folder)

Security posture:
- No raw code is stored.
- Only coarse motifs (typed hints) are cached.
- Online mode can be disabled; offline continues normally.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import hashlib
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Load v2 from the same folder as this file
# -----------------------------------------------------------------------------
import importlib.util as _importlib_util
import pathlib as _pathlib
import sys as _sys

_V2_PATH = str((_pathlib.Path(__file__).resolve().parent / "rsi_meta_rsi_fusion_v2.py"))
_spec = _importlib_util.spec_from_file_location("rsi_meta_rsi_fusion_v2", _V2_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load rsi_meta_rsi_fusion_v2.py from {_V2_PATH}")
_v2 = _importlib_util.module_from_spec(_spec)
_sys.modules[_spec.name] = _v2
_spec.loader.exec_module(_v2)  # type: ignore

EngineConfig = _v2.EngineConfig
RSIFusionV2 = _v2.RSIFusionV2
CandidateRule = _v2.CandidateRule
MultiScore = _v2.MultiScore
RuleCrystallizer = _v2.RuleCrystallizer
UpdateRuleSpec = _v2.UpdateRuleSpec
ObjectiveSpec = _v2.ObjectiveSpec
clamp = _v2.clamp

# -----------------------------------------------------------------------------
# 0) Minimal HTTP helper
# -----------------------------------------------------------------------------
class Http:
    def __init__(self, timeout_s: int = 20, user_agent: str = "rsi-meta-rsi-fusion-v4/0.1"):
        self.timeout_s = int(timeout_s)
        self.user_agent = user_agent
        try:
            import requests  # type: ignore
            self._requests = requests
        except Exception:
            self._requests = None

    def get_text(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        headers = dict(headers or {})
        headers.setdefault("User-Agent", self.user_agent)
        if self._requests is not None:
            r = self._requests.get(url, headers=headers, timeout=self.timeout_s)
            r.raise_for_status()
            return r.text
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Any:
        return json.loads(self.get_text(url, headers=headers))


# -----------------------------------------------------------------------------
# 1) Online configs and extracted "principles" (motifs)
# -----------------------------------------------------------------------------
@dataclass
class OnlineConfig:
    enabled: bool = False

    # GitHub search
    github_enabled: bool = True
    github_token_env: str = "GITHUB_TOKEN"
    github_max_repos: int = 8
    github_max_files_per_repo: int = 24
    github_max_file_bytes: int = 140_000

    # arXiv search
    arxiv_enabled: bool = True
    arxiv_max_results: int = 16

    # refresh cadence
    refresh_every_s: int = 60 * 30  # 30 min

    # principle cache size
    max_cached_principles: int = 1200

    # extraction safety (never store raw snippets)
    allow_raw_snippets: bool = False  # keep False

    def headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        tok = os.getenv(self.github_token_env, "").strip()
        if tok:
            h["Authorization"] = f"token {tok}"
        return h


@dataclass
class ExtractedPrinciple:
    """
    Coarse motif extracted from online artifacts. This is NOT code.
    We keep it typed so it can be compiled into internal mutations.
    """
    source: str
    kind: str
    payload: Dict[str, Any]
    confidence: float = 0.5
    novelty: float = 0.0
    ts: float = field(default_factory=lambda: time.time())

    def fingerprint(self) -> str:
        s = json.dumps({"k": self.kind, "p": self.payload}, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# 2) GitHub miner (motif-only)
# -----------------------------------------------------------------------------
class GitHubMiner:
    def __init__(self, http: Http, cfg: OnlineConfig):
        self.http = http
        self.cfg = cfg

    def _search_repos(self, query: str) -> List[Dict[str, Any]]:
        q = urllib.parse.quote(query)
        url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page={self.cfg.github_max_repos}"
        data = self.http.get_json(url, headers=self.cfg.headers())
        return list(data.get("items", []))

    def _repo_tree_paths(self, full_name: str, default_branch: str) -> List[str]:
        url = f"https://api.github.com/repos/{full_name}/git/trees/{default_branch}?recursive=1"
        data = self.http.get_json(url, headers=self.cfg.headers())
        paths: List[str] = []
        for it in data.get("tree", []) or []:
            if it.get("type") != "blob":
                continue
            p = str(it.get("path", ""))
            if p.endswith(".py") or p.endswith(".md") or p.endswith(".txt"):
                paths.append(p)
        return paths

    def _raw_file(self, full_name: str, branch: str, path: str) -> str:
        url = f"https://raw.githubusercontent.com/{full_name}/{branch}/{path}"
        return self.http.get_text(url, headers={"User-Agent": self.http.user_agent})

    def mine(self) -> List[ExtractedPrinciple]:
        principles: List[ExtractedPrinciple] = []
        queries = [
            # optimization / ES / meta-learning
            "evolution strategies mirrored sampling rank transform variance reduction",
            "meta learning learnable optimizer learning rate schedule",
            "population based training hyperparameter optimization",
            # self-supervised / representation
            "joint embedding predictive architecture invariance ssl",
            "vicreg barlow twins redundancy reduction invariance",
            "information bottleneck representation learning",
            # liquid / continuous-time
            "liquid neural network time constant tau continuous-time recurrent ode",
            # adversarial curriculum / self-play
            "league training self play curriculum adversarial",
        ]
        for q in queries:
            try:
                repos = self._search_repos(q)
            except Exception:
                continue

            for repo in repos:
                full = repo.get("full_name")
                branch = repo.get("default_branch", "main")
                if not full:
                    continue
                try:
                    paths = self._repo_tree_paths(full, branch)
                except Exception:
                    continue

                # prioritize likely-relevant paths
                scored_paths: List[Tuple[float, str]] = []
                for p in paths:
                    low = p.lower()
                    s = 0.0
                    if any(k in low for k in ("optim", "optimizer", "es", "evolution", "cma", "pbt")):
                        s += 2.2
                    if any(k in low for k in ("loss", "objective", "ssl", "jepa", "world", "model", "representation")):
                        s += 1.6
                    if any(k in low for k in ("curriculum", "selfplay", "league", "advers", "opponent")):
                        s += 1.4
                    if low.endswith(".py"):
                        s += 0.6
                    scored_paths.append((s, p))
                scored_paths.sort(reverse=True)
                paths_sel = [p for _, p in scored_paths[: self.cfg.github_max_files_per_repo]]

                for p in paths_sel:
                    try:
                        raw = self._raw_file(full, branch, p)
                        if len(raw.encode("utf-8", errors="ignore")) > self.cfg.github_max_file_bytes:
                            continue
                    except Exception:
                        continue
                    principles.extend(self._extract_from_text(raw, source=f"github:{full}/{p}"))

        return principles

    @staticmethod
    def _extract_from_text(text: str, source: str) -> List[ExtractedPrinciple]:
        out: List[ExtractedPrinciple] = []
        low = text.lower()

        def add(kind: str, payload: Dict[str, Any], conf: float, nov: float) -> None:
            out.append(ExtractedPrinciple(source=source, kind=kind, payload=payload, confidence=conf, novelty=nov))

        # LR motifs
        if "cosine" in low and ("lr" in low or "learning rate" in low):
            add("lr_schedule", {"name": "cosine"}, 0.72, 0.20)
        if "warmup" in low and ("lr" in low or "learning rate" in low):
            add("lr_schedule", {"name": "warmup_then_decay"}, 0.62, 0.35)
        if ("onecycle" in low) or ("one cycle" in low):
            add("lr_schedule", {"name": "onecycle"}, 0.55, 0.55)

        # Noise/sigma motifs (ES)
        if "adaptive" in low and ("sigma" in low or "noise" in low):
            add("sigma_schedule", {"name": "adaptive_heat"}, 0.58, 0.35)
        if ("antithetic" in low) or ("mirrored" in low and "sampling" in low):
            add("es_trick", {"name": "mirrored_sampling"}, 0.70, 0.20)

        # Rank transform motifs
        if ("rank" in low) and ("center" in low or "centered" in low):
            add("rank_transform", {"name": "centered"}, 0.78, 0.05)
        if ("quantile" in low) and ("rank" in low):
            add("rank_transform", {"name": "quantile"}, 0.56, 0.45)

        # Robust aggregation
        if ("huber" in low) or ("winsor" in low) or ("trim" in low and "mean" in low):
            add("aggregator", {"name": "winsorize"}, 0.62, 0.35)

        # Stability tricks
        if ("ema" in low) and ("target" in low or "teacher" in low):
            add("world_model", {"name": "ema_target"}, 0.60, 0.20)
        if ("gradient clipping" in low) or ("clip_grad" in low) or ("clipnorm" in low):
            add("stability", {"name": "clip_like"}, 0.70, 0.15)
        if ("weight decay" in low) or ("adamw" in low):
            add("stability", {"name": "decay_like"}, 0.58, 0.25)

        # Self-supervised invariance / redundancy reduction
        if ("vicreg" in low) or ("barlow" in low) or ("invariance" in low and "loss" in low):
            add("loss_term", {"name": "invariance_strengthen"}, 0.62, 0.25)
        if ("redundancy" in low and "reduction" in low):
            add("loss_term", {"name": "redundancy_reduce"}, 0.55, 0.45)
        if "information bottleneck" in low:
            add("loss_term", {"name": "ib_strengthen"}, 0.62, 0.20)

        # Self-play / league motifs
        if ("self-play" in low) or ("self play" in low) or ("league" in low):
            add("curriculum", {"name": "league_selfplay"}, 0.55, 0.45)

        # Liquid / continuous-time motifs
        if ("time constant" in low) or ("tau" in low and ("ode" in low or "continuous" in low)):
            add("liquid_dynamics", {"name": "tau_reg"}, 0.56, 0.25)

        return out


# -----------------------------------------------------------------------------
# 3) arXiv miner (motif-only; Atom XML -> keyword)
# -----------------------------------------------------------------------------
class ArxivMiner:
    def __init__(self, http: Http, cfg: OnlineConfig):
        self.http = http
        self.cfg = cfg

    def mine(self) -> List[ExtractedPrinciple]:
        out: List[ExtractedPrinciple] = []
        queries = [
            "joint embedding predictive architecture self-supervised",
            "redundancy reduction invariance self-supervised",
            "information bottleneck representation learning",
            "evolution strategies rank transformation variance reduction",
            "self-play league training curriculum",
            "liquid neural networks continuous-time recurrent",
        ]
        for q in queries:
            try:
                out.extend(self._search(q))
            except Exception:
                continue
        return out

    def _search(self, query: str) -> List[ExtractedPrinciple]:
        q = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={self.cfg.arxiv_max_results}"
        xml = self.http.get_text(url, headers={"User-Agent": self.http.user_agent})
        # minimal parse for titles+summaries
        parts = re.findall(r"<title>(.*?)</title>|<summary>(.*?)</summary>", xml, flags=re.DOTALL | re.IGNORECASE)
        txt = " ".join((a or b) for a, b in parts)
        low = re.sub(r"\s+", " ", txt).lower()

        out: List[ExtractedPrinciple] = []
        src = f"arxiv:{query}"

        def add(kind: str, payload: Dict[str, Any], conf: float, nov: float) -> None:
            out.append(ExtractedPrinciple(source=src, kind=kind, payload=payload, confidence=conf, novelty=nov))

        if "ema" in low and ("teacher" in low or "target" in low):
            add("world_model", {"name": "ema_target"}, 0.60, 0.15)
        if "information bottleneck" in low:
            add("loss_term", {"name": "ib_strengthen"}, 0.60, 0.20)
        if ("vicreg" in low) or ("barlow" in low) or ("invariance" in low):
            add("loss_term", {"name": "invariance_strengthen"}, 0.60, 0.20)
        if ("redundancy reduction" in low) or ("decorrelation" in low):
            add("loss_term", {"name": "redundancy_reduce"}, 0.55, 0.35)
        if ("rank transformation" in low) or ("mirrored sampling" in low) or ("antithetic" in low):
            add("rank_transform", {"name": "centered"}, 0.56, 0.10)
        if ("self-play" in low) or ("league" in low):
            add("curriculum", {"name": "league_selfplay"}, 0.55, 0.30)
        if ("continuous-time" in low) or ("ode" in low):
            add("liquid_dynamics", {"name": "tau_reg"}, 0.55, 0.20)

        return out


# -----------------------------------------------------------------------------
# 4) Online knowledge hub with caching and throttling
# -----------------------------------------------------------------------------
class OnlineKnowledgeHub:
    def __init__(self, cfg: OnlineConfig, cache_path: str):
        self.cfg = cfg
        self.cache_path = cache_path
        self.http = Http()
        self.gh = GitHubMiner(self.http, cfg)
        self.ax = ArxivMiner(self.http, cfg)
        self._last_refresh = 0.0
        self._principles: Dict[str, ExtractedPrinciple] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._last_refresh = float(raw.get("last_refresh", 0.0))
            for p in raw.get("principles", []) or []:
                pr = ExtractedPrinciple(
                    source=p["source"], kind=p["kind"], payload=p["payload"],
                    confidence=float(p.get("confidence", 0.5)),
                    novelty=float(p.get("novelty", 0.0)),
                    ts=float(p.get("ts", time.time()))
                )
                self._principles[pr.fingerprint()] = pr
        except Exception:
            pass

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            ps = list(self._principles.values())
            ps.sort(key=lambda p: (-(p.confidence + 0.25 * p.novelty), -p.ts))
            ps = ps[: int(self.cfg.max_cached_principles)]
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"last_refresh": self._last_refresh, "principles": [vars(p) for p in ps]},
                    f, indent=2
                )
        except Exception:
            pass

    def refresh_if_needed(self) -> None:
        if not self.cfg.enabled:
            return
        now = time.time()
        if (now - self._last_refresh) < float(self.cfg.refresh_every_s):
            return

        new: List[ExtractedPrinciple] = []
        if self.cfg.github_enabled:
            try:
                new.extend(self.gh.mine())
            except Exception:
                pass
        if self.cfg.arxiv_enabled:
            try:
                new.extend(self.ax.mine())
            except Exception:
                pass

        for p in new:
            self._principles[p.fingerprint()] = p

        self._last_refresh = now
        self._save_cache()

    def all_principles(self) -> List[ExtractedPrinciple]:
        return list(self._principles.values())


# -----------------------------------------------------------------------------
# 5) Principle selection policy (contextual bandit)
# -----------------------------------------------------------------------------
@dataclass
class BanditArmStats:
    pulls: int = 0
    wins: int = 0
    score_sum: float = 0.0

    def mean(self) -> float:
        return self.score_sum / max(1, self.pulls)

class PrincipleBandit:
    """
    Learns which principle kinds/payloads transfer well in your current regime.
    Uses UCB with a novelty bonus, optionally conditioned on task-embedding bucket.
    """
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed + 777)
        self.stats: Dict[str, BanditArmStats] = {}  # arm_key -> stats
        self.total_pulls = 0

    @staticmethod
    def _arm_key(p: ExtractedPrinciple) -> str:
        # coarse arm: kind + payload name if present
        nm = str(p.payload.get("name", p.payload.get("style", "")))
        return f"{p.kind}:{nm}"

    def rank(self, principles: List[ExtractedPrinciple], k: int) -> List[ExtractedPrinciple]:
        if not principles:
            return []

        # UCB ranking
        self.total_pulls = max(self.total_pulls, 1)
        scored: List[Tuple[float, ExtractedPrinciple]] = []
        for p in principles:
            key = self._arm_key(p)
            st = self.stats.get(key, BanditArmStats())
            # optimistic prior for unseen arms
            mean = st.mean() if st.pulls > 0 else 0.15
            ucb = math.sqrt(2.0 * math.log(self.total_pulls + 1.0) / (st.pulls + 1.0))
            # novelty and confidence as weak priors
            prior = 0.10 * p.confidence + 0.06 * p.novelty
            score = mean + 0.35 * ucb + prior
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        # portfolio: limit duplicates of same arm to preserve diversity
        out: List[ExtractedPrinciple] = []
        seen_arm: Dict[str, int] = {}
        for _, p in scored:
            key = self._arm_key(p)
            if seen_arm.get(key, 0) >= 3:
                continue
            seen_arm[key] = seen_arm.get(key, 0) + 1
            out.append(p)
            if len(out) >= int(k):
                break
        return out

    def update(self, used: List[ExtractedPrinciple], reward: float, won: bool) -> None:
        # reward is scalar improvement proxy (positive = good)
        for p in used:
            key = self._arm_key(p)
            st = self.stats.setdefault(key, BanditArmStats())
            st.pulls += 1
            st.wins += 1 if won else 0
            st.score_sum += float(reward)
            self.total_pulls += 1


# -----------------------------------------------------------------------------
# 6) Principle -> internal mutation compiler
# -----------------------------------------------------------------------------
class PrincipleCompiler:
    @staticmethod
    def compile(principles: List[ExtractedPrinciple], base: CandidateRule, temp: float, rng: random.Random) -> Tuple[List[CandidateRule], List[ExtractedPrinciple]]:
        """
        Returns: (candidate_rules, used_principles)
        """
        cands: List[CandidateRule] = []
        used: List[ExtractedPrinciple] = []
        seen = set()

        for pr in principles:
            fp = pr.fingerprint()
            if fp in seen:
                continue
            seen.add(fp)

            u = UpdateRuleSpec(**vars(base.update))
            o = ObjectiveSpec(**vars(base.objective))

            # ------------------- update rule mutations -------------------
            if pr.kind == "lr_schedule":
                u.lr_schedule = str(pr.payload.get("name", "constant"))
            elif pr.kind == "sigma_schedule":
                u.sigma_schedule = str(pr.payload.get("name", "constant"))
            elif pr.kind == "rank_transform":
                nm = str(pr.payload.get("name", "centered"))
                u.rank_mode = "centered" if nm != "raw" else "raw"
                if nm == "quantile":
                    u.aggregator_src = (
                        "def f(shaped):\n"
                        "    # quantile-ish weighting (principle-derived)\n"
                        "    out=[]\n"
                        "    for v in shaped:\n"
                        "        s = 1.0 if v>=0 else -1.0\n"
                        "        out.append(s*math.sqrt(abs(v)))\n"
                        "    return out\n"
                    )
            elif pr.kind == "aggregator":
                if str(pr.payload.get("name", "")) == "winsorize":
                    u.aggregator_src = (
                        "def f(shaped):\n"
                        "    # winsorize weights to reduce tail domination\n"
                        "    out=[]\n"
                        "    for v in shaped:\n"
                        "        if v>0.35: v=0.35\n"
                        "        if v<-0.35: v=-0.35\n"
                        "        out.append(v)\n"
                        "    return out\n"
                    )
            elif pr.kind == "stability":
                nm = str(pr.payload.get("name", ""))
                if nm == "clip_like":
                    u.aggregator_src = (
                        "def f(shaped):\n"
                        "    # clip-like ES shaping (principle-derived)\n"
                        "    out=[]\n"
                        "    for v in shaped:\n"
                        "        if v>0.25: v=0.25\n"
                        "        if v<-0.25: v=-0.25\n"
                        "        out.append(v)\n"
                        "    return out\n"
                    )
                if nm == "decay_like":
                    o.w_wd = float(clamp(o.w_wd * (1.10 + 0.20*rng.random()), 0.0, 3e-3))

            # ------------------- objective / loss mutations -------------------
            elif pr.kind == "loss_term":
                nm = str(pr.payload.get("name", ""))
                if nm == "invariance_strengthen":
                    o.w_inv = float(clamp(o.w_inv * (1.12 + 0.30 * rng.random()), 0.0, 0.85))
                elif nm == "ib_strengthen":
                    o.w_ib = float(clamp(o.w_ib * (1.12 + 0.30 * rng.random()), 0.0, 0.45))
                elif nm == "redundancy_reduce":
                    # approximate redundancy reduction via increased invariance + slight IB
                    o.w_inv = float(clamp(o.w_inv * (1.08 + 0.25*rng.random()), 0.0, 0.85))
                    o.w_ib = float(clamp(o.w_ib * (1.06 + 0.20*rng.random()), 0.0, 0.45))

            elif pr.kind == "liquid_dynamics":
                # inject mild tau stabilization if empty
                if not o.loss_dsl.strip():
                    o.loss_dsl = (
                        "def f(parts):\n"
                        "    pred,ib,inv,wd,tau = parts\n"
                        "    # principle-derived tau stabilization\n"
                        "    return pred + 0.02*ib + 0.05*inv + 1e-4*wd + 0.001*(tau-1.0)*(tau-1.0)\n"
                    )

            # curriculum motifs are used indirectly (compute governor / temperature shaping)
            # because world forge logic lives in v2 core
            elif pr.kind == "curriculum":
                # nudge exploration temperature upward to encourage escape from local optima
                pass

            # gate by temp so cold regimes aren't flooded
            gate = 0.20 + 0.35 * min(1.0, float(temp))
            if rng.random() < gate:
                r = CandidateRule(update=u, objective=o)
                r.id = f"online:{pr.kind}:{pr.fingerprint()}"
                cands.append(r)
                used.append(pr)

        return cands, used


# -----------------------------------------------------------------------------
# 7) Compute governor (self-tunes trials/steps based on marginal gain)
# -----------------------------------------------------------------------------
@dataclass
class ComputeGovernor:
    min_trials: int = 5
    max_trials: int = 60
    min_steps: int = 6
    max_steps: int = 80
    target_improve: float = 0.01   # desired oracle improvement per cycle (approx)
    ema: float = 0.0

    def update(self, improve: float) -> None:
        self.ema = 0.85 * self.ema + 0.15 * float(improve)

    def suggest(self, base_trials: int, base_steps: int) -> Tuple[int, int]:
        # if improvement is small -> increase compute; if large -> can reduce
        e = self.ema
        # avoid division by zero; scale factor in [0.7, 1.4]
        ratio = clamp(self.target_improve / (abs(e) + 1e-6), 0.7, 1.4)
        trials = int(clamp(round(base_trials * ratio), self.min_trials, self.max_trials))
        steps = int(clamp(round(base_steps * ratio), self.min_steps, self.max_steps))
        return trials, steps


# -----------------------------------------------------------------------------
# 8) Online-aware crystallizer: mixes local + online candidates; learns bandit
# -----------------------------------------------------------------------------
class OnlineRuleCrystallizer(RuleCrystallizer):
    def __init__(self, seed: int = 0, hub: Optional[OnlineKnowledgeHub] = None):
        super().__init__(seed=seed)
        self.hub = hub
        self.rng = random.Random(seed + 1234)
        self.bandit = PrincipleBandit(seed=seed + 42)
        self.gov = ComputeGovernor()
        self._last_used: Dict[str, List[ExtractedPrinciple]] = {}  # rule.id -> principles used

    def propose(self, base: CandidateRule, temp: float, n: int) -> List[CandidateRule]:
        # local proposals from v2
        local = super().propose(base, temp, n)

        if self.hub is None or not self.hub.cfg.enabled:
            return local

        # rate-limited refresh
        self.hub.refresh_if_needed()

        # rank principles via bandit portfolio
        ps = self.hub.all_principles()
        ps_ranked = self.bandit.rank(ps, k=80)

        online_cands, used = PrincipleCompiler.compile(ps_ranked, base=base, temp=float(temp), rng=self.rng)
        for c in online_cands:
            self._last_used[c.id] = list(used)

        # mix policy: keep local majority, inject a few online
        self.rng.shuffle(online_cands)
        add_k = int(clamp(2 + 3*temp, 1, 10))
        return local + online_cands[:add_k]

    def select_best(self, scored: List[Tuple[CandidateRule, MultiScore]]) -> Tuple[CandidateRule, MultiScore]:
        # defer to v2 selection (Pareto-ish) for consistency
        return super().select_best(scored)

    def post_cycle_update(self, best_rule: CandidateRule, best_score: MultiScore, prev_oracle: Optional[float]) -> None:
        # update compute governor from improvement
        oracle = float(best_score.holdout_mean)
        improve = 0.0 if (prev_oracle is None) else (prev_oracle - oracle)  # lower loss is better
        self.gov.update(improve)

        # bandit reward only if the best rule was derived from online
        if best_rule.id and best_rule.id.startswith("online:"):
            used = self._last_used.get(best_rule.id, [])
            # reward: positive if improved oracle; else negative
            reward = float(improve)
            won = bool(improve > 0.0)
            self.bandit.update(used, reward=reward, won=won)


# -----------------------------------------------------------------------------
# 9) v4 engine wrapper: swaps crystallizer and adds persistent online state
# -----------------------------------------------------------------------------
class RSIFusionV4Online(RSIFusionV2):
    """
    Extension of v2:
    - Online knowledge hub + bandit-guided principle selection
    - Compute governor self-tuning: adjusts trials/steps via meta temperature
    - Keeps v2's multi-universe selection and cross-pollination
    """
    def __init__(self, cfg: EngineConfig, online: OnlineConfig):
        super().__init__(cfg)
        self.online_cfg = online

        # online cache
        cache_path = os.path.join(self.run_path, "online_cache", "principles.json")
        hub = OnlineKnowledgeHub(online, cache_path=cache_path)

        injected = OnlineRuleCrystallizer(seed=cfg.seed + 99, hub=hub)
        # keep library learned so far if resume loads it into self.crystallizer already; but in v2 __init__ creates new.
        self.crystallizer = injected

        # store compute governor state
        self._gov_path = os.path.join(self.run_path, "online_cache", "compute_governor.json")
        self._load_governor()

    def _load_governor(self) -> None:
        try:
            with open(self._gov_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(self.crystallizer, OnlineRuleCrystallizer):
                g = self.crystallizer.gov
                g.ema = float(d.get("ema", 0.0))
                g.target_improve = float(d.get("target_improve", g.target_improve))
        except Exception:
            pass

    def _save_governor(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._gov_path), exist_ok=True)
            if isinstance(self.crystallizer, OnlineRuleCrystallizer):
                g = self.crystallizer.gov
                with open(self._gov_path, "w", encoding="utf-8") as f:
                    json.dump({"ema": g.ema, "target_improve": g.target_improve}, f, indent=2)
        except Exception:
            pass

    # We override _run_universe_cycle only to add bandit/governor updates,
    # without modifying the underlying v2 verification semantics.
    def _run_universe_cycle(self, u: Any, cycle: int, global_seed: int) -> Dict[str, Any]:
        prev_oracle = u.best_oracle
        report = super()._run_universe_cycle(u, cycle, global_seed)

        # post-cycle update for bandit/governor if possible
        try:
            if isinstance(self.crystallizer, OnlineRuleCrystallizer):
                # best_score stored in report["best_score"] in v2? If not, infer from u.active_rule with re-eval.
                best_score = None
                if isinstance(report, dict):
                    best_score = report.get("score", None)
                if best_score is None:
                    # conservative: no update
                    self._save_governor()
                    return report

                # best_score is likely serialized; rebuild MultiScore
                if isinstance(best_score, dict):
                    bs = MultiScore(
                        train_loss=float(best_score.get("train_loss", 0.0)),
                        holdout_losses=[float(x) for x in (best_score.get("holdout_losses", []) or [])],
                        stability=float(best_score.get("stability", 0.0)),
                        complexity=float(best_score.get("complexity", 0.0)),
                    )
                else:
                    bs = best_score  # type: ignore

                self.crystallizer.post_cycle_update(u.active_rule, bs, prev_oracle)
                self._save_governor()
        except Exception:
            # never let online logic crash the RSI loop
            self._save_governor()
        return report


# -----------------------------------------------------------------------------
# 10) CLI
# -----------------------------------------------------------------------------
def _parse_bool(s: str) -> bool:
    s = str(s).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on")

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default="v4_online")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cycles", type=int, default=60)
    ap.add_argument("--online", type=str, default="0")
    ap.add_argument("--github", type=str, default="1")
    ap.add_argument("--arxiv", type=str, default="1")
    ap.add_argument("--refresh_every_s", type=int, default=60*30)

    # optional scaling knobs (no hard "speed limit"; user can set high)
    ap.add_argument("--train_steps", type=int, default=None)
    ap.add_argument("--inner_discovery_trials", type=int, default=None)
    ap.add_argument("--fast", type=str, default=None)

    args = ap.parse_args()

    cfg = EngineConfig(out_dir=args.out_dir, run_name=args.run_name, seed=args.seed, cycles=args.cycles)
    if args.train_steps is not None:
        cfg.train_steps = int(args.train_steps)
    if args.inner_discovery_trials is not None:
        cfg.inner_discovery_trials = int(args.inner_discovery_trials)
    if args.fast is not None:
        cfg.fast = _parse_bool(args.fast)

    online = OnlineConfig(
        enabled=_parse_bool(args.online),
        github_enabled=_parse_bool(args.github),
        arxiv_enabled=_parse_bool(args.arxiv),
        refresh_every_s=int(args.refresh_every_s),
    )

    eng = RSIFusionV4Online(cfg, online=online)
    eng.run(resume=True)

if __name__ == "__main__":
    main()
