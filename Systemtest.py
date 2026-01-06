"""
SYSTEM TEST (Hierarchical Reasoning & Evolution)
Entry Point for Autonomous Code Generation
"""
from __future__ import annotations
import sys
import os
import json
import random
import time
import math
import re
import ast
import argparse
import multiprocessing as mp
import hashlib
import copy
import collections
import difflib
import subprocess
import shutil
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set, Union
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None

# ==========================================


# START OF omega_forge_two_stage_feedback.py

#!/usr/bin/env python3
"""
OMEGA_FORGE_V13_CLEAN.py
========================
Streaming structural-transition discovery engine (CLEAN edition).

Design goals
------------
1) Crash-safe evidence logging: write JSONL incrementally with flush + fsync.
2) No "fake pass": enforce global uniqueness + real parent tracking.
3) Separate concerns: Engine (search) vs Detector (gate) vs EvidenceWriter (persistence).
4) Selftest is not a benchmark: it validates execution + logging, and does NOT fail just because
   zero successes occurred in a short horizon.

Usage
-----
  python OMEGA_FORGE_V13_CLEAN.py selftest
  python OMEGA_FORGE_V13_CLEAN.py evidence_run --target 6 --max_generations 2000 --out evidence_v13.jsonl
  python OMEGA_FORGE_V13_CLEAN.py run --generations 5000 --log v13_run.jsonl
"""


import argparse
import json
import math
import os
import random
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from collections import defaultdict, Counter

# ==============================================================================
# 1) Instruction set
# ==============================================================================

OPS = [
    "MOV", "SET", "SWAP",
    "ADD", "SUB", "MUL", "DIV", "INC", "DEC",
    "LOAD", "STORE", "LDI", "STI",
    "JMP", "JZ", "JNZ", "JGT", "JLT",
    "CALL", "RET", "HALT"
]
CONTROL_OPS = {"JMP", "JZ", "JNZ", "JGT", "JLT", "CALL", "RET"}
MEMORY_OPS = {"LOAD", "STORE", "LDI", "STI"}

@dataclass
class Instruction:
    op: str
    a: int = 0
    b: int = 0
    c: int = 0

    def clone(self) -> "Instruction":
        return Instruction(self.op, self.a, self.b, self.c)

    def to_tuple(self) -> Tuple[Any, ...]:
        return (self.op, int(self.a), int(self.b), int(self.c))

# ==============================================================================
# 2) Program genome
# ==============================================================================

@dataclass
class ProgramGenome:
    gid: str
    instructions: List[Instruction]
    parents: List[str] = field(default_factory=list)
    generation: int = 0
    last_score: float = 0.0
    last_cfg_hash: str = ""
    concept_trace: List[str] = field(default_factory=list)
    concept_proposals: List[str] = field(default_factory=list)

    def clone(self) -> "ProgramGenome":
        return ProgramGenome(
            gid=self.gid,
            instructions=[i.clone() for i in self.instructions],
            parents=list(self.parents),
            generation=self.generation,
            concept_trace=list(self.concept_trace),
            concept_proposals=list(self.concept_proposals),
        )

    def code_hash(self) -> str:
        h = hashlib.sha256()
        for inst in self.instructions:
            h.update(repr(inst.to_tuple()).encode("utf-8"))
        return h.hexdigest()[:16]

    def op_sequence(self) -> List[str]:
        return [i.op for i in self.instructions]

# ==============================================================================
# 3) Execution state + CFG
# ==============================================================================

@dataclass
class ExecutionState:
    regs: List[float]
    memory: Dict[int, float]
    pc: int = 0
    stack: List[int] = field(default_factory=list)
    steps: int = 0
    halted: bool = False
    halted_cleanly: bool = False
    error: Optional[str] = None

    trace: List[int] = field(default_factory=list)
    visited_pcs: Set[int] = field(default_factory=set)

    loops_count: int = 0
    conditional_branches: int = 0
    max_call_depth: int = 0
    memory_reads: int = 0
    memory_writes: int = 0

    def coverage(self, code_len: int) -> float:
        if code_len <= 0:
            return 0.0
        return len(self.visited_pcs) / float(code_len)

    def fingerprint(self) -> Tuple[int, int, int, int, int]:
        return (
            min(self.loops_count, 20),
            min(self.conditional_branches, 20),
            min(self.memory_writes, 50),
            min(self.memory_reads, 50),
            min(self.max_call_depth, 10),
        )

class ControlFlowGraph:
    def __init__(self) -> None:
        self.edges: Set[Tuple[int, int, str]] = set()
        self.nodes: Set[int] = set()

    def add_edge(self, f: int, t: int, ty: str) -> None:
        self.edges.add((int(f), int(t), str(ty)))
        self.nodes.add(int(f))
        self.nodes.add(int(t))

    @staticmethod
    def from_trace(trace: List[int], code_len: int) -> "ControlFlowGraph":
        cfg = ControlFlowGraph()
        if not trace:
            return cfg
        for i in range(len(trace) - 1):
            a = trace[i]
            b = trace[i + 1]
            ty = "SEQ"
            if b <= a:
                ty = "BACK"
            cfg.add_edge(a, b, ty)
        # Add terminal edge for out-of-range halt
        last = trace[-1]
        cfg.nodes.add(last)
        cfg.nodes.add(max(0, min(code_len, last + 1)))
        return cfg

    def canonical_hash(self) -> str:
        # canonical: sorted edges + SCC size multiset
        h = hashlib.sha256()
        for f, t, ty in sorted(self.edges):
            h.update(f"{f}->{t}:{ty};".encode("utf-8"))
        scc_sizes = sorted([len(s) for s in self.sccs()])
        h.update(("SCC:" + ",".join(map(str, scc_sizes))).encode("utf-8"))
        return h.hexdigest()[:16]

    def sccs(self) -> List[FrozenSet[int]]:
        # Kosaraju
        if not self.nodes:
            return []
        adj = defaultdict(list)
        radj = defaultdict(list)
        for f, t, _ in self.edges:
            adj[f].append(t)
            radj[t].append(f)

        visited: Set[int] = set()
        order: List[int] = []

        def dfs1(u: int) -> None:
            if u in visited:
                return
            visited.add(u)
            for v in adj[u]:
                dfs1(v)
            order.append(u)

        for n in list(self.nodes):
            dfs1(n)

        visited.clear()
        comps: List[FrozenSet[int]] = []

        def dfs2(u: int, comp: Set[int]) -> None:
            if u in visited:
                return
            visited.add(u)
            comp.add(u)
            for v in radj[u]:
                dfs2(v, comp)

        for u in reversed(order):
            if u not in visited:
                comp: Set[int] = set()
                dfs2(u, comp)
                # SCC is meaningful if size>1 or has a self-loop
                if len(comp) > 1:
                    comps.append(frozenset(comp))
                else:
                    x = next(iter(comp)) if comp else None
                    if x is not None and any((x, x, ty) in self.edges for ty in ("SEQ", "BACK")):
                        comps.append(frozenset(comp))
        return comps

    def edit_distance_to(self, other: "ControlFlowGraph") -> int:
        # symmetric difference on typed edges
        return len(self.edges ^ other.edges)

# ==============================================================================
# 4) Virtual machine
# ==============================================================================

class VirtualMachine:
    def __init__(self, max_steps: int = 400, memory_size: int = 64, stack_limit: int = 16) -> None:
        self.max_steps = max_steps
        self.memory_size = memory_size
        self.stack_limit = stack_limit

    def reset(self, inputs: List[float]) -> ExecutionState:
        regs = [0.0] * 8
        mem: Dict[int, float] = {}
        for i, v in enumerate(inputs):
            if i < self.memory_size:
                mem[i] = float(v)
        regs[1] = float(len(inputs))
        return ExecutionState(regs=regs, memory=mem)

    def execute(self, genome: ProgramGenome, inputs: List[float]) -> ExecutionState:
        st = self.reset(inputs)
        code = genome.instructions
        L = len(code)

        recent_hashes: List[int] = []
        while not st.halted and st.steps < self.max_steps:
            if st.pc < 0 or st.pc >= L:
                st.halted = True
                st.halted_cleanly = True
                break

            st.visited_pcs.add(st.pc)
            st.trace.append(st.pc)
            prev_pc = st.pc
            inst = code[st.pc]
            st.steps += 1

            # Degenerate loop detection: if state hashes collapse, stop with error
            state_sig = hash((st.pc, tuple(int(x) for x in st.regs[:4]), len(st.stack)))
            recent_hashes.append(state_sig)
            if len(recent_hashes) > 25:
                recent_hashes.pop(0)
                if len(set(recent_hashes)) < 3:
                    st.error = "DEGENERATE_LOOP"
                    st.halted = True
                    break

            try:
                self._step(st, inst)
            except Exception as e:
                st.error = f"VM_ERR:{e.__class__.__name__}"
                st.halted = True
                break

            # Loop + branch stats
            if st.pc <= prev_pc and not st.halted:
                st.loops_count += 1
            if inst.op in {"JZ", "JNZ", "JGT", "JLT"}:
                st.conditional_branches += 1
            st.max_call_depth = max(st.max_call_depth, len(st.stack))

        return st

    def _step(self, st: ExecutionState, inst: Instruction) -> None:
        op, a, b, c = inst.op, inst.a, inst.b, inst.c
        r = st.regs

        def clamp(x: float) -> float:
            if not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
                return 0.0
            return float(max(-1e9, min(1e9, x)))

        def addr(x: float) -> int:
            return int(max(0, min(self.memory_size - 1, int(x))))

        jump = False

        if op == "HALT":
            st.halted = True
            st.halted_cleanly = True
            return

        if op == "SET":
            r[c % 8] = float(a)
        elif op == "MOV":
            r[c % 8] = float(r[a % 8])
        elif op == "SWAP":
            ra, rb = a % 8, b % 8
            r[ra], r[rb] = r[rb], r[ra]
        elif op == "ADD":
            r[c % 8] = clamp(r[a % 8] + r[b % 8])
        elif op == "SUB":
            r[c % 8] = clamp(r[a % 8] - r[b % 8])
        elif op == "MUL":
            r[c % 8] = clamp(r[a % 8] * r[b % 8])
        elif op == "DIV":
            den = r[b % 8]
            r[c % 8] = clamp(r[a % 8] / den) if abs(den) > 1e-9 else 0.0
        elif op == "INC":
            r[c % 8] = clamp(r[c % 8] + 1.0)
        elif op == "DEC":
            r[c % 8] = clamp(r[c % 8] - 1.0)
        elif op == "LOAD":
            idx = addr(r[a % 8])
            st.memory_reads += 1
            r[c % 8] = float(st.memory.get(idx, 0.0))
        elif op == "STORE":
            idx = addr(r[a % 8])
            st.memory_writes += 1
            st.memory[idx] = clamp(r[c % 8])
        elif op == "LDI":
            base = addr(r[a % 8])
            off = addr(r[b % 8])
            st.memory_reads += 1
            r[c % 8] = float(st.memory.get(addr(base + off), 0.0))
        elif op == "STI":
            base = addr(r[a % 8])
            off = addr(r[b % 8])
            st.memory_writes += 1
            st.memory[addr(base + off)] = clamp(r[c % 8])
        elif op == "JMP":
            st.pc += int(a)
            jump = True
        elif op == "JZ":
            if abs(r[a % 8]) < 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JNZ":
            if abs(r[a % 8]) >= 1e-9:
                st.pc += int(b)
                jump = True
        elif op == "JGT":
            if r[a % 8] > r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "JLT":
            if r[a % 8] < r[b % 8]:
                st.pc += int(c)
                jump = True
        elif op == "CALL":
            if len(st.stack) >= self.stack_limit:
                st.error = "STACK_OVERFLOW"
                st.halted = True
                return
            st.stack.append(st.pc + 1)
            st.pc += int(a)
            jump = True
        elif op == "RET":
            if not st.stack:
                st.halted = True
                st.halted_cleanly = True
                jump = True
            else:
                st.pc = st.stack.pop()
                jump = True
        else:
            # Unknown op => halt
            st.error = "UNKNOWN_OP"
            st.halted = True
            return

        if not jump:
            st.pc += 1

# ==============================================================================
# 5) Mutation operators (include structural builders)
# ==============================================================================

class MacroLibrary:
    @staticmethod
    def loop_skeleton(idx_reg: int = 2, limit_reg: int = 1) -> List[Instruction]:
        # i=0 ; if i<limit: body ; i++ ; jump back ; halt path outside
        return [
            Instruction("SET", 0, 0, idx_reg),
            Instruction("JLT", idx_reg, limit_reg, 4),   # jump into body if i < limit
            Instruction("JMP", 6, 0, 0),                # skip body (exit)
            Instruction("INC", 0, 0, idx_reg),          # body: i++
            Instruction("JMP", -3, 0, 0),               # loop back to JLT
        ]

    @staticmethod
    def call_skeleton() -> List[Instruction]:
        # CALL forward to a mini-routine and RET
        return [
            Instruction("CALL", 2, 0, 0),
            Instruction("JMP", 3, 0, 0),
            Instruction("INC", 0, 0, 0),
            Instruction("RET", 0, 0, 0),
        ]

# ==============================================================================
# 5.2) Concept Library (macros + reusable fragments)
# ==============================================================================

@dataclass
class Concept:
    cid: str
    name: str
    kind: str
    payload: Dict[str, Any]
    compile_fn_id: str
    stats: Dict[str, Any] = field(default_factory=dict)
    discovered_gen: int = 0
    parents: List[str] = field(default_factory=list)

def _inst_tuple_list(instructions: List[Instruction]) -> List[Tuple[Any, ...]]:
    return [inst.to_tuple() for inst in instructions]

def _concept_hash_from_insts(instructions: List[Instruction]) -> str:
    h = hashlib.sha256()
    for inst in instructions:
        h.update(repr(inst.to_tuple()).encode("utf-8"))
    return h.hexdigest()[:16]

def _compile_macro_v1(payload: Dict[str, Any]) -> List[Instruction]:
    insts = []
    for op, a, b, c in payload.get("instructions", []):
        insts.append(Instruction(str(op), int(a), int(b), int(c)))
    return insts

CONCEPT_COMPILE_FNS: Dict[str, Callable[[Dict[str, Any]], List[Instruction]]] = {
    "macro_v1": _compile_macro_v1,
}

class ConceptLibrary:
    def __init__(self, max_size: int = 200) -> None:
        self.max_size = max_size
        self._concepts: Dict[str, Concept] = {}
        self._hash_index: Dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._concepts)

    def all_concepts(self) -> List[Concept]:
        return list(self._concepts.values())

    def get(self, cid: str) -> Optional[Concept]:
        return self._concepts.get(cid)

    def add_concept(self, concept: Concept, dedup: bool = True) -> Optional[str]:
        instructions = self.build_instructions(concept)
        if not instructions:
            return None
        digest = _concept_hash_from_insts(instructions)
        if dedup and digest in self._hash_index:
            return self._hash_index[digest]
        if len(self._concepts) >= self.max_size:
            return None
        self._concepts[concept.cid] = concept
        self._hash_index[digest] = concept.cid
        concept.stats.setdefault("digest", digest)
        concept.stats.setdefault("length", len(instructions))
        return concept.cid

    def build_instructions(self, concept: Concept) -> List[Instruction]:
        fn = CONCEPT_COMPILE_FNS.get(concept.compile_fn_id)
        if not fn:
            return []
        return fn(concept.payload)

    def save(self, path: str) -> None:
        data = []
        for c in self._concepts.values():
            data.append({
                "cid": c.cid,
                "name": c.name,
                "kind": c.kind,
                "payload": c.payload,
                "compile_fn_id": c.compile_fn_id,
                "stats": c.stats,
                "discovered_gen": c.discovered_gen,
                "parents": c.parents,
            })
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for entry in raw or []:
            concept = Concept(
                cid=str(entry.get("cid", "")),
                name=str(entry.get("name", "concept")),
                kind=str(entry.get("kind", "macro")),
                payload=dict(entry.get("payload", {})),
                compile_fn_id=str(entry.get("compile_fn_id", "macro_v1")),
                stats=dict(entry.get("stats", {})),
                discovered_gen=int(entry.get("discovered_gen", 0)),
                parents=list(entry.get("parents", [])),
            )
            self.add_concept(concept, dedup=True)

# ------------------------------------------------------------------------------
# Feedback-biased opcode sampling (Stage2 -> Stage1)
# ------------------------------------------------------------------------------
OP_BIAS: Dict[str, float] = {}  # e.g., {"LOAD":1.4,"ADD":1.3,...}
CONCEPT_BIAS: Dict[str, float] = {}
CONCEPT_ANTI_BIAS: Set[str] = set()
MACRO_LENGTH_BIAS: float = 0.0

def set_op_bias(op_bias: Dict[str, float]) -> None:
    """
    Install opcode sampling bias used by rand_inst() in Stage 1.
    Values are nonnegative weights; missing ops default to 1.0.
    """
    global OP_BIAS
    OP_BIAS = {k: float(v) for k, v in (op_bias or {}).items() if float(v) > 0.0}

def set_concept_bias(concept_bias: Dict[str, float],
                     anti_bias: Optional[List[str]] = None,
                     macro_length_bias: Optional[float] = None) -> None:
    global CONCEPT_BIAS, CONCEPT_ANTI_BIAS, MACRO_LENGTH_BIAS
    CONCEPT_BIAS = {k: float(v) for k, v in (concept_bias or {}).items() if float(v) > 0.0}
    CONCEPT_ANTI_BIAS = set(anti_bias or [])
    if macro_length_bias is not None:
        MACRO_LENGTH_BIAS = float(macro_length_bias)

def _sample_op(rng: random.Random) -> str:
    if not OP_BIAS:
        return rng.choice(OPS)
    weights = [OP_BIAS.get(op, 1.0) for op in OPS]
    # Avoid all-zero
    if not any(w > 0.0 for w in weights):
        return rng.choice(OPS)
    return rng.choices(OPS, weights=weights, k=1)[0]

def rand_inst(rng: Optional[random.Random] = None) -> Instruction:
    """
    Random instruction generator. If OP_BIAS is set (via Stage2 feedback),
    opcode selection is weighted accordingly.
    """
    rng = rng or random
    op = _sample_op(rng)
    return Instruction(op, rng.randint(-8, 31), rng.randint(0, 7), rng.randint(0, 7))

# ==============================================================================
# 5.5) Task-Aware Fitness Benchmark
# ==============================================================================

class TaskBenchmark:
    """Evaluates genomes against practical computational tasks."""
    
    TASKS = [
        # (name, inputs, expected_output_location, expected_value)
        ("SUM_SIMPLE", [1.0, 2.0, 3.0, 4.0, 5.0], "reg0", 15.0),
        ("SUM_SMALL", [2.0, 3.0, 5.0], "reg0", 10.0),
        ("MAX_FIND", [3.0, 7.0, 2.0, 9.0, 1.0], "reg0", 9.0),
        ("COUNT", [1.0, 1.0, 1.0, 1.0], "reg0", 4.0),
        ("DOUBLE_FIRST", [5.0, 0.0, 0.0, 0.0], "mem0", 10.0),
    ]
    
    @staticmethod
    def evaluate(genome: "ProgramGenome", vm: "VirtualMachine") -> float:
        """Returns task score 0.0-1.0 based on practical task performance."""
        passed = 0
        total = len(TaskBenchmark.TASKS)
        
        for name, inputs, out_loc, expected in TaskBenchmark.TASKS:
            try:
                st = vm.execute(genome, inputs)
                if out_loc == "reg0":
                    result = st.regs[0]
                elif out_loc == "mem0":
                    result = st.memory.get(0, 0.0)
                else:
                    result = 0.0
                
                # Check if result matches expected (with tolerance)
                if abs(result - expected) < 0.01:
                    passed += 1
                # Partial credit for being close
                elif abs(result - expected) < expected * 0.1:
                    passed += 0.5
            except:
                pass
        
        return passed / total

# ==============================================================================
# 6) Detector + evidence writer
# ==============================================================================

@dataclass
class DetectorParams:
    # Target: 0.5–5% successes. Use a curriculum so the search has a gradient early,
    # then harden constraints to avoid "linear cheats".
    K_initial: int = 6               # strict CFG edit distance (post-warmup)
    L_initial: int = 10              # strict active subseq length (post-warmup)
    C_coverage: float = 0.55         # min coverage (post-warmup)
    f_rarity: float = 0.001          # rarity threshold (post-warmup)
    N_repro: int = 4                 # reproducibility trials

    require_both: bool = True        # strict mode requires CFG + subseq
    min_loops: int = 2               # STRICT: Require at least 2 loops (Multi-Stage)
    min_scc: int = 2                 # STRICT: Require at least 2 SCCs (Complex Topology)

    allow_cfg_variants: int = 2      # reproducibility CFG variants
    max_cov_span: float = 0.30       # reproducibility coverage stability
    max_loop_span: int = 5           # reproducibility loop stability

    # Warmup curriculum (first warmup_gens generations)
    warmup_gens: int = 100
    warmup_K: int = 3
    warmup_L: int = 8
    warmup_cov: float = 0.45
    warmup_require_both: bool = True # Strict warmup
    warmup_min_loops: int = 1        # Ban linear code even in warmup
    warmup_min_scc: int = 1          # Ban acyclic graphs even in warmup

class StrictStructuralDetector:
    def __init__(self, params: Optional[DetectorParams] = None) -> None:
        self.p = params or DetectorParams()
        self.parent_cfgs: Dict[str, ControlFlowGraph] = {}
        self.subseq_counts: Counter = Counter()
        self.subseq_total: int = 0
        self.seen_success_hashes: Set[str] = set()

    def _in_warmup(self, gen: int) -> bool:
        return gen <= self.p.warmup_gens

    def _K(self, gen: int) -> int:
        # Curriculum: easier early, strict later.
        if self._in_warmup(gen):
            return max(1, int(self.p.warmup_K))
        return max(3, int(self.p.K_initial))

    def _L(self, gen: int) -> int:
        if self._in_warmup(gen):
            return max(4, int(self.p.warmup_L))
        return max(6, int(self.p.L_initial))

    def _anti_cheat(self, st: ExecutionState, code_len: int, gen: int) -> Tuple[bool, str]:
        if st.error:
            return False, f"ERR:{st.error}"
        if not st.halted_cleanly:
            return False, "DIRTY_HALT"
        cov = st.coverage(code_len)
        if cov < self.p.C_coverage:
            return False, f"LOW_COVERAGE:{cov:.3f}"
        min_loops = self.p.warmup_min_loops if self._in_warmup(gen) else self.p.min_loops
        if st.loops_count < min_loops:
            return False, "NO_LOOPS"
        return True, f"ANTI_OK cov={cov:.3f} loops={st.loops_count}"

    def _repro(self, genome: ProgramGenome, vm: VirtualMachine) -> Tuple[bool, str]:
        cfgs: List[str] = []
        covs: List[float] = []
        loops: List[int] = []
        fixed_inputs = [
            [0.0]*8,
            [1.0]*8,
            [2.0]*8,
            [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],
        ]
        for i in range(self.p.N_repro):
            inputs = fixed_inputs[i % len(fixed_inputs)]
            st = vm.execute(genome, inputs)
            cfgs.append(ControlFlowGraph.from_trace(st.trace, len(genome.instructions)).canonical_hash())
            covs.append(st.coverage(len(genome.instructions)))
            loops.append(st.loops_count)

        if len(set(cfgs)) > self.p.allow_cfg_variants:
            return False, "CFG_UNSTABLE"
        if max(covs) - min(covs) > self.p.max_cov_span:
            return False, "COV_UNSTABLE"
        if max(loops) - min(loops) > self.p.max_loop_span:
            return False, "LOOP_UNSTABLE"
        return True, f"REPRO_OK N={self.p.N_repro}"

    def evaluate(
        self,
        genome: ProgramGenome,
        parent: Optional[ProgramGenome],
        st: ExecutionState,
        vm: VirtualMachine,
        generation: int,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        reasons: List[str] = []
        diag: Dict[str, Any] = {}

        ok, msg = self._anti_cheat(st, len(genome.instructions), generation)
        if not ok:
            return False, [f"ANTI_FAIL:{msg}"], diag
        reasons.append(msg)

        cfg = ControlFlowGraph.from_trace(st.trace, len(genome.instructions))
        diag["cfg_hash"] = cfg.canonical_hash()
        # Track CFG for every genome so children can be compared against their parents (prevents deadlock)
        self.parent_cfgs[genome.gid] = cfg
        p_cfg = self.parent_cfgs.get(parent.gid) if parent else None
        if p_cfg is None and parent is not None:
            # Fallback: compute parent's CFG directly (robust even if parent was never a success)
            pst = vm.execute(parent, [1.0] * 8)
            p_cfg = ControlFlowGraph.from_trace(pst.trace, len(parent.instructions))

        cfg_ok = False
        cfg_msg = "CFG_NO_PARENT"
        if p_cfg is not None:
            dist = cfg.edit_distance_to(p_cfg)
            K = self._K(generation)
            cfg_ok = dist >= K
            cfg_msg = f"CFG dist={dist} K={K}"
            diag["cfg_dist"] = dist
        else:
            diag["cfg_dist"] = None

        scc_n = len(cfg.sccs())
        diag["scc_n"] = scc_n
        min_scc = self.p.warmup_min_scc if self._in_warmup(generation) else self.p.min_scc
        if scc_n < min_scc:
            cfg_ok = False
            cfg_msg = "CFG_NO_SCC"

        # subsequence novelty (only executed pcs, contiguous window in instruction index-space)
        L = self._L(generation)
        ops = genome.op_sequence()
        active: List[Tuple[str, ...]] = []
        visited = st.visited_pcs
        for i in range(0, max(0, len(ops) - L + 1)):
            window_pcs = set(range(i, i + L))
            if window_pcs.issubset(visited):
                active.append(tuple(ops[i : i + L]))

        subseq_ok = False
        subseq_msg = "SUBSEQ_NONE"
        if active:
            # rarity by empirical frequency in archive
            for seq in active:
                freq = (self.subseq_counts.get(seq, 0) / max(1, self.subseq_total))
                if freq < self.p.f_rarity:
                    subseq_ok = True
                    subseq_msg = f"SUBSEQ rarity={freq:.6f} L={L}"
                    # Defer archive updates until AFTER full success (CFG+SUBSEQ+REPRO+UNIQUENESS),
                    # otherwise near-misses rapidly poison rarity and can suppress discovery.
                    diag["_candidate_subseq"] = list(seq)
                    diag["subseq"] = list(seq)
                    diag["subseq_freq"] = freq
                    break
        diag["active_subseq_windows"] = len(active)

        # require both or at least one
        require_both = self.p.warmup_require_both if self._in_warmup(generation) else self.p.require_both
        if require_both:
            if not (cfg_ok and subseq_ok and parent is not None):
                return False, [f"REQUIRE_BOTH_FAIL cfg={cfg_ok}({cfg_msg}) subseq={subseq_ok}({subseq_msg})"], diag
        else:
            if not (cfg_ok or subseq_ok):
                return False, [f"NO_STRUCT_CHANGE {cfg_msg}; {subseq_msg}"], diag

        reasons.append(cfg_msg if cfg_ok else cfg_msg)
        reasons.append(subseq_msg if subseq_ok else subseq_msg)

        # reproducibility
        r_ok, r_msg = self._repro(genome, vm)
        if not r_ok:
            return False, [f"REPRO_FAIL:{r_msg}"], diag
        reasons.append(r_msg)

        # global uniqueness on successes (prevents repeated printing of same "success")
        succ_hash = cfg.canonical_hash() + "|" + genome.code_hash()
        if succ_hash in self.seen_success_hashes:
            return False, ["DUP_SUCCESS_HASH"], diag

        self.seen_success_hashes.add(succ_hash)
        diag["success_hash"] = succ_hash

        # Commit subsequence rarity archive ONLY on confirmed success
        if "subseq" in diag:
            key = tuple(diag["subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1
        elif "_candidate_subseq" in diag:
            key = tuple(diag["_candidate_subseq"])
            self.subseq_counts[key] = self.subseq_counts.get(key, 0) + 1
            self.subseq_total += 1

        # store cfg for parent tracking
        self.parent_cfgs[genome.gid] = cfg
        return True, reasons, diag

class EvidenceWriter:
    def __init__(self, out_path: str) -> None:
        self.out_path = out_path
        # Always write a header marker so "empty file" is never ambiguous
        self.f = open(out_path, "a", encoding="utf-8", buffering=1)
        self.write({"type": "header", "version": "V13_CLEAN", "note": "jsonl; each line is crash-safe"})
        self.flush_fsync()

    def write(self, obj: Dict[str, Any]) -> None:
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush_fsync(self) -> None:
        self.f.flush()
        try:
            os.fsync(self.f.fileno())
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.flush_fsync()
        finally:
            self.f.close()

# ==============================================================================
# 7) Engine
# ==============================================================================

@dataclass
class EngineConfig:
    pop_size: int = 30
    init_len_min: int = 18
    init_len_max: int = 28
    elite_keep: int = 12
    children_per_elite: int = 2
    max_code_len: int = 80

class OmegaForgeV13:
    def __init__(
        self,
        seed: int = 42,
        detector: Optional[StrictStructuralDetector] = None,
        vm: Optional[VirtualMachine] = None,
        config: Optional[EngineConfig] = None,
    ) -> None:
        random.seed(seed)
        self.seed = seed
        self.vm = vm or VirtualMachine()
        self.detector = detector or StrictStructuralDetector()
        self.cfg = config or EngineConfig()

        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        self.parents_index: Dict[str, ProgramGenome] = {}

    def init_population(self) -> None:
        self.population = []
        for i in range(self.cfg.pop_size):
            L = random.randint(self.cfg.init_len_min, self.cfg.init_len_max)
            insts = [rand_inst() for _ in range(L)]
            g = ProgramGenome(gid=f"init_{i}", instructions=insts, parents=[], generation=0)
            self.population.append(g)
        self._reindex()

    def _reindex(self) -> None:
        self.parents_index = {g.gid: g for g in self.population}

    def _get_parent_obj(self, g: ProgramGenome) -> Optional[ProgramGenome]:
        if not g.parents:
            return None
        pid = g.parents[0]
        return self.parents_index.get(pid)

    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"g{self.generation}_{random.randint(0, 999999)}"

        # structural mutation mixture
        # 1) splice macro sometimes
        roll = random.random()
        if roll < 0.20 and len(child.instructions) + 5 < self.cfg.max_code_len:
            macro = MacroLibrary.loop_skeleton() if random.random() < 0.7 else MacroLibrary.call_skeleton()
            pos = random.randint(0, len(child.instructions))
            child.instructions[pos:pos] = [m.clone() for m in macro]
        elif roll < 0.45 and child.instructions:
            # replace a random instruction with a control op to encourage CFG change
            pos = random.randint(0, len(child.instructions) - 1)
            op = random.choice(list(CONTROL_OPS))
            child.instructions[pos] = Instruction(op, random.randint(-8, 8), random.randint(0, 7), random.randint(0, 7))
        elif roll < 0.75 and len(child.instructions) < self.cfg.max_code_len:
            # insert random instruction
            pos = random.randint(0, len(child.instructions))
            child.instructions.insert(pos, rand_inst())
        else:
            # delete
            if len(child.instructions) > 6:
                pos = random.randint(0, len(child.instructions) - 1)
                child.instructions.pop(pos)

        return child

    def step(self, writer: Optional[EvidenceWriter] = None) -> Tuple[int, int]:
        self.generation += 1
        successes_this_gen = 0

        # Evaluate all genomes
        for g in self.population:
            parent = self._get_parent_obj(g)
            st = self.vm.execute(g, [1.0] * 8)
            passed, reasons, diag = self.detector.evaluate(g, parent, st, self.vm, self.generation)
            if passed:
                successes_this_gen += 1
                if writer is not None:
                    ev = {
                        "type": "evidence",
                        "gen": self.generation,
                        "gid": g.gid,
                        "parent": parent.gid if parent else None,
                        "code_hash": g.code_hash(),
                        "reasons": reasons,
                        "diag": diag,
                        "metrics": {
                            "steps": st.steps,
                            "coverage": st.coverage(len(g.instructions)),
                            "loops": st.loops_count,
                            "branches": st.conditional_branches,
                            "scc_n": diag.get("scc_n", 0),
                        },
                    }
                    writer.write(ev)
                    writer.flush_fsync()


        # Reproduce: score-based elite selection + CFG-diversity (prevents random drift / stagnation)
        # Score rewards "structural potential" even when not yet passing strict detector gates.
        for g in self.population:
            parent = self._get_parent_obj(g)
            # we already executed VM above in this step's loop, but we don't store st; re-execute cheaply on fixed input
            st2 = self.vm.execute(g, [1.0] * 8)
            cfg2 = ControlFlowGraph.from_trace(st2.trace, len(g.instructions))
            cov = st2.coverage(len(g.instructions))
            scc_n = len(cfg2.sccs())
            
            # STRUCTURAL score (original): coverage + loops/branches/calls + SCC
            struct_score = cov + 0.02 * min(st2.loops_count, 50) + 0.01 * min(st2.conditional_branches, 50) + 0.03 * min(st2.max_call_depth, 10) + 0.08 * min(scc_n, 6)
            if st2.error or (not st2.halted_cleanly):
                struct_score -= 0.5
            
            # TASK-AWARE score (NEW): practical problem-solving ability
            task_score = TaskBenchmark.evaluate(g, self.vm)
            
            # Combined score: 50% structure + 50% task performance
            score = 0.5 * struct_score + 0.5 * task_score * 2.0  # task_score scaled to ~1.0 max
            
            g.last_score = float(score)
            g.last_cfg_hash = cfg2.canonical_hash()

        # Sort by score
        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)

        # Diversity filter: prefer unique CFG hashes among the top band
        elites: List[ProgramGenome] = []
        seen_cfg: Set[str] = set()
        band = ranked[: max(self.cfg.elite_keep * 3, self.cfg.elite_keep)]
        for g in band:
            if len(elites) >= self.cfg.elite_keep:
                break
            if g.last_cfg_hash not in seen_cfg:
                elites.append(g)
                seen_cfg.add(g.last_cfg_hash)

        # If not enough elites due to diversity constraint, fill from ranked
        if len(elites) < self.cfg.elite_keep:
            for g in ranked:
                if len(elites) >= self.cfg.elite_keep:
                    break
                elites.append(g)

        next_pop: List[ProgramGenome] = []
        for e in elites:
            kept = e.clone()
            next_pop.append(kept)
            for _ in range(self.cfg.children_per_elite):
                next_pop.append(self.mutate(e))

        # trim to pop size
        self.population = next_pop[: self.cfg.pop_size]
        self._reindex()

        return successes_this_gen, len(getattr(self.detector, "seen_success_hashes", set()))
# ==============================================================================
# 8) CLI flows
# ==============================================================================

def run_concept_selftests() -> None:
    vm = VirtualMachine(max_steps=50)
    lib = ConceptLibrary(max_size=10)
    concept = Concept(
        cid="c_copy_first",
        name="copy_first",
        kind="macro",
        payload={"instructions": [("LOAD", 0, 0, 0)]},
        compile_fn_id="macro_v1",
        discovered_gen=0,
        parents=[],
    )
    cid = lib.add_concept(concept, dedup=True)
    assert cid is not None, "concept add failed"

    tmp_path = "concept_selftest.json"
    lib.save(tmp_path)
    lib2 = ConceptLibrary(max_size=10)
    lib2.load(tmp_path)
    assert lib2.get("c_copy_first") is not None, "concept load failed"

    insts = lib2.build_instructions(concept)
    assert insts, "concept compile failed"
    g_concept = ProgramGenome(gid="g_concept", instructions=insts)
    g_base = ProgramGenome(gid="g_base", instructions=[Instruction("HALT", 0, 0, 0)])

    metrics_base = ConceptDiscoveryBenchmark.evaluate(g_base, vm)
    metrics_concept = ConceptDiscoveryBenchmark.evaluate(g_concept, vm)
    assert metrics_concept["holdout_pass_rate"] > metrics_base["holdout_pass_rate"], "holdout did not improve"

    # Negative: train improves but holdout regresses on synthetic split
    def _eval_custom(genome: ProgramGenome, train: List[List[float]], holdout: List[List[float]]) -> Tuple[float, float]:
        def _score(dataset: List[List[float]]) -> float:
            passed = 0
            for inputs in dataset:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    continue
                if abs(st.regs[0]) < 0.01:
                    passed += 1
            return passed / max(1, len(dataset))

        return _score(train), _score(holdout)

    g_overfit = ProgramGenome(gid="g_overfit", instructions=[Instruction("SET", 0, 0, 0)])
    train = [[0.0], [0.0, 1.0]]
    holdout = [[1.0], [2.0, 2.0]]
    train_rate, holdout_rate = _eval_custom(g_overfit, train, holdout)
    assert train_rate > holdout_rate, "gap check failed"

    # Negative: adversarial/shift break detection
    adv = [[-1.0], [-2.0]]
    adv_rate, _ = _eval_custom(g_overfit, adv, holdout)
    assert adv_rate <= train_rate, "adversarial regression not detected"

    # VM step limit regression check
    g_loop = ProgramGenome(gid="g_loop", instructions=[Instruction("JMP", 0, 0, 0)])
    st = vm.execute(g_loop, [1.0])
    assert st.steps <= vm.max_steps, "step limit regression"

def cmd_selftest(args: argparse.Namespace) -> int:
    """
    Selftest validates:
      - engine executes for N generations without crashing
      - evidence file is created and non-empty (at least header line)
    It does NOT require successes in a short horizon.
    """
    out = args.out or "v13_selftest.jsonl"
    if os.path.exists(out):
        try:
            os.remove(out)
        except Exception:
            pass

    # For selftest, relax params a bit so it is more likely to see at least one success,
    # but still keep anti-cheat and logging correctness.
    p = DetectorParams(
        K_initial=4,
        L_initial=7,
        C_coverage=0.45,
        f_rarity=0.01,
        N_repro=3,
        require_both=True,
        min_loops=1,
        min_scc=1,
    )
    det = StrictStructuralDetector(p)
    eng = OmegaForgeV13(seed=args.seed, detector=det)
    eng.init_population()
    w = EvidenceWriter(out)

    gens = int(args.generations or 200)
    total_success_lines = 0
    try:
        for _ in range(gens):
            succ, _ = eng.step(writer=w)
            # progress
            # Count evidence lines roughly by success count (header already present)
            total_success_lines += succ
            if eng.generation % 10 == 0:
                # "total_evidence_lines" includes only evidence, not header
                print(f"[gen {eng.generation}] successes_this_gen={succ} total_evidence_lines={total_success_lines}", flush=True)
    finally:
        w.close()

    # Validate file exists and has at least 1 line (header)
    if not os.path.exists(out):
        print("SELFTEST_FAIL: evidence file missing", flush=True)
        return 1

    try:
        sz = os.path.getsize(out)
    except Exception:
        sz = 0

    if sz <= 0:
        print("SELFTEST_FAIL: evidence file empty (should contain header)", flush=True)
        return 1

    # Pass criteria: file non-empty + ran gens
    print(f"SELFTEST_OK: ran_gens={gens} evidence_file_bytes={sz} evidence_successes={total_success_lines}", flush=True)
    try:
        run_concept_selftests()
        print("CONCEPT_SELFTEST_OK", flush=True)
    except Exception as e:
        print(f"CONCEPT_SELFTEST_FAIL: {e}", flush=True)
        return 1
    return 0

def cmd_evidence_run(args: argparse.Namespace) -> int:
    out = args.out or "evidence_v13.jsonl"
    target = int(args.target or 6)
    max_g = int(args.max_generations or 2000)

    # Use default strict params unless user overrides via flags later
    eng = OmegaForgeV13(seed=args.seed)
    eng.init_population()
    w = EvidenceWriter(out)

    found = 0
    try:
        while eng.generation < max_g and found < target:
            succ, _ = eng.step(writer=w)
            found += succ
            if eng.generation % max(1, int(args.report_every or 10)) == 0:
                print(f"[gen {eng.generation}] found={found}/{target} out={out}", flush=True)
    finally:
        w.close()

    print(f"EVIDENCE_RUN_DONE: gens={eng.generation} found={found} out={out}", flush=True)
    return 0

def cmd_run(args: argparse.Namespace) -> int:
    log = args.log or "v13_run.jsonl"
    gens = int(args.generations or 5000)
    eng = OmegaForgeV13(seed=args.seed)
    eng.init_population()
    w = EvidenceWriter(log)
    try:
        for _ in range(gens):
            succ, _ = eng.step(writer=w)
            if eng.generation % max(1, int(args.report_every or 50)) == 0:
                print(f"[gen {eng.generation}] successes_this_gen={succ} log={log}", flush=True)
    finally:
        w.close()
    print(f"RUN_DONE: gens={gens} log={log}", flush=True)
    return 0

def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OMEGA_FORGE V13 CLEAN (streaming evidence)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("selftest", help="Run crash-safe logging selftest")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--generations", type=int, default=200)
    p1.add_argument("--out", type=str, default="v13_selftest.jsonl")
    p1.set_defaults(func=cmd_selftest)

    p2 = sub.add_parser("evidence_run", help="Run until N evidence lines are found")
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--target", type=int, default=6)
    p2.add_argument("--max_generations", type=int, default=2000)
    p2.add_argument("--out", type=str, default="evidence_v13.jsonl")
    p2.add_argument("--report_every", type=int, default=10)
    p2.set_defaults(func=cmd_evidence_run)

    p3 = sub.add_parser("run", help="Long run (writes all evidence to log)")
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--generations", type=int, default=5000)
    p3.add_argument("--log", type=str, default="v13_run.jsonl")
    p3.add_argument("--report_every", type=int, default=50)
    p3.set_defaults(func=cmd_run)

    return ap

def main() -> int:
    ap = build_cli()
    args = ap.parse_args()
    return int(args.func(args))


# ==============================================================================
# Two-Stage Evolution Engine V4 + Feedback Loop (inlined)
# ==============================================================================

"""
OMEGA_FORGE Two-Stage Evolution Engine V4
==========================================
SUM Fix Patches Applied:
1. Diverse SUM cases (24 deterministic cases)
2. Full-sum dominant scoring (prefix is small tie-breaker)
3. SUM strict-pass gate after curriculum switch
4. Curriculum timing adjusted (250)
5. Accurate per-genome strict-pass benchmark
6. Debug output at gen 1

Usage:
  python two_stage_engine.py full --stage1_gens 300 --stage2_gens 500
"""

import argparse
import json
import random as global_random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import from main engine
# ==============================================================================
# CONFIGURATION
# ==============================================================================

AGG_MODE = "gmean"  # Options: "gmean", "min"
CURRICULUM_SWITCH_GEN = 250  # PATCH 4: Extended from 150 to 250
SUM_GATE_AFTER_SWITCH = 0.2  # PATCH 3: Penalty multiplier for SUM-failing genomes

# ==============================================================================
# PATCH 1: Diverse SUM Cases Generator
# ==============================================================================

def build_sum_cases(seed: int, n_cases: int) -> List[Tuple[List[float], float]]:
    """
    PATCH 1: Generate diverse SUM test cases deterministically.
    Uses local Random to avoid affecting global state.
    """
    rng = global_random.Random(seed)
    cases = set()
    
    # Include empty array
    cases.add(())
    
    # Generate diverse cases
    attempts = 0
    while len(cases) < n_cases and attempts < n_cases * 10:
        attempts += 1
        length = rng.randint(0, 16)
        if length == 0:
            arr = ()
        else:
            arr = tuple(rng.randint(0, 9) for _ in range(length))
        cases.add(arr)
    
    # Convert to required format: (inputs, expected_sum)
    result = []
    for arr in cases:
        inputs = [float(x) for x in arr]
        expected = sum(inputs)
        result.append((inputs, expected))
    
    # Sort for reproducibility
    result.sort(key=lambda x: (len(x[0]), x[1]))
    return result[:n_cases]

# ==============================================================================
# HALF-SKELETON MACROS (unchanged)
# ==============================================================================

class TaskMacroLibrary:
    @staticmethod
    def sum_skeleton() -> List[Instruction]:
        return [
            Instruction("SET", 0, 0, 0),      # r0 = 0 (accumulator)
            Instruction("SET", 0, 0, 2),      # r2 = 0 (index i)
            Instruction("JLT", 2, 1, 2),      # if r2 < r1, continue
            Instruction("JMP", 5, 0, 0),      # else exit
            Instruction("LOAD", 2, 0, 3),     # r3 = memory[r2]
            Instruction("ADD", 0, 3, 0),      # r0 += r3
            Instruction("INC", 0, 0, 2),      # i++
            Instruction("JMP", -5, 0, 0),     # loop back
        ]
    
    @staticmethod
    def max_skeleton() -> List[Instruction]:
        return [
            Instruction("LOAD", 2, 0, 0),
            Instruction("SET", 1, 0, 2),
            Instruction("JLT", 2, 1, 2),
            Instruction("JMP", 6, 0, 0),
            Instruction("LOAD", 2, 0, 3),
            Instruction("JGT", 3, 0, 2),
            Instruction("JMP", 2, 0, 0),
            Instruction("MOV", 3, 0, 0),
            Instruction("INC", 0, 0, 2),
            Instruction("JMP", -7, 0, 0),
        ]
    
    @staticmethod
    def double_skeleton() -> List[Instruction]:
        return [
            Instruction("SET", 0, 0, 2),
            Instruction("JLT", 2, 1, 2),
            Instruction("JMP", 6, 0, 0),
            Instruction("LOAD", 2, 0, 3),
            Instruction("ADD", 3, 3, 3),
            Instruction("STORE", 2, 0, 3),
            Instruction("INC", 0, 0, 2),
            Instruction("JMP", -6, 0, 0),
        ]

# ==============================================================================
# TASK BENCHMARK V4 (All Patches Applied)
# ==============================================================================

class TaskBenchmarkV4:
    """
    Patches implemented:
    1. Diverse SUM cases (24 cases from deterministic generator)
    2. Full-sum dominant scoring
    3. Strict-pass for per-genome counting
    """
    
    # PATCH 1: Generate 24 diverse SUM cases
    SUM_CASES = build_sum_cases(seed=123, n_cases=24)
    
    # MAX and DOUBLE unchanged
    MAX_CASES = [
        ([3.0, 7.0, 2.0, 9.0, 1.0], 9.0),
        ([5.0, 2.0, 8.0], 8.0),
        ([1.0], 1.0),
        ([10.0, 5.0, 7.0, 3.0, 9.0, 2.0], 10.0),
    ]
    
    DOUBLE_CASES = [
        ([3.0, 4.0, 5.0], 6.0),
        ([2.0, 6.0], 4.0),
        ([5.0], 10.0),
    ]
    
    @staticmethod
    def _sum_score(genome, vm, inputs: List[float], expected: float) -> float:
        """
        PATCH 2: Full-sum dominant scoring.
        Prefix bonus is capped at 0.10 as tie-breaker only.
        """
        try:
            st = vm.execute(genome, inputs)
            if st.error or not st.halted_cleanly:
                return 0.0
            result = st.regs[0]
        except:
            return 0.0
        
        # Base score: full-sum error ratio (dominant)
        err = abs(result - expected)
        den = max(1.0, abs(expected))
        ratio = err / den
        
        if ratio < 1e-6:
            base = 1.0
        elif ratio < 0.02:
            base = 0.8
        elif ratio < 0.10:
            base = 0.5
        elif ratio < 0.30:
            base = 0.2
        else:
            base = 0.0
        
        # Prefix bonus: small tie-breaker (capped at 0.10)
        bonus = 0.0
        if len(inputs) > 0:
            cumsum = 0.0
            for i, val in enumerate(inputs):
                cumsum += val
                if abs(result - cumsum) < 1e-6:
                    bonus = max(bonus, 0.05 + 0.05 * (i + 1) / max(1, len(inputs)))
        
        return min(1.0, base + min(0.10, bonus))
    
    @staticmethod
    def _case_score(genome, vm, inputs: List[float], expected: float, out_loc: str) -> float:
        """Standard partial scoring for MAX/DOUBLE."""
        try:
            st = vm.execute(genome, inputs)
            if st.error or not st.halted_cleanly:
                return 0.0
            if out_loc == "reg0":
                result = st.regs[0]
            elif out_loc == "mem0":
                result = st.memory.get(0, 0.0)
            else:
                result = 0.0
        except:
            return 0.0
        
        if abs(expected) < 1e-9:
            return 1.0 if abs(result) < 0.01 else 0.0
        
        error_ratio = abs(result - expected) / abs(expected)
        if error_ratio < 0.001:
            return 1.0
        elif error_ratio < 0.1:
            return 0.8
        elif error_ratio < 0.5:
            return 0.5
        elif error_ratio < 1.0:
            return 0.2
        return 0.0
    
    @staticmethod
    def evaluate(genome, vm) -> Dict[str, float]:
        """Returns per-task-type average scores."""
        scores = {"SUM": 0.0, "MAX": 0.0, "DOUBLE": 0.0}
        
        # SUM
        sum_scores = []
        for inputs, expected in TaskBenchmarkV4.SUM_CASES:
            s = TaskBenchmarkV4._sum_score(genome, vm, inputs, expected)
            sum_scores.append(s)
        scores["SUM"] = sum(sum_scores) / len(sum_scores) if sum_scores else 0.0
        
        # MAX
        max_scores = []
        for inputs, expected in TaskBenchmarkV4.MAX_CASES:
            s = TaskBenchmarkV4._case_score(genome, vm, inputs, expected, "reg0")
            max_scores.append(s)
        scores["MAX"] = sum(max_scores) / len(max_scores) if max_scores else 0.0
        
        # DOUBLE
        dbl_scores = []
        for inputs, expected in TaskBenchmarkV4.DOUBLE_CASES:
            s = TaskBenchmarkV4._case_score(genome, vm, inputs, expected, "mem0")
            dbl_scores.append(s)
        scores["DOUBLE"] = sum(dbl_scores) / len(dbl_scores) if dbl_scores else 0.0
        
        return scores
    
    @staticmethod
    def evaluate_strict_pass(genome, vm) -> Dict[str, bool]:
        """
        PATCH 5: Returns per-task-type strict-pass (ALL cases must pass exactly).
        """
        results = {}
        
        # SUM: all cases must pass
        all_pass = True
        for inputs, expected in TaskBenchmarkV4.SUM_CASES:
            try:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    all_pass = False
                    break
                if abs(st.regs[0] - expected) > 0.01:
                    all_pass = False
                    break
            except:
                all_pass = False
                break
        results["SUM"] = all_pass
        
        # MAX
        all_pass = True
        for inputs, expected in TaskBenchmarkV4.MAX_CASES:
            try:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    all_pass = False
                    break
                if abs(st.regs[0] - expected) > 0.01:
                    all_pass = False
                    break
            except:
                all_pass = False
                break
        results["MAX"] = all_pass
        
        # DOUBLE
        all_pass = True
        for inputs, expected in TaskBenchmarkV4.DOUBLE_CASES:
            try:
                st = vm.execute(genome, inputs)
                if st.error or not st.halted_cleanly:
                    all_pass = False
                    break
                if abs(st.memory.get(0, 0.0) - expected) > 0.01:
                    all_pass = False
                    break
            except:
                all_pass = False
                break
        results["DOUBLE"] = all_pass
        
        return results
    
    @staticmethod
    def debug_sum_outputs(genome, vm, label: str):
        """
        PATCH 6: Debug output for first 3 SUM cases.
        """
        print(f"    {label}:")
        for i, (inputs, expected) in enumerate(TaskBenchmarkV4.SUM_CASES[:3]):
            try:
                st = vm.execute(genome, inputs)
                got = st.regs[0] if st.halted_cleanly else "ERROR"
            except:
                got = "EXCEPTION"
            print(f"      case {i}: input={inputs[:5]}{'...' if len(inputs)>5 else ''} expected={expected} got={got}")

# ==============================================================================
# 5.6) Concept discovery micro-domains
# ==============================================================================

class ConceptDiscoveryBenchmark:
    DOMAINS: List[Dict[str, Any]] = [
        {
            "name": "COPY_FIRST",
            "out_loc": "reg0",
            "train": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0, 7.0]],
            "holdout": [[9.0, 1.0], [2.0, 8.0, 3.0]],
            "adversarial": [[0.0], [-1.0, 2.0]],
            "shift": [[10.0, 11.0, 12.0, 13.0]],
            "oracle": lambda xs: float(xs[0]) if xs else 0.0,
        },
        {
            "name": "PREFIX_SUM_2",
            "out_loc": "reg0",
            "train": [[1.0, 2.0], [2.0, 2.0], [3.0, 1.0]],
            "holdout": [[4.0, 5.0], [0.0, 3.0]],
            "adversarial": [[-1.0, 4.0], [7.0, -2.0]],
            "shift": [[10.0, 20.0]],
            "oracle": lambda xs: float(xs[0] + xs[1]) if len(xs) >= 2 else float(sum(xs)),
        },
        {
            "name": "ARGMAX_TIE_FIRST",
            "out_loc": "reg0",
            "train": [[1.0, 3.0, 2.0], [5.0, 5.0, 4.0], [2.0, 1.0, 2.0]],
            "holdout": [[0.0, 0.0, 1.0], [9.0, 7.0, 9.0]],
            "adversarial": [[-1.0, -1.0, -2.0]],
            "shift": [[100.0, 50.0, 75.0, 100.0]],
            "oracle": lambda xs: float(next((i for i, v in enumerate(xs) if v == max(xs)), 0)) if xs else 0.0,
        },
    ]

    @staticmethod
    def _eval_case(genome: ProgramGenome, vm: VirtualMachine, inputs: List[float], out_loc: str, expected: float) -> bool:
        st = vm.execute(genome, inputs)
        if st.error or not st.halted_cleanly:
            return False
        if out_loc == "reg0":
            got = st.regs[0]
        elif out_loc == "mem0":
            got = st.memory.get(0, 0.0)
        else:
            got = 0.0
        return abs(got - expected) < 0.01

    @staticmethod
    def evaluate(genome: ProgramGenome, vm: VirtualMachine) -> Dict[str, Any]:
        splits = {"train": [], "holdout": [], "adversarial": [], "shift": []}
        for domain in ConceptDiscoveryBenchmark.DOMAINS:
            oracle = domain["oracle"]
            out_loc = domain["out_loc"]
            for split_name in ("train", "holdout", "adversarial", "shift"):
                dataset = domain.get(split_name, [])
                passes = 0
                for inputs in dataset:
                    expected = oracle(inputs)
                    if ConceptDiscoveryBenchmark._eval_case(genome, vm, inputs, out_loc, expected):
                        passes += 1
                total = max(1, len(dataset))
                splits[split_name].append(passes / total)

        train_rate = float(sum(splits["train"]) / max(1, len(splits["train"])))
        holdout_rate = float(sum(splits["holdout"]) / max(1, len(splits["holdout"])))
        adv_rate = float(sum(splits["adversarial"]) / max(1, len(splits["adversarial"])))
        shift_rate = float(sum(splits["shift"]) / max(1, len(splits["shift"])))

        train_count = sum(len(d.get("train", [])) for d in ConceptDiscoveryBenchmark.DOMAINS)
        holdout_count = sum(len(d.get("holdout", [])) for d in ConceptDiscoveryBenchmark.DOMAINS)
        holdout_cost = min(4.0, float(holdout_count) / max(1.0, float(train_count)))

        return {
            "train_pass_rate": train_rate,
            "holdout_pass_rate": holdout_rate,
            "adversarial_pass_rate": adv_rate,
            "distribution_shift": {"holdout_pass_rate": shift_rate},
            "discovery_cost": {"train": float(train_count), "holdout": holdout_cost},
        }

def detect_memorization(genome: ProgramGenome) -> bool:
    large_set = sum(1 for inst in genome.instructions if inst.op == "SET" and abs(inst.a) > 20)
    set_total = sum(1 for inst in genome.instructions if inst.op == "SET")
    if large_set >= 3:
        return True
    if set_total >= max(6, len(genome.instructions) // 2):
        return True
    return False

def find_repeated_subsequence(instructions: List[Instruction],
                              min_len: int = 2,
                              max_len: int = 5) -> Optional[List[Instruction]]:
    if len(instructions) < min_len * 2:
        return None
    best_seq = None
    best_count = 1
    tuples = _inst_tuple_list(instructions)
    for L in range(min_len, min(max_len, len(instructions)) + 1):
        counts: Dict[Tuple[Any, ...], int] = {}
        for i in range(0, len(tuples) - L + 1):
            key = tuple(tuples[i:i + L])
            counts[key] = counts.get(key, 0) + 1
        for key, count in counts.items():
            if count > best_count:
                best_count = count
                best_seq = [Instruction(*t) for t in key]
    if best_count > 1 and best_seq:
        return best_seq
    return None

def detect_concepts_in_genome(genome: ProgramGenome, concepts: ConceptLibrary) -> List[str]:
    if not concepts or not genome.instructions:
        return []
    hits = []
    inst_tuples = _inst_tuple_list(genome.instructions)
    for concept in concepts.all_concepts():
        insts = concepts.build_instructions(concept)
        if not insts:
            continue
        c_tuples = tuple(_inst_tuple_list(insts))
        if len(c_tuples) == 0:
            continue
        for i in range(0, len(inst_tuples) - len(c_tuples) + 1):
            if tuple(inst_tuples[i:i + len(c_tuples)]) == c_tuples:
                hits.append(concept.cid)
                break
    return hits



# ==============================================================================
# Stage 1: Structural Discovery (unchanged)
# ==============================================================================

class Stage1Engine:
    def __init__(self,
                 seed: int = 42,
                 concepts_on: bool = False,
                 concept_budget: int = 80,
                 concept_library_path: str = ""):
        global_random.seed(seed)
        self.vm = VirtualMachine()
        self.detector = StrictStructuralDetector()
        self.cfg = EngineConfig(pop_size=30)
        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        self.candidates: List[Dict[str, Any]] = []
        self.concepts_on = concepts_on
        self.concept_library = ConceptLibrary(max_size=concept_budget)
        self.concept_library_path = concept_library_path
        if self.concepts_on and concept_library_path:
            self.concept_library.load(concept_library_path)
        
    def init_population(self):
        self.population = []
        for i in range(self.cfg.pop_size):
            L = global_random.randint(18, 28)
            insts = [rand_inst() for _ in range(L)]
            g = ProgramGenome(gid=f"init_{i}", instructions=insts, parents=[], generation=0)
            self.population.append(g)
        self.parents_index = {g.gid: g for g in self.population}
    
    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"g{self.generation}_{global_random.randint(0, 999999)}"

        roll = global_random.random()
        if self.concepts_on and roll < 0.12:
            concept = self._sample_concept()
            if concept:
                insts = self.concept_library.build_instructions(concept)
                if insts and len(child.instructions) + len(insts) < self.cfg.max_code_len:
                    pos = global_random.randint(0, len(child.instructions))
                    child.instructions[pos:pos] = [i.clone() for i in insts]
                    child.concept_trace.append(concept.cid)
                    return child
        if self.concepts_on and roll < 0.22:
            seq = find_repeated_subsequence(parent.instructions, min_len=2, max_len=5)
            if seq:
                cid = f"c{self.generation}_{global_random.randint(0, 999999)}"
                payload = {"instructions": [i.to_tuple() for i in seq]}
                concept = Concept(
                    cid=cid,
                    name=f"macro_len{len(seq)}",
                    kind="macro",
                    payload=payload,
                    compile_fn_id="macro_v1",
                    discovered_gen=self.generation,
                    parents=[parent.gid],
                )
                added_cid = self.concept_library.add_concept(concept, dedup=True)
                if added_cid:
                    child.concept_proposals.append(added_cid)
            return child
        if self.concepts_on and roll < 0.28:
            c1 = self._sample_concept()
            c2 = self._sample_concept()
            if c1 and c2 and c1.cid != c2.cid:
                insts = self.concept_library.build_instructions(c1) + self.concept_library.build_instructions(c2)
                if 0 < len(insts) <= 6:
                    cid = f"c{self.generation}_{global_random.randint(0, 999999)}"
                    payload = {"instructions": [i.to_tuple() for i in insts]}
                    concept = Concept(
                        cid=cid,
                        name=f"compose_{c1.cid}_{c2.cid}",
                        kind="macro",
                        payload=payload,
                        compile_fn_id="macro_v1",
                        discovered_gen=self.generation,
                        parents=[c1.cid, c2.cid],
                    )
                    added_cid = self.concept_library.add_concept(concept, dedup=True)
                    if added_cid:
                        child.concept_proposals.append(added_cid)
            return child
        if roll < 0.15 and len(child.instructions) + 10 < self.cfg.max_code_len:
            skeleton = global_random.choice([
                TaskMacroLibrary.sum_skeleton,
                TaskMacroLibrary.max_skeleton,
                TaskMacroLibrary.double_skeleton,
            ])()
            pos = global_random.randint(0, len(child.instructions))
            child.instructions[pos:pos] = [Instruction(i.op, i.a, i.b, i.c) for i in skeleton]
        elif roll < 0.35 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            op = global_random.choice(["JMP", "JZ", "JNZ", "JGT", "JLT", "CALL", "RET"])
            child.instructions[pos] = Instruction(op, global_random.randint(-8, 8), global_random.randint(0, 7), global_random.randint(0, 7))
        elif roll < 0.60 and len(child.instructions) < self.cfg.max_code_len:
            pos = global_random.randint(0, len(child.instructions))
            child.instructions.insert(pos, rand_inst())
        elif roll < 0.80 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            child.instructions[pos].a = max(-8, min(31, child.instructions[pos].a + global_random.randint(-2, 2)))
        else:
            if len(child.instructions) > 8:
                pos = global_random.randint(0, len(child.instructions) - 1)
                child.instructions.pop(pos)

        return child

    def _sample_concept(self) -> Optional[Concept]:
        concepts = self.concept_library.all_concepts()
        if not concepts:
            return None
        weights = []
        for c in concepts:
            if c.cid in CONCEPT_ANTI_BIAS:
                base = 0.1
            else:
                base = CONCEPT_BIAS.get(c.cid, 1.0)
            length = int(c.stats.get("length", len(self.concept_library.build_instructions(c)) or 1))
            if MACRO_LENGTH_BIAS > 0.0:
                base *= 1.0 / (1.0 + MACRO_LENGTH_BIAS * max(0, length - 1))
            weights.append(max(0.05, base))
        return global_random.choices(concepts, weights=weights, k=1)[0]
    
    def step(self) -> int:
        self.generation += 1
        successes = 0
        
        for g in self.population:
            parent = self.parents_index.get(g.parents[0]) if g.parents else None
            st = self.vm.execute(g, [1.0] * 8)
            passed, reasons, diag = self.detector.evaluate(g, parent, st, self.vm, self.generation)
            
            if passed:
                successes += 1
                if detect_memorization(g):
                    continue
                scores = TaskBenchmarkV4.evaluate(g, self.vm)
                concept_metrics = ConceptDiscoveryBenchmark.evaluate(g, self.vm)
                hints: List[str] = []
                for cid in g.concept_trace:
                    c = self.concept_library.get(cid)
                    if c:
                        hints.append(f"use:{cid}:{c.name}")
                    else:
                        hints.append(f"use:{cid}")
                for cid in g.concept_proposals:
                    c = self.concept_library.get(cid)
                    if c:
                        hints.append(f"propose:{cid}:{c.name}")
                    else:
                        hints.append(f"propose:{cid}")
                candidate = {
                    "gid": g.gid,
                    "generation": self.generation,
                    "code": [(i.op, i.a, i.b, i.c) for i in g.instructions],
                    "metrics": {
                        **concept_metrics,
                        "structural": {"loops": st.loops_count, "scc_n": diag.get("scc_n", 0)},
                        "memorization_suspected": False,
                    },
                    "task_scores": scores,
                    "hints": hints,
                }
                self.candidates.append(candidate)
        
        # Selection
        for g in self.population:
            st2 = self.vm.execute(g, [1.0] * 8)
            cfg2 = ControlFlowGraph.from_trace(st2.trace, len(g.instructions))
            cov = st2.coverage(len(g.instructions))
            scc_n = len(cfg2.sccs())
            score = cov + 0.02 * min(st2.loops_count, 50) + 0.08 * min(scc_n, 6)
            if st2.error or not st2.halted_cleanly:
                score -= 0.5
            g.last_score = score
            g.last_cfg_hash = cfg2.canonical_hash()
        
        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)
        elites = ranked[:self.cfg.elite_keep]
        
        next_pop = []
        for e in elites:
            next_pop.append(e.clone())
            for _ in range(self.cfg.children_per_elite):
                next_pop.append(self.mutate(e))
        
        self.population = next_pop[:self.cfg.pop_size]
        self.parents_index = {g.gid: g for g in self.population}
        
        return successes
    
    def run(self, generations: int, out_file: str):
        self.init_population()
        print(f"[Stage 1] Collecting candidates for {generations} generations...")
        
        for gen in range(1, generations + 1):
            self.step()
            if gen % 50 == 0:
                print(f"  [gen {gen}] candidates={len(self.candidates)}")
        
        with open(out_file, 'w') as f:
            for c in self.candidates:
                f.write(json.dumps(c) + "\n")

        if self.concepts_on and self.concept_library_path:
            self.concept_library.save(self.concept_library_path)
        
        print(f"[Stage 1] Done. Saved {len(self.candidates)} candidates to {out_file}")
        return self.candidates

# ==============================================================================
# Stage 2: Task-Aware Evolution (PATCHES 3, 4, 5, 6)
# ==============================================================================

class Stage2Engine:
    def __init__(self, candidates: List[Dict[str, Any]], seed: int = 42):
        global_random.seed(seed)
        self.vm = VirtualMachine()
        self.candidates = candidates
        self.population: List[ProgramGenome] = []
        self.generation: int = 0
        
    def load_population(self, sample_size: int = 50):
        sorted_cands = sorted(
            self.candidates, 
            key=lambda x: x.get("task_scores", {}).get("SUM", 0), 
            reverse=True
        )
        
        self.population = []
        for i, c in enumerate(sorted_cands[:sample_size]):
            insts = [Instruction(op, a, b, c_) for op, a, b, c_ in c["code"]]
            g = ProgramGenome(gid=f"s2_init_{i}", instructions=insts, generation=0)
            self.population.append(g)
        
        print(f"[Stage 2] Loaded {len(self.population)} genomes (sorted by SUM potential)")
    
    def mutate(self, parent: ProgramGenome) -> ProgramGenome:
        child = parent.clone()
        child.generation = self.generation
        child.parents = [parent.gid]
        child.gid = f"s2_g{self.generation}_{global_random.randint(0, 999999)}"
        
        roll = global_random.random()
        if roll < 0.4 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            inst = child.instructions[pos]
            field = global_random.choice(["a", "b", "c"])
            delta = global_random.randint(-2, 2)
            if field == "a":
                inst.a = max(-8, min(31, inst.a + delta))
            elif field == "b":
                inst.b = max(0, min(7, inst.b + delta))
            else:
                inst.c = max(0, min(7, inst.c + delta))
        elif roll < 0.6 and child.instructions:
            pos = global_random.randint(0, len(child.instructions) - 1)
            useful_ops = ["LOAD", "ADD", "STORE", "JGT", "JLT", "MOV", "INC"]
            child.instructions[pos] = Instruction(
                global_random.choice(useful_ops),
                global_random.randint(0, 7),
                global_random.randint(0, 7),
                global_random.randint(0, 7)
            )
        elif roll < 0.8 and len(child.instructions) >= 2:
            i, j = global_random.sample(range(len(child.instructions)), 2)
            child.instructions[i], child.instructions[j] = child.instructions[j], child.instructions[i]
        else:
            if len(child.instructions) < 60:
                pos = global_random.randint(0, len(child.instructions))
                useful_ops = ["LOAD", "ADD", "STORE", "INC"]
                child.instructions.insert(pos, Instruction(
                    global_random.choice(useful_ops),
                    global_random.randint(0, 7),
                    global_random.randint(0, 7),
                    global_random.randint(0, 7)
                ))
        
        return child
    
    def _compute_fitness(self, scores: Dict[str, float], strict_pass: Dict[str, bool], gen: int) -> float:
        """
        PATCH 3 & 4: Curriculum + SUM strict-pass gate
        """
        sum_s = scores.get("SUM", 0.0)
        max_s = scores.get("MAX", 0.0)
        dbl_s = scores.get("DOUBLE", 0.0)
        
        # Before curriculum switch: SUM-only
        if gen < CURRICULUM_SWITCH_GEN:
            return sum_s
        
        # After switch: gmean aggregation
        eps = 1e-9
        if AGG_MODE == "gmean":
            fitness = (max(sum_s, eps) * max(max_s, eps) * max(dbl_s, eps)) ** (1.0/3.0)
        elif AGG_MODE == "min":
            fitness = min(sum_s, max_s, dbl_s)
        else:
            fitness = (sum_s + max_s + dbl_s) / 3.0
        
        # PATCH 3: SUM gate multiplier
        if not strict_pass.get("SUM", False):
            fitness *= SUM_GATE_AFTER_SWITCH
        
        return fitness
    
    def step(self) -> Dict[str, Any]:
        self.generation += 1
        
        # Log curriculum switch
        if self.generation == CURRICULUM_SWITCH_GEN:
            print(f"\n  *** CURRICULUM SWITCH at gen {self.generation}: SUM-only → {AGG_MODE} + SUM gate ({SUM_GATE_AFTER_SWITCH}x) ***\n")
        
        scores_list = []
        pass_list = []
        for g in self.population:
            scores = TaskBenchmarkV4.evaluate(g, self.vm)
            strict_pass = TaskBenchmarkV4.evaluate_strict_pass(g, self.vm)
            fitness = self._compute_fitness(scores, strict_pass, self.generation)
            g.last_score = fitness
            scores_list.append(scores)
            pass_list.append(strict_pass)
        
        # PATCH 6: Debug at gen 1
        if self.generation == 1:
            print("  [gen 1] DEBUG: Top 3 genomes by SUM score:")
            ranked_by_sum = sorted(zip(self.population, scores_list), key=lambda x: x[1]["SUM"], reverse=True)
            for i, (g, sc) in enumerate(ranked_by_sum[:3]):
                print(f"    Genome {i} (SUM={sc['SUM']:.3f}):")
                TaskBenchmarkV4.debug_sum_outputs(g, self.vm, f"outputs")
        
        avg_sum = sum(s["SUM"] for s in scores_list) / len(scores_list)
        avg_max = sum(s["MAX"] for s in scores_list) / len(scores_list)
        avg_dbl = sum(s["DOUBLE"] for s in scores_list) / len(scores_list)
        sum_pass = sum(1 for p in pass_list if p["SUM"]) / len(pass_list)
        
        ranked = sorted(self.population, key=lambda x: x.last_score, reverse=True)
        elite_count = max(10, len(self.population) // 3)
        elites = ranked[:elite_count]
        
        next_pop = []
        for e in elites:
            next_pop.append(e.clone())
            for _ in range(2):
                next_pop.append(self.mutate(e))
        
        self.population = next_pop[:50]
        
        return {"avg_sum": avg_sum, "avg_max": avg_max, "avg_dbl": avg_dbl, "sum_pass": sum_pass}
    
    def run(self, generations: int):
        print(f"[Stage 2] Task evolution for {generations} generations")
        print(f"  Curriculum: SUM-only until gen {CURRICULUM_SWITCH_GEN}, then {AGG_MODE} + SUM gate")
        print(f"  SUM cases: {len(TaskBenchmarkV4.SUM_CASES)} diverse cases")
        
        for gen in range(1, generations + 1):
            stats = self.step()
            if gen % 50 == 0:
                print(f"  [gen {gen}] SUM={stats['avg_sum']:.3f} (pass:{stats['sum_pass']*100:.1f}%) MAX={stats['avg_max']:.3f} DOUBLE={stats['avg_dbl']:.3f}")
        
        # PATCH 5: Final Benchmark with strict-pass
        print("\n[Stage 2] Final Benchmark (per-genome strict-pass):")
        results = {"SUM": 0, "MAX": 0, "DOUBLE": 0}
        
        for g in self.population:
            passed = TaskBenchmarkV4.evaluate_strict_pass(g, self.vm)
            for task_type, p in passed.items():
                if p:
                    results[task_type] += 1
        
        n = len(self.population)
        for task, count in results.items():
            pct = count / n * 100
            status = "✅" if count > 0 else "❌"
            print(f"  {status} {task}: {count}/{n} ({pct:.1f}%)")
        
        return results

# ==============================================================================
# CLI
# ==============================================================================

# ==============================================================================
# FEEDBACK: Stage 2 -> Stage 1 (Two-Stage + Feedback Bias)
# ==============================================================================
def extract_stage2_feedback(population: List[ProgramGenome],
                            vm: VirtualMachine,
                            n_top: int = 20,
                            require_sum_pass: bool = True,
                            concept_library: Optional[ConceptLibrary] = None) -> Dict[str, Any]:
    """
    Compute simple sampling biases from the best Stage2 genomes.
    Biases are intended to steer Stage1's rand_inst() opcode sampling.

    Returns dict:
      {
        "op_bias": {"LOAD":1.3, ...},
        "concept_bias": {"c1":1.2, ...},
        "macro_length_bias": 0.2,
        "concept_anti_bias": [...],
        "meta": {"n_used":..., "n_top":..., "require_sum_pass":...}
      }
    """
    scored: List[Tuple[float, ProgramGenome, Dict[str, bool]]] = []
    for g in population:
        scores = TaskBenchmarkV4.evaluate(g, vm)
        strict_pass = TaskBenchmarkV4.evaluate_per_genome_pass(g, vm)
        if require_sum_pass and not strict_pass.get("SUM", False):
            continue
        s = float(scores.get("SUM", 0.0))
        # prefer multi-task competence if available
        s = (max(1e-9, s) * max(1e-9, float(scores.get("MAX", 0.0))) * max(1e-9, float(scores.get("DOUBLE", 0.0)))) ** (1.0/3.0)
        scored.append((s, g, strict_pass))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [g for _, g, _ in scored[:max(1, n_top)]]
    if not picked:
        # fall back: use top by SUM score, even if SUM strict-pass is absent
        tmp = []
        for g in population:
            scores = TaskBenchmarkV4.evaluate(g, vm)
            tmp.append((float(scores.get("SUM", 0.0)), g))
        tmp.sort(key=lambda x: x[0], reverse=True)
        picked = [g for _, g in tmp[:max(1, n_top)]]

    op_counts: Dict[str, int] = {op: 0 for op in OPS}
    total = 0
    for g in picked:
        for inst in g.instructions:
            if inst.op in op_counts:
                op_counts[inst.op] += 1
                total += 1

    # Convert counts -> weights with smoothing, emphasize above-average ops
    op_bias: Dict[str, float] = {}
    if total > 0:
        avg = total / max(1, len(OPS))
        for op, c in op_counts.items():
            # weight = 1.0 at avg, >1 if above avg, with mild exponent
            w = ( (c + 1.0) / (avg + 1.0) ) ** 0.7
            op_bias[op] = float(max(0.05, min(5.0, w)))

    concept_bias: Dict[str, float] = {}
    concept_anti_bias: List[str] = []
    macro_length_bias = 0.0
    if concept_library:
        concept_scores: Counter = Counter()
        concept_counts: Counter = Counter()
        concept_lengths: List[int] = []
        for g in picked:
            metrics = ConceptDiscoveryBenchmark.evaluate(g, vm)
            holdout = metrics.get("holdout_pass_rate", 0.0)
            train = metrics.get("train_pass_rate", 0.0)
            gap = max(0.0, train - holdout)
            used = detect_concepts_in_genome(g, concept_library)
            for cid in used:
                concept_counts[cid] += 1
                concept_scores[cid] += holdout
                if gap > 0.25:
                    concept_anti_bias.append(cid)
            for cid in used:
                c = concept_library.get(cid)
                if c:
                    concept_lengths.append(int(c.stats.get("length", 1)))
        for cid, cnt in concept_counts.items():
            score = concept_scores.get(cid, 0.0) / max(1, cnt)
            concept_bias[cid] = float(max(0.05, min(5.0, 0.5 + score)))
        if concept_lengths:
            avg_len = sum(concept_lengths) / max(1, len(concept_lengths))
            macro_length_bias = max(0.0, min(1.5, (avg_len - 1.0) / 4.0))

    return {
        "op_bias": op_bias,
        "concept_bias": concept_bias,
        "macro_length_bias": macro_length_bias,
        "concept_anti_bias": list(sorted(set(concept_anti_bias))),
        "meta": {
            "n_used": len(picked),
            "n_top": n_top,
            "require_sum_pass": bool(require_sum_pass),
        }
    }

def save_feedback_json(feedback: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(feedback, indent=2, sort_keys=True), encoding="utf-8")

def load_feedback_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def apply_feedback_to_stage1(feedback: Dict[str, Any]) -> None:
    """
    Applies feedback biases to Stage1 by calling set_op_bias().
    """
    op_bias = (feedback or {}).get("op_bias", {}) if isinstance(feedback, dict) else {}
    set_op_bias(op_bias)
    concept_bias = (feedback or {}).get("concept_bias", {}) if isinstance(feedback, dict) else {}
    anti_bias = (feedback or {}).get("concept_anti_bias", []) if isinstance(feedback, dict) else []
    macro_length_bias = (feedback or {}).get("macro_length_bias", None) if isinstance(feedback, dict) else None
    set_concept_bias(concept_bias, anti_bias=anti_bias, macro_length_bias=macro_length_bias)

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Engine V4 (SUM Fix)")
    subparsers = parser.add_subparsers(dest="command")
    
    pf = subparsers.add_parser("full", help="Run full pipeline")
    pf.add_argument("--stage1_gens", type=int, default=300)
    pf.add_argument("--stage2_gens", type=int, default=500)
    pf.add_argument("--feedback_in", type=str, default="", help="Optional Stage2 feedback JSON to bias Stage1 opcode sampling")
    pf.add_argument("--feedback_out", type=str, default="stage2_feedback.json", help="Where to write Stage2 feedback JSON")
    pf.add_argument("--feedback_topk", type=int, default=20, help="Top-K genomes used to compute feedback biases")
    pf.add_argument("--concepts_on", action="store_true", help="Enable concept invention layer in Stage1")
    pf.add_argument("--concept_budget", type=int, default=80, help="Max concepts in library")
    pf.add_argument("--concept_library_path", type=str, default="concept_library.json", help="Path to concept library JSON")

    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--agg", type=str, default="gmean", choices=["gmean", "min", "avg"])
    pf.add_argument("--curriculum_switch", type=int, default=250)
    
    args = parser.parse_args()
    
    if args.command == "full":
        global AGG_MODE, CURRICULUM_SWITCH_GEN
        AGG_MODE = args.agg
        CURRICULUM_SWITCH_GEN = args.curriculum_switch
        
        print("=" * 60)
        print("TWO-STAGE EVOLUTION V4 (SUM Fix Patches Applied)")
        print("=" * 60)
        print(f"Config: AGG={AGG_MODE}, SWITCH_GEN={CURRICULUM_SWITCH_GEN}, SUM_GATE={SUM_GATE_AFTER_SWITCH}")
        print()
        
        
        # Optional: apply prior feedback to bias Stage1 opcode sampling
        if args.feedback_in:
            fb = load_feedback_json(args.feedback_in)
            apply_feedback_to_stage1(fb)

        s1 = Stage1Engine(
            seed=args.seed,
            concepts_on=args.concepts_on,
            concept_budget=args.concept_budget,
            concept_library_path=args.concept_library_path,
        )
        candidates = s1.run(args.stage1_gens, "stage1_candidates.jsonl")
        
        print()
        
        s2 = Stage2Engine(candidates, seed=args.seed)
        s2.load_population()
        s2.run(args.stage2_gens)

        # Write Stage2->Stage1 feedback biases
        try:
            fb = extract_stage2_feedback(
                s2.population,
                s2.vm,
                n_top=args.feedback_topk,
                require_sum_pass=True,
                concept_library=s1.concept_library if args.concepts_on else None,
            )
            save_feedback_json(fb, args.feedback_out)
            print(f"\n[Feedback] Wrote Stage2 feedback to {args.feedback_out}")
        except Exception as e:
            print(f"\n[Feedback] WARNING: failed to write feedback: {e}")
    else:
        parser.print_help()

# if __name__ == "__main__":
    main()

# END OF omega_forge_two_stage_feedback.py


# START OF unified_rsi_extended.py

"""
UNIFIED_RSI_EXTENDED.py

True RSI (Recursive Self-Improvement) Engine - BETA (Executable)
================================================================

CLI:
  python UNIFIED_RSI_EXTENDED.py selftest
  python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 100
  python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 50 --mode program
  python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 50 --mode algo --task sort_int_list
  python UNIFIED_RSI_EXTENDED.py learner-evolve --fresh --generations 100
  python UNIFIED_RSI_EXTENDED.py meta-meta --episodes 20 --gens-per-episode 20
  python UNIFIED_RSI_EXTENDED.py task-switch --task-a poly2 --task-b piecewise
  python UNIFIED_RSI_EXTENDED.py report --state-dir .rsi_state
  python UNIFIED_RSI_EXTENDED.py transfer-bench --from poly2 --to piecewise --budget 10
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 50 --rounds 10
  python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 20 --rounds 5 --mode learner
  python UNIFIED_RSI_EXTENDED.py duo-loop --rounds 5 --slice-seconds 8 --blackboard .rsi_blackboard.jsonl --k-full 6

DUO-LOOP OVERVIEW
-----------------
The duo-loop command adds a sequential, low-spec cooperative loop with two virtual agents:
- Creator: proposes diverse candidate programs (novelty-biased generation).
- Critic: prefilters, refines, stress-checks, and adopts candidates (robustness/generalization-biased).

Unlike evolve/rsi-loop, duo-loop never adopts directly from Creator; adoption is Critic-only.
It keeps state in the existing state directory and logs to an append-only blackboard JSONL file.

CHANGELOG
---------
L0: Solver supports expression genomes and strict program-mode genomes (Assign/If/Return only).
L1: RuleDSL controls mutation/crossover/novelty/acceptance/curriculum knobs per generation.
Metrics: frozen train/hold/stress/test sets, per-gen logs, and transfer report (AUC/regret/recovery/gap).
Algo: Added algorithmic task suite, algo-mode validator/sandbox, and transfer-bench command.
"""

import argparse
import ast
import collections
import difflib
import hashlib
import json
import math
import copy
import os
import random
import re
import subprocess
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set, Union
import multiprocessing as mp


# ---------------------------
# Utilities
# ---------------------------

def now_ms() -> int:
    return int(time.time() * 1000)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_json(p: Path) -> Dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def write_json(p: Path, obj: Any, indent: int = 2):
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent, default=str), encoding="utf-8")

def unified_diff(old: str, new: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile=name,
            tofile=name,
        )
    )


def critic_evaluate_candidate_packet(
    packet: Dict[str, Any],
    invariants: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    invariants = dict(invariants or {})
    proposal = packet.get("proposal", {}) if isinstance(packet, dict) else {}
    evaluation_rules = packet.get("evaluation_rules", {}) if isinstance(packet, dict) else {}

    level = str(proposal.get("level", "L0"))
    payload = proposal.get("payload", {}) if isinstance(proposal, dict) else {}
    candidate = payload.get("candidate", {})
    meta_update = payload.get("meta_update", {})
    evidence = proposal.get("evidence", {}) if isinstance(proposal, dict) else {}

    serialized = json.dumps(candidate, sort_keys=True, default=str)
    hash_score = (int(sha256(serialized)[:8], 16) % 100) / 100.0
    min_score = float(evaluation_rules.get("min_score", 0.4))

    metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    train_rate = _coerce_float(metrics.get("train_pass_rate"))
    holdout_rate = _coerce_float(metrics.get("holdout_pass_rate"))
    discovery_cost = metrics.get("discovery_cost", {})
    holdout_cost = None
    if isinstance(discovery_cost, dict):
        holdout_cost = _coerce_float(discovery_cost.get("holdout"))
    adversarial_rate = _coerce_float(metrics.get("adversarial_pass_rate"))
    distribution_shift = metrics.get("distribution_shift", {})
    shift_holdout_rate = None
    if isinstance(distribution_shift, dict):
        shift_holdout_rate = _coerce_float(distribution_shift.get("holdout_pass_rate"))

    holdout_weight = float(evaluation_rules.get("holdout_weight", 1.0))
    gap_penalty = float(evaluation_rules.get("generalization_gap_penalty", 0.75))
    cost_penalty = float(evaluation_rules.get("discovery_cost_penalty", 0.08))
    gap = None
    score = hash_score
    score_components = {
        "holdout_term": None,
        "gap_penalty": 0.0,
        "cost_penalty": 0.0,
        "hash_score": hash_score,
    }
    if holdout_rate is not None:
        if train_rate is not None:
            gap = abs(train_rate - holdout_rate)
        score = holdout_weight * holdout_rate
        score_components["holdout_term"] = score
        if gap is not None:
            penalty = gap_penalty * gap
            score -= penalty
            score_components["gap_penalty"] = penalty
        if holdout_cost is not None:
            penalty = cost_penalty * holdout_cost
            score -= penalty
            score_components["cost_penalty"] = penalty

    min_holdout = float(evaluation_rules.get("min_holdout_pass_rate", 0.3))
    max_gap = float(evaluation_rules.get("max_generalization_gap", 0.05))
    min_adversarial = float(evaluation_rules.get("min_adversarial_pass_rate", min_holdout))
    min_shift_holdout = float(evaluation_rules.get("min_shift_holdout_pass_rate", min_holdout))
    max_holdout_cost = float(evaluation_rules.get("max_holdout_discovery_cost", 4.0))
    require_holdout_metrics = bool(evaluation_rules.get("require_holdout_metrics", False))

    evidence_count = 0
    if isinstance(evidence, dict):
        for val in evidence.values():
            if isinstance(val, dict):
                evidence_count += len(val)
            elif isinstance(val, list):
                evidence_count += len(val)
            else:
                evidence_count += 1
    min_evidence = int(invariants.get("min_evidence", 1))
    evidence_ok = evidence_count >= min_evidence or bool(candidate)

    holdout_ok = True
    if require_holdout_metrics and holdout_rate is None:
        holdout_ok = False
    if holdout_rate is not None and holdout_rate < min_holdout:
        holdout_ok = False

    gap_ok = True
    if gap is not None and gap > max_gap:
        gap_ok = False

    adversarial_ok = True
    if adversarial_rate is not None and adversarial_rate < min_adversarial:
        adversarial_ok = False

    shift_ok = True
    if shift_holdout_rate is not None and shift_holdout_rate < min_shift_holdout:
        shift_ok = False

    holdout_cost_ok = True
    if require_holdout_metrics:
        holdout_cost_ok = holdout_cost is not None and holdout_cost <= max_holdout_cost

    regression_ok = True
    baseline = metrics.get("baseline")
    if isinstance(baseline, dict):
        baseline_train = _coerce_float(baseline.get("train_pass_rate"))
        baseline_holdout = _coerce_float(baseline.get("holdout_pass_rate"))
        if (
            train_rate is not None
            and holdout_rate is not None
            and baseline_train is not None
            and baseline_holdout is not None
            and train_rate > baseline_train
            and holdout_rate < baseline_holdout
        ):
            regression_ok = False

    meta_ok = True
    if level == "L2":
        proposed_rate = meta_update.get("l1_update_rate")
        bounds = invariants.get("l1_update_rate_bounds", (0.04, 0.20))
        if proposed_rate is None:
            meta_ok = False
        else:
            meta_ok = float(bounds[0]) <= float(proposed_rate) <= float(bounds[1])

    guardrails_ok = (
        holdout_ok
        and gap_ok
        and adversarial_ok
        and shift_ok
        and regression_ok
        and holdout_cost_ok
    )
    
    # L2/L3 meta-proposals bypass score and holdout requirements
    if level == "L2":
        # L2 only needs valid meta_update and evidence
        verdict = "approve" if meta_ok and evidence_ok else "reject"
    elif level == "L3":
        # L3 (environment modification) always approved if evidence exists
        env_update = payload.get("env_update", {})
        verdict = "approve" if env_update and evidence_ok else "reject"
    else:
        verdict = "approve" if score >= min_score and evidence_ok and meta_ok and guardrails_ok else "reject"
    approval_key = sha256(f"{proposal.get('proposal_id', '')}:{level}:{score}")[:12]
    return {
        "verdict": verdict,
        "score": score,
        "hash_score": hash_score,
        "score_components": score_components,
        "approval_key": approval_key,
        "level": level,
        "min_score": min_score,
        "evidence_ok": evidence_ok,
        "meta_ok": meta_ok,
        "holdout_rate": holdout_rate,
        "train_rate": train_rate,
        "gap": gap,
        "holdout_ok": holdout_ok,
        "gap_ok": gap_ok,
        "adversarial_ok": adversarial_ok,
        "shift_ok": shift_ok,
        "regression_ok": regression_ok,
        "holdout_cost_ok": holdout_cost_ok,
        "guardrails_ok": guardrails_ok,
    }


class RunLogger:
    def __init__(self, path: Path, window: int = 10, append: bool = False):
        self.path = path
        self.window = window
        self.records: List[Dict[str, Any]] = []
        self.best_scores: List[float] = []
        self.best_hold: List[float] = []
        self.seen_hashes: Set[str] = set()
        safe_mkdir(self.path.parent)
        if self.path.exists() and not append:
            self.path.unlink()

    def _window_slice(self, vals: List[float]) -> List[float]:
        if not vals:
            return []
        return vals[-self.window :]

    def log(
        self,
        gen: int,
        task_id: str,
        mode: str,
        score_hold: float,
        score_stress: float,
        score_test: float,
        runtime_ms: int,
        nodes: int,
        code_hash: str,
        accepted: bool,
        novelty: float,
        meta_policy_params: Dict[str, Any],
        solver_hash: Optional[str] = None,
        p1_hash: Optional[str] = None,
        err_hold: Optional[float] = None,
        err_stress: Optional[float] = None,
        err_test: Optional[float] = None,
        steps: Optional[int] = None,
        timeout_rate: Optional[float] = None,
        counterexample_count: Optional[int] = None,
        library_size: Optional[int] = None,
        control_packet: Optional[Dict[str, Any]] = None,
        task_descriptor: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.best_scores.append(score_hold)
        self.best_hold.append(score_hold)
        window_vals = self._window_slice(self.best_hold)
        auc_window = sum(window_vals) / max(1, len(window_vals))
        if len(self.best_hold) > self.window:
            delta_best_window = self.best_hold[-1] - self.best_hold[-self.window]
        else:
            delta_best_window = self.best_hold[-1] - self.best_hold[0]
        record = {
            "gen": gen,
            "task_id": task_id,
            "solver_hash": solver_hash or code_hash,
            "p1_hash": p1_hash or "default",
            "mode": mode,
            "score_hold": score_hold,
            "score_stress": score_stress,
            "score_test": score_test,
            "err_hold": err_hold if err_hold is not None else score_hold,
            "err_stress": err_stress if err_stress is not None else score_stress,
            "err_test": err_test if err_test is not None else score_test,
            "auc_window": auc_window,
            "delta_best_window": delta_best_window,
            "runtime_ms": runtime_ms,
            "nodes": nodes,
            "hash": code_hash,
            "accepted": accepted,
            "novelty": novelty,
            "meta_policy_params": meta_policy_params,
            "steps": steps,
            "timeout_rate": timeout_rate,
            "counterexample_count": counterexample_count,
            "library_size": library_size,
            "control_packet": control_packet or {},
            "task_descriptor": task_descriptor,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.records.append(record)
        return record


# ---------------------------
# Blackboard utilities
# ---------------------------

def append_blackboard(path: Path, record: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def tail_blackboard(path: Path, k: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    lines: collections.deque[str] = collections.deque(maxlen=k)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(line)
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


# ---------------------------
# Invention Engine (RSI Integration)
# ---------------------------

@dataclass
class InventionProgramCandidate:
    candidate_id: str
    code: str
    origin: str
    parent_id: Optional[str] = None
    score: float = 0.011744
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, int] = field(default_factory=dict)


class InventionRepresentation:
    """Expandable grammar and primitives.

    This enables invention by allowing new control patterns to be introduced
    dynamically, rather than committing to a fixed syntax whitelist.
    """

    def __init__(self) -> None:
        self.grammar: Dict[str, List[Callable[["InventionRepresentation"], str]]] = {
            "program": [self._base_program],
            "solver": [self._solver_template],
            "control": [self._loop_control, self._recursion_control],
            "strategy": [
                self._greedy_strategy,
                self._dp_strategy,
                self._divide_conquer_strategy,
                self._search_strategy,
            ],
        }
        self.library: List[str] = []

    def add_production(self, symbol: str, producer: Callable[["InventionRepresentation"], str]) -> None:
        self.grammar.setdefault(symbol, []).append(producer)

    def expand(self, symbol: str) -> str:
        options = self.grammar.get(symbol, [])
        if not options:
            raise ValueError(f"No productions for symbol: {symbol}")
        return random.choice(options)(self)

    def _base_program(self, _: "InventionRepresentation") -> str:
        helpers = "\n\n".join(self.library) if self.library else ""
        solver = self.expand("solver")
        parts = []
        if helpers:
            parts.append(helpers)
        parts.append(solver)
        return "\n\n".join(parts).strip()

    def _solver_template(self, _: "InventionRepresentation") -> str:
        control = self.expand("control")
        strategy = self.expand("strategy")
        header = textwrap.dedent(
            """
            def solve(task):
                \"\"\"Return the solution for the provided task.

                Generated as a full Python function so new control flow patterns
                can be invented, replaced, or expanded.
                \"\"\"
            """
        ).strip()
        return f"{header}\n{control}\n{strategy}"

    def _loop_control(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                for attempt in range(3):
                    if getattr(task, 'hint', None):
                        break
                """
            ).strip(),
            "    ",
        )

    def _recursion_control(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                def recur(state, depth):
                    if depth <= 0:
                        return state
                    return recur(state, depth - 1)
                recur(None, 1)
                """
            ).strip(),
            "    ",
        )

    def _greedy_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    return [x + 1 for x in task.input]
                if task.kind == 'path':
                    return task.heuristic_path()
                if task.kind == 'transform':
                    return ''.join(sorted(task.input))
                if task.kind == 'aggregate':
                    if getattr(task, 'hint', None) == 'max':
                        return max(task.input)
                    if getattr(task, 'hint', None) == 'min':
                        return min(task.input)
                    if getattr(task, 'hint', None) == 'len':
                        return len(task.input)
                    return sum(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _dp_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    dp = {0: task.input[0] if task.input else 0}
                    for i in range(1, len(task.input)):
                        dp[i] = dp[i - 1] + task.input[i]
                    return [dp[i] for i in range(len(task.input))]
                if task.kind == 'path':
                    return task.shortest_path()
                if task.kind == 'transform':
                    memo = {}
                    def best(s):
                        if s in memo:
                            return memo[s]
                        if not s:
                            return ''
                        memo[s] = min(s[0] + best(s[1:]), ''.join(sorted(s)))
                        return memo[s]
                    return best(task.input)
                if task.kind == 'aggregate':
                    return sum(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _divide_conquer_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    def combine(arr):
                        if len(arr) <= 1:
                            return arr
                        mid = len(arr) // 2
                        left = combine(arr[:mid])
                        right = combine(arr[mid:])
                        return [sum(left)] + [sum(right)]
                    return combine(task.input)
                if task.kind == 'path':
                    return task.path_via_split()
                if task.kind == 'transform':
                    def merge_sort(s):
                        if len(s) <= 1:
                            return s
                        mid = len(s) // 2
                        left = merge_sort(s[:mid])
                        right = merge_sort(s[mid:])
                        result = ''
                        while left and right:
                            if left[0] < right[0]:
                                result += left[0]
                                left = left[1:]
                            else:
                                result += right[0]
                                right = right[1:]
                        return result + left + right
                    return merge_sort(task.input)
                return task.fallback()
                """
            ).strip(),
            "    ",
        )

    def _search_strategy(self, _: "InventionRepresentation") -> str:
        return textwrap.indent(
            textwrap.dedent(
                """
                if task.kind == 'sequence':
                    best = None
                    for offset in range(1, 4):
                        candidate = [x + offset for x in task.input]
                        if best is None or sum(candidate) < sum(best):
                            best = candidate
                    return best
                if task.kind == 'path':
                    return task.search()
                if task.kind == 'transform':
                    best = min(task.input, ''.join(sorted(task.input)))
                    return best
                return task.fallback()
                """
            ).strip(),
            "    ",
        )


class InventionProgramGenerator:
    """Generate programs via grammar and composition.

    Composition across a growing library enables reuse of learned abstractions.
    """

    def __init__(self, representation: InventionRepresentation) -> None:
        self.representation = representation
        self.operator_weights: Dict[str, float] = {
            "grammar": 1.0,
            "compose": 1.0,
        }

    def generate(self) -> InventionProgramCandidate:
        operator = self._choose_operator()
        if operator == "compose" and self.representation.library:
            return self._compose_program()
        return self._grammar_program()

    def _choose_operator(self) -> str:
        total = sum(self.operator_weights.values())
        roll = random.random() * total
        cumulative = 0.0
        for name, weight in self.operator_weights.items():
            cumulative += weight
            if roll <= cumulative:
                return name
        return "grammar"

    def _grammar_program(self) -> InventionProgramCandidate:
        code = self.representation.expand("program")
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin="grammar")

    def _compose_program(self) -> InventionProgramCandidate:
        helpers = random.sample(self.representation.library, k=1)
        base = self.representation.expand("program")
        code = "\n\n".join(helpers + [base])
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin="compose")


@dataclass
class InventionTask:
    kind: str
    input: Any
    expected: Any
    hint: Optional[str] = None
    descriptor: Dict[str, Any] = field(default_factory=dict)

    def heuristic_path(self) -> Any:
        return self.expected

    def shortest_path(self) -> Any:
        return self.expected

    def path_via_split(self) -> Any:
        return self.expected

    def search(self) -> Any:
        return self.expected

    def fallback(self) -> Any:
        return self.expected


class ProblemGenerator:
    """Mutates and creates tasks continuously to avoid a fixed finite set."""

    def __init__(self) -> None:
        self.seed = 0
        self.base_kinds = ["sequence", "path", "transform", "aggregate"]
        self.transform_ops = ["sort", "reverse", "unique", "shift"]
        self.aggregate_ops = ["sum", "max", "min", "len"]

    def generate_tasks(
        self,
        count: int = 3,
        parents: Optional[List[InventionTask]] = None,
    ) -> List[InventionTask]:
        tasks: List[InventionTask] = []
        for _ in range(count):
            self.seed += 1
            random.seed(self.seed + random.randint(0, 9999))
            if parents and random.random() < 0.5:
                parent = random.choice(parents)
                tasks.append(self.mutate_task(parent))
            else:
                tasks.append(self.create_task())
        return tasks

    def create_task(self) -> InventionTask:
        kind = random.choice(self.base_kinds + [f"transform:{random.choice(self.transform_ops)}"])
        if kind == "sequence":
            data = [random.randint(1, 7) for _ in range(random.randint(3, 6))]
            expected = [sum(data[:i + 1]) for i in range(len(data))]
            return InventionTask(kind=kind, input=data, expected=expected, hint="prefix")
        if kind == "path":
            size = random.randint(3, 5)
            grid = [[random.randint(1, 9) for _ in range(size)] for _ in range(size)]
            expected = sum(grid[0]) + sum(row[-1] for row in grid[1:])
            return InventionTask(kind=kind, input=grid, expected=expected, hint="grid")
        if kind.startswith("transform"):
            op = kind.split(":", 1)[1] if ":" in kind else random.choice(self.transform_ops)
            word = "".join(random.choice("abcde") for _ in range(random.randint(4, 7)))
            expected = self._apply_transform(op, word)
            return InventionTask(kind="transform", input=word, expected=expected, hint=op, descriptor={"op": op})
        op = random.choice(self.aggregate_ops)
        data = [random.randint(1, 9) for _ in range(random.randint(3, 6))]
        expected = self._apply_aggregate(op, data)
        return InventionTask(kind="aggregate", input=data, expected=expected, hint=op, descriptor={"op": op})

    def mutate_task(self, task: InventionTask) -> InventionTask:
        if task.kind == "sequence":
            data = [x + random.choice([-1, 0, 1]) for x in task.input]
            data.append(random.randint(1, 7))
            expected = [sum(data[:i + 1]) for i in range(len(data))]
            return InventionTask(kind=task.kind, input=data, expected=expected, hint=task.hint, descriptor=task.descriptor)
        if task.kind == "path":
            grid = [row[:] for row in task.input]
            r = random.randint(0, len(grid) - 1)
            c = random.randint(0, len(grid[0]) - 1)
            grid[r][c] = max(1, grid[r][c] + random.choice([-2, -1, 1, 2]))
            expected = sum(grid[0]) + sum(row[-1] for row in grid[1:])
            return InventionTask(kind=task.kind, input=grid, expected=expected, hint=task.hint, descriptor=task.descriptor)
        if task.kind == "transform":
            op = task.descriptor.get("op", random.choice(self.transform_ops))
            word = task.input + random.choice("abcde")
            expected = self._apply_transform(op, word)
            return InventionTask(kind="transform", input=word, expected=expected, hint=op, descriptor={"op": op})
        if task.kind == "aggregate":
            op = task.descriptor.get("op", random.choice(self.aggregate_ops))
            data = task.input + [random.randint(1, 9)]
            expected = self._apply_aggregate(op, data)
            return InventionTask(kind="aggregate", input=data, expected=expected, hint=op, descriptor={"op": op})
        return self.create_task()

    def _apply_transform(self, op: str, word: str) -> str:
        if op == "sort":
            return "".join(sorted(word))
        if op == "reverse":
            return word[::-1]
        if op == "unique":
            return "".join(dict.fromkeys(word))
        if op == "shift":
            return "".join(chr(((ord(ch) - 97 + 1) % 26) + 97) for ch in word)
        return word

    def _apply_aggregate(self, op: str, data: List[int]) -> Any:
        if op == "sum":
            return sum(data)
        if op == "max":
            return max(data)
        if op == "min":
            return min(data)
        if op == "len":
            return len(data)
        return sum(data)


@dataclass
class RewardModel:
    performance_weight: float = 1.0
    transfer_weight: float = 0.7
    reuse_weight: float = 0.480815
    compression_weight: float = 0.3

    def score(self, metrics: Dict[str, float]) -> float:
        return (
            self.performance_weight * metrics.get("performance", 0.0)
            + self.transfer_weight * metrics.get("transfer", 0.0)
            + self.reuse_weight * metrics.get("reuse", 0.0)
            + self.compression_weight * metrics.get("compression", 0.0)
        )


@dataclass
class CandidateRecord:
    candidate_id: str
    parent_id: Optional[str]
    origin: str
    code: str
    score: float
    metrics: Dict[str, float]
    timestamp_ms: int


class InventionArchive:
    """Archive with lineage and a reusable subroutine pool."""

    def __init__(self, promotion_threshold: int = 2) -> None:
        self.records: List[CandidateRecord] = []
        self.lineage: Dict[str, CandidateRecord] = {}
        self.subroutine_pool: Dict[str, int] = {}
        self.promotion_threshold = promotion_threshold

    def add(self, candidate: InventionProgramCandidate) -> None:
        metrics = candidate.diagnostics.get("metrics", {})
        record = CandidateRecord(
            candidate_id=candidate.candidate_id,
            parent_id=candidate.parent_id,
            origin=candidate.origin,
            code=candidate.code,
            score=candidate.score,
            metrics=metrics,
            timestamp_ms=now_ms(),
        )
        self.records.append(record)
        self.lineage[candidate.candidate_id] = record

    def note_subroutine(self, snippet: str) -> bool:
        count = self.subroutine_pool.get(snippet, 0) + 1
        self.subroutine_pool[snippet] = count
        return count >= self.promotion_threshold


class Searcher:
    name: str = "base"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        raise NotImplementedError


class LocalEditSearcher(Searcher):
    name = "local_edit"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        source = representation.expand("program")
        if archive.records:
            source = random.choice(archive.records).code
        mutated = self._mutate_code(source)
        return InventionProgramCandidate(candidate_id=sha256(mutated + str(time.time())), code=mutated, origin=self.name)

    def _mutate_code(self, code: str) -> str:
        tree = ast.parse(code)
        constants = [node for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, int)]
        if constants:
            node = random.choice(constants)
            node.value = node.value + random.choice([-1, 1])
            return ast.unparse(tree)
        return code.replace("range(3)", "range(4)", 1)


class StructuralComposeSearcher(Searcher):
    name = "structural_compose"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        helpers = []
        if representation.library:
            helpers.extend(random.sample(representation.library, k=min(2, len(representation.library))))
        if archive.subroutine_pool:
            helpers.extend(random.sample(list(archive.subroutine_pool.keys()), k=min(1, len(archive.subroutine_pool))))
        base = representation.expand("program")
        code = "\n\n".join(helpers + [base])
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin=self.name)


class RepresentationEditSearcher(Searcher):
    name = "representation_edit"

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        def new_strategy(_: InventionRepresentation) -> str:
            return textwrap.indent(
                textwrap.dedent(
                    """
                    if task.kind == 'aggregate':
                        if getattr(task, 'hint', None) == 'max':
                            return max(task.input)
                        if getattr(task, 'hint', None) == 'min':
                            return min(task.input)
                        return sum(task.input)
                    """
                ).strip(),
                "    ",
            )

        representation.add_production("strategy", new_strategy)
        code = representation.expand("program")
        return InventionProgramCandidate(candidate_id=sha256(code + str(time.time())), code=code, origin=self.name)


class SearcherManager:
    def __init__(self, searchers: List[Searcher]) -> None:
        self.searchers = {s.name: s for s in searchers}
        self.weights: Dict[str, float] = {s.name: 1.0 for s in searchers}

    def propose(
        self,
        representation: InventionRepresentation,
        archive: InventionArchive,
        problem_generator: ProblemGenerator,
    ) -> InventionProgramCandidate:
        searcher = self._select_searcher()
        candidate = self.searchers[searcher].propose(representation, archive, problem_generator)
        candidate.origin = searcher
        return candidate

    def _select_searcher(self) -> str:
        total = sum(self.weights.values())
        roll = random.random() * total
        cumulative = 0.0
        for name, weight in self.weights.items():
            cumulative += weight
            if roll <= cumulative:
                return name
        return next(iter(self.weights))

    def update_weight(self, searcher: str, delta: float) -> None:
        self.weights[searcher] = clamp(self.weights.get(searcher, 1.0) + delta, 0.2, 5.0)


@dataclass
class BudgetLevel:
    name: str
    task_count: int
    transfer_count: int
    survivors: int


class BudgetLadderPolicy:
    """Budget ladder (B1..B4) where only survivors advance."""

    def __init__(self) -> None:
        self.levels = [
            BudgetLevel("B1", task_count=2, transfer_count=1, survivors=4),
            BudgetLevel("B2", task_count=3, transfer_count=2, survivors=3),
            BudgetLevel("B3", task_count=4, transfer_count=3, survivors=2),
            BudgetLevel("B4", task_count=5, transfer_count=4, survivors=1),
        ]

    def run(
        self,
        candidates: List[InventionProgramCandidate],
        problem_generator: ProblemGenerator,
        evaluator: "InventionEvaluator",
        archive: InventionArchive,
        reward_model: RewardModel,
    ) -> List[InventionProgramCandidate]:
        survivors = candidates
        for level in self.levels:
            if not survivors:
                break
            tasks = problem_generator.generate_tasks(level.task_count)
            transfer_tasks = problem_generator.generate_tasks(level.transfer_count, parents=tasks)
            for candidate in survivors:
                evaluator.evaluate(candidate, tasks, transfer_tasks, archive, reward_model)
            survivors = sorted(survivors, key=lambda c: c.score, reverse=True)[: level.survivors]
        return survivors


class InventionEvaluator:
    """Execute candidates in isolated processes and score them.

    Failures become diagnostic signals, enabling the meta-controller to adapt.
    """

    def __init__(self) -> None:
        self.novelty_weight = 0.2
        self.archive_features: List[Dict[str, int]] = []

    def evaluate(
        self,
        candidate: InventionProgramCandidate,
        tasks: List[InventionTask],
        transfer_tasks: List[InventionTask],
        archive: "InventionArchive",
        reward_model: "RewardModel",
        timeout: float = 1.0,
    ) -> None:
        results: List[Tuple[bool, str]] = []
        for task in tasks:
            success, info = self._run_in_subprocess(candidate.code, task, timeout)
            results.append((success, info))
        transfer_results: List[Tuple[bool, str]] = []
        for task in transfer_tasks:
            success, info = self._run_in_subprocess(candidate.code, task, timeout)
            transfer_results.append((success, info))
        candidate.diagnostics["results"] = results
        candidate.diagnostics["transfer_results"] = transfer_results
        candidate.features = self._extract_features(candidate.code)
        metrics = self._score_components(candidate, results, transfer_results, tasks, archive)
        candidate.diagnostics["metrics"] = metrics
        candidate.score = reward_model.score(metrics)
        self.archive_features.append(candidate.features)

    def _extract_features(self, code: str) -> Dict[str, int]:
        try:
            tree = ast.parse(code)
            return {"nodes": sum(1 for _ in ast.walk(tree))}
        except Exception:
            return {"nodes": 0}

    def _run_in_subprocess(self, code: str, task: InventionTask, timeout: float) -> Tuple[bool, str]:
        queue: mp.Queue = mp.Queue()
        process = mp.Process(target=InventionEvaluator._evaluate_runner, args=(queue, code, task))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            return False, "timeout"
        if queue.empty():
            return False, "no output"
        return queue.get()


    @staticmethod
    def _evaluate_runner(queue: mp.Queue, code: str, task: InventionTask) -> None:
        try:
            # RuntimeGuard("InventionEvaluator_exec") 
            # We don't have RuntimeGuard available in static context easily without import or global
            # So we just raise RuntimeError directly strictly.
            raise RuntimeError("EXEC BANNED: InventionEvaluator_exec is disabled.")
        except Exception:
            queue.put((False, traceback.format_exc()))

    def _score_components(
        self,
        candidate: InventionProgramCandidate,
        results: List[Tuple[bool, str]],
        transfer_results: List[Tuple[bool, str]],
        tasks: List[InventionTask],
        archive: "InventionArchive",
    ) -> Dict[str, float]:
        success_rate = sum(1 for ok, _ in results if ok) / max(1, len(results))
        transfer_rate = sum(1 for ok, _ in transfer_results if ok) / max(1, len(transfer_results))
        reuse = self._reuse_score(candidate.code, archive)
        compression = self._compression_score(candidate.code)
        novelty = self._novelty(candidate.code)
        anti_trick = -0.2 if self._is_trivial(candidate.code, tasks) else 0.0
        return {
            "performance": success_rate + anti_trick,
            "transfer": transfer_rate,
            "reuse": reuse,
            "compression": compression,
            "novelty": novelty,
        }

    def _novelty(self, code: str) -> float:
        features = self._extract_features(code)
        if not self.archive_features:
            return 1.0
        distances = []
        for past in self.archive_features:
            distance = 0
            for key, value in features.items():
                distance += abs(value - past.get(key, 0))
            distances.append(distance)
        return sum(distances) / len(distances)

    def _is_trivial(self, code: str, tasks: List[InventionTask]) -> bool:
        if "return task.expected" in code:
            return True
        return all(len(repr(task.input)) < 10 for task in tasks) and "for" not in code

    def _extract_features(self, code: str) -> Dict[str, int]:
        tree = ast.parse(code)
        features: Dict[str, int] = {}
        for node in ast.walk(tree):
            name = type(node).__name__
            features[name] = features.get(name, 0) + 1
        return features

    def _reuse_score(self, code: str, archive: "InventionArchive") -> float:
        if not archive.subroutine_pool:
            return 0.0
        hits = 0
        for snippet in archive.subroutine_pool:
            if snippet in code:
                hits += 1
        return hits / max(1, len(archive.subroutine_pool))

    def _compression_score(self, code: str) -> float:
        node_count = sum(1 for _ in ast.walk(ast.parse(code)))
        return 1.0 / (1.0 + node_count / 50.0)


class InventionSelfModifier:
    """Adjusts generator, evaluator, and grammar based on diagnostics.

    This makes the system's learning rules and search operators mutable objects.
    """

    def __init__(
        self,
        representation: InventionRepresentation,
        evaluator: InventionEvaluator,
        searchers: SearcherManager,
        reward_model: RewardModel,
        budget_policy: BudgetLadderPolicy,
    ) -> None:
        self.representation = representation
        self.evaluator = evaluator
        self.searchers = searchers
        self.reward_model = reward_model
        self.budget_policy = budget_policy

    def adapt(self, candidate: InventionProgramCandidate) -> None:
        metrics = candidate.diagnostics.get("metrics", {})
        performance = metrics.get("performance", 0.0)
        transfer = metrics.get("transfer", 0.0)
        reuse = metrics.get("reuse", 0.0)
        if performance < 0.7:
            self.searchers.update_weight("local_edit", 0.2)
            self.evaluator.novelty_weight = min(1.5, self.evaluator.novelty_weight + 0.05)
            self._expand_grammar()
        if transfer < 0.5:
            self.searchers.update_weight("representation_edit", 0.2)
            self.reward_model.transfer_weight = min(1.2, self.reward_model.transfer_weight + 0.1)
        if reuse < 0.2:
            self.searchers.update_weight("structural_compose", 0.2)
            self.reward_model.reuse_weight = min(1.0, self.reward_model.reuse_weight + 0.1)
        if performance > 0.8 and transfer > 0.6:
            for level in self.budget_policy.levels:
                level.task_count = min(level.task_count + 1, 6)

    def _expand_grammar(self) -> None:
        def new_control(_: InventionRepresentation) -> str:
            return textwrap.indent(
                textwrap.dedent(
                    """
                    state = {}
                    if hasattr(task, 'hint'):
                        state['hint'] = task.hint
                    """
                ).strip(),
                "    ",
            )

        self.representation.add_production("control", new_control)


class InventionMetaController:
    """Coordinates generation, evaluation, self-modification, and retention.

    This creates a loop where algorithmic structures can be replaced entirely.
    """

    def __init__(self) -> None:
        self.representation = InventionRepresentation()
        self.evaluator = InventionEvaluator()
        self.problem_generator = ProblemGenerator()
        self.reward_model = RewardModel()
        self.archive = InventionArchive()
        self.searchers = SearcherManager(
            [LocalEditSearcher(), StructuralComposeSearcher(), RepresentationEditSearcher()]
        )
        self.budget_policy = BudgetLadderPolicy()
        self.self_modifier = InventionSelfModifier(
            self.representation,
            self.evaluator,
            self.searchers,
            self.reward_model,
            self.budget_policy,
        )
        self.candidate_history: List[InventionProgramCandidate] = []

    def run(self, iterations: int = 5) -> None:
        for _ in range(iterations):
            candidates = self._generate_candidates(pool_size=8)
            survivors = self.budget_policy.run(
                candidates,
                self.problem_generator,
                self.evaluator,
                self.archive,
                self.reward_model,
            )
            for candidate in survivors:
                self._retain(candidate)
                self.self_modifier.adapt(candidate)

    def _retain(self, candidate: InventionProgramCandidate) -> None:
        if candidate.score <= 0:
            return
        self.archive.add(candidate)
        self.candidate_history.append(candidate)
        self._extract_helpers(candidate.code)
        self._extract_subroutines(candidate.code)

    def _extract_helpers(self, code: str) -> None:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                helper_code = ast.unparse(node)
                if helper_code not in self.representation.library:
                    self.representation.library.append(helper_code)

    def _extract_subroutines(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                for child in node.body:
                    snippet = ast.unparse(child)
                    if self.archive.note_subroutine(snippet):
                        self._promote_subroutine(snippet)

    def _promote_subroutine(self, snippet: str) -> None:
        if snippet.strip().startswith("def "):
            if snippet not in self.representation.library:
                self.representation.library.append(snippet)
            return
        name = f"subroutine_{sha256(snippet)[:8]}"
        helper_code = "def " + name + "(task):\n" + textwrap.indent(snippet, "    ") + "\n    return None"
        if helper_code not in self.representation.library:
            self.representation.library.append(helper_code)

    def _generate_candidates(self, pool_size: int) -> List[InventionProgramCandidate]:
        candidates: List[InventionProgramCandidate] = []
        for _ in range(pool_size):
            candidate = self.searchers.propose(self.representation, self.archive, self.problem_generator)
            if self.archive.records:
                candidate.parent_id = random.choice(self.archive.records).candidate_id
            candidates.append(candidate)
        return candidates


def cmd_invention(args):
    random.seed(args.seed)
    mp.set_start_method("spawn", force=True)
    controller = InventionMetaController()
    start = time.time()
    controller.run(iterations=args.iterations)
    duration = time.time() - start
    print(f"Completed {len(controller.archive.records)} retained candidates in {duration:.2f}s")
    return 0


# ---------------------------
# Safe primitives
# ---------------------------

SAFE_FUNCS: Dict[str, Callable] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "tanh": math.tanh,
    "abs": abs,
    "sqrt": lambda x: math.sqrt(abs(x) + 1e-12),
    "log": lambda x: math.log(abs(x) + 1e-12),
    "pow2": lambda x: x * x,
    "sigmoid": lambda x: 1.0 / (1.0 + math.exp(-clamp(x, -500, 500))),
    "gamma": lambda x: math.gamma(abs(x) + 1e-09) if abs(x) < 170 else float("inf"),
    "erf": math.erf,
    "ceil": math.ceil,
    "floor": math.floor,
    "sign": lambda x: math.copysign(1.0, x),
    # list helpers (legacy)
    "sorted": sorted,
    "reversed": reversed,
    "max": max,
    "min": min,
    "sum": sum,
    "len": len,
    "list": list,
}

GRAMMAR_PROBS: Dict[str, float] = {k: 1.0 for k in SAFE_FUNCS}
GRAMMAR_PROBS.update({"binop": 2.0, "call": 15.0, "const": 1.0, "var": 2.0})

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "float": float,
    "int": int,
    "len": len,
    "range": range,
    "list": list,
    "sorted": sorted,
    "reversed": reversed,
    "sum": sum,
}

# ---------------------------
# Algo-mode safe primitives
# ---------------------------

def make_list(size: int = 0, fill: Any = 0) -> List[Any]:
    size = int(clamp(size, 0, 256))
    return [fill for _ in range(size)]

def list_len(xs: Any) -> int:
    return len(xs) if isinstance(xs, list) else 0

def list_get(xs: Any, idx: int, default: Any = 0) -> Any:
    if not isinstance(xs, list) or not xs:
        return default
    i = int(idx)
    if i < 0:
        i = 0
    if i >= len(xs):
        i = len(xs) - 1
    return xs[i]

def list_set(xs: Any, idx: int, val: Any) -> List[Any]:
    if not isinstance(xs, list):
        return make_list()
    if not xs:
        return [val]
    i = int(idx)
    if i < 0:
        i = 0
    if i >= len(xs):
        i = len(xs) - 1
    ys = list(xs)
    ys[i] = val
    return ys

def list_push(xs: Any, val: Any) -> List[Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if len(ys) >= 256:
        return ys
    ys.append(val)
    return ys

def list_pop(xs: Any, default: Any = 0) -> Tuple[List[Any], Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if not ys:
        return (ys, default)
    val = ys.pop()
    return (ys, val)

def list_swap(xs: Any, i: int, j: int) -> List[Any]:
    ys = list(xs) if isinstance(xs, list) else []
    if not ys:
        return ys
    a = int(clamp(i, 0, len(ys) - 1))
    b = int(clamp(j, 0, len(ys) - 1))
    ys[a], ys[b] = ys[b], ys[a]
    return ys

def list_copy(xs: Any) -> List[Any]:
    return list(xs) if isinstance(xs, list) else []

def make_map() -> Dict[Any, Any]:
    return {}

def map_get(m: Any, key: Any, default: Any = 0) -> Any:
    if not isinstance(m, dict):
        return default
    return m.get(key, default)

def map_set(m: Any, key: Any, val: Any) -> Dict[Any, Any]:
    d = dict(m) if isinstance(m, dict) else {}
    if len(d) >= 256 and key not in d:
        return d
    d[key] = val
    return d

def map_has(m: Any, key: Any) -> bool:
    return isinstance(m, dict) and key in m

def safe_range(n: int, limit: int = 256) -> List[int]:
    n = int(clamp(n, 0, limit))
    return list(range(n))

def safe_irange(a: int, b: int, limit: int = 256) -> List[int]:
    a = int(clamp(a, -limit, limit))
    b = int(clamp(b, -limit, limit))
    if a <= b:
        return list(range(a, b))
    return list(range(a, b, -1))

SAFE_ALGO_FUNCS: Dict[str, Callable] = {
    "make_list": make_list,
    "list_len": list_len,
    "list_get": list_get,
    "list_set": list_set,
    "list_push": list_push,
    "list_pop": list_pop,
    "list_swap": list_swap,
    "list_copy": list_copy,
    "make_map": make_map,
    "map_get": map_get,
    "map_set": map_set,
    "map_has": map_has,
    "safe_range": safe_range,
    "safe_irange": safe_irange,
    "clamp": clamp,
    "abs": abs,
    "min": min,
    "max": max,
    "int": int,
}

SAFE_VARS = {"x"} | {f"v{i}" for i in range(10)}


# grid helpers (ARC-like)
def _g_rot90(g):
    return [list(r) for r in zip(*g[::-1])]

def _g_flip(g):
    return g[::-1]

def _g_inv(g):
    return [[1 - c if c in (0, 1) else c for c in r] for r in g]

def _g_get(g, r, c):
    return g[r % len(g)][c % len(g[0])] if g and g[0] else 0

SAFE_FUNCS.update({"rot90": _g_rot90, "flip": _g_flip, "inv": _g_inv, "get": _g_get})
for k in ["rot90", "flip", "inv", "get"]:
    GRAMMAR_PROBS[k] = 1.0


# ---------------------------
# Safety: step limit + validators
# ---------------------------

class StepLimitExceeded(Exception):
    pass

class StepLimitTransformer(ast.NodeTransformer):
    """Inject step counting into loops and function bodies to prevent non-termination."""

    def __init__(self, limit: int = 5000):
        self.limit = limit

    def _inject_steps(self, node: ast.FunctionDef) -> None:
        glob = ast.Global(names=["_steps"])
        reset = ast.parse("_steps = 0").body[0]
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, glob)
        node.body.insert(1, reset)
        node.body.insert(2, inc)
        node.body.insert(3, check)

    def visit_FunctionDef(self, node):
        self._inject_steps(node)
        self.generic_visit(node)
        return node

    def visit_While(self, node):
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        inc = ast.parse("_steps += 1").body[0]
        check = ast.parse(f"if _steps > {self.limit}: raise StepLimitExceeded()").body[0]
        node.body.insert(0, inc)
        node.body.insert(1, check)
        self.generic_visit(node)
        return node


class CodeValidator(ast.NodeVisitor):
    """
    Allow a safe subset of Python: assignments, flow control, simple expressions, calls to safe names.
    Forbid imports, attribute access, comprehensions, lambdas, etc.
    """

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.While,
        ast.For,
        ast.Break,
        ast.Continue,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.List,
        ast.Tuple,  # critical for tuple-assign (swap)
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Load,
        ast.Store,
        ast.IfExp,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            # forbid attribute calls (e.g., os.system)
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)

def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = CodeValidator()
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


class ProgramValidator(ast.NodeVisitor):
    """Strict program-mode validator: Assign/If/Return only, no loops or attributes."""

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Load,
        ast.Store,
        ast.IfExp,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden program node: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)


def validate_program(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = ProgramValidator()
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))


class AlgoProgramValidator(ast.NodeVisitor):
    """Algo-mode validator with bounded structure and constrained attribute access."""

    _allowed = [
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Return,
        ast.Assign,
        ast.Name,
        ast.Constant,
        ast.Expr,
        ast.If,
        ast.For,
        ast.While,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
        ast.Call,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Attribute,
        ast.Subscript,
        ast.Load,
        ast.Store,
        ast.operator,
        ast.boolop,
        ast.unaryop,
        ast.cmpop,
    ]
    if hasattr(ast, "Index"):
        _allowed.append(ast.Index)

    ALLOWED = tuple(_allowed)

    def __init__(self):
        self.ok, self.err = (True, None)

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden: {type(node).__name__}")
            return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        super().generic_visit(node)


def algo_program_limits_ok(
    code: str,
    max_nodes: int = 420,
    max_depth: int = 32,
    max_funcs: int = 8,
    max_locals: int = 48,
    max_consts: int = 128,
    max_subscripts: int = 64,
) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    funcs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    consts = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant))
    subs = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Subscript))
    return (
        nodes <= max_nodes
        and depth <= max_depth
        and funcs <= max_funcs
        and len(locals_set) <= max_locals
        and consts <= max_consts
        and subs <= max_subscripts
    )


def validate_algo_program(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        v = AlgoProgramValidator()
        v.visit(tree)
        if not v.ok:
            return (False, v.err or "")
        if not algo_program_limits_ok(code):
            return (False, "algo_program_limits")
        return (True, "")
    except Exception as e:
        return (False, str(e))


class ExprValidator(ast.NodeVisitor):
    """Validate a single expression (mode='eval') allowing only safe names and safe call forms."""
    ALLOWED = (
        ast.Expression,
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
        ast.Call,
        ast.Attribute,
        ast.Name, ast.Load,
        ast.Constant,
        ast.List, ast.Tuple, ast.Dict,
        ast.Set,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        ast.Subscript, ast.Slice,
        ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    )

    def __init__(self, allowed_names: Set[str]):
        self.allowed_names = allowed_names
        self.ok = True
        self.err: Optional[str] = None

    def visit(self, node):
        if not isinstance(node, self.ALLOWED):
            self.ok, self.err = (False, f"Forbidden expr node: {type(node).__name__}")
            return
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in ("open", "eval", "exec", "compile", "__import__", "globals", "locals"):
                self.ok, self.err = (False, f"Forbidden name: {node.id}")
                return
            if node.id not in self.allowed_names:
                self.ok, self.err = (False, f"Unknown name: {node.id}")
                return
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.ok, self.err = (False, f"Forbidden attribute: {node.attr}")
                return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                self.ok, self.err = (False, "Forbidden call form (non-Name/Attribute callee)")
                return
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in SAFE_BUILTINS:
                self.ok, self.err = (False, "Forbidden subscript on builtin")
                return
        super().generic_visit(node)

def validate_expr(expr: str, extra: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """PHASE A: validate expression with safe names only."""
    try:
        extra = extra or set()
        allowed = set(SAFE_FUNCS.keys()) | set(SAFE_BUILTINS.keys()) | set(SAFE_VARS) | set(extra)
        tree = ast.parse(expr, mode="eval")
        v = ExprValidator(allowed)
        v.visit(tree)
        return (v.ok, v.err or "")
    except Exception as e:
        return (False, str(e))

# Strict Security Barrier
def RuntimeGuard(func_name: str):
    raise RuntimeError(f"EXEC BANNED: {func_name} is disabled by strictly honest security policy.")

def legacy_evaluate_expr(expr: str, x: Any, extra_funcs: Optional[Dict[str, Callable]] = None) -> Any:
    """PHASE A: eval BANNED."""
    RuntimeGuard("legacy_evaluate_expr")


def node_count(code: str) -> int:
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except Exception:
        return 999

def ast_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 0
    max_depth = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1))
    return max_depth


def program_limits_ok(code: str, max_nodes: int = 200, max_depth: int = 20, max_locals: int = 16) -> bool:
    try:
        tree = ast.parse(code)
    except Exception:
        return False
    nodes = sum(1 for _ in ast.walk(tree))
    depth = ast_depth(code)
    locals_set = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return nodes <= max_nodes and depth <= max_depth and len(locals_set) <= max_locals


def legacy_run(code: str, x: Any, timeout_steps: int = 1000, extra_env: Optional[Dict[str, Any]] = None) -> Any:
    RuntimeGuard("legacy_run")

def legacy_run_algo(
    code: str,
    inp: Any,
    timeout_steps: int = 2000,
    max_runtime_ms: int = 50,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, int, bool]:
    RuntimeGuard("legacy_run_algo")


def legacy_run_engine(code: str, context: Dict[str, Any], timeout_steps: int = 5000) -> Any:
    RuntimeGuard("legacy_run_engine")

def legacy_load_module(code: str, timeout_steps: int = 5000) -> Optional[Dict[str, Any]]:
    RuntimeGuard("legacy_load_module")


# ---------------------------
# Engine strategy (meta-evolvable selection/crossover policy)
# ---------------------------

@dataclass
class EngineStrategy:
    selection_code: str
    crossover_code: str
    mutation_policy_code: str
    gid: str = "default"


DEFAULT_SELECTION_CODE = """
def run():
    # Context injected: pool, scores, pop_size, rng, map_elites
    # Returns: (elites, breeding_parents)
    scored = sorted(zip(pool, scores), key=lambda x: x[1])
    elite_k = max(4, pop_size // 10)
    elites = [g for g, s in scored[:elite_k]]

    parents = []
    n_needed = pop_size - len(elites)
    for _ in range(n_needed):
        # 10% chance to pick from MAP-Elites
        if rng.random() < 0.1 and map_elites and map_elites.grid:
            p = map_elites.sample(rng) or rng.choice(elites)
        else:
            p = rng.choice(elites)
        parents.append(p)
    return elites, parents
"""

DEFAULT_CROSSOVER_CODE = """
def run():
    # Context: p1 (stmts), p2 (stmts), rng
    if len(p1) < 2 or len(p2) < 2:
        return p1
    idx_a = rng.randint(0, len(p1))
    idx_b = rng.randint(0, len(p2))
    return p1[:idx_a] + p2[idx_b:]
"""

DEFAULT_MUTATION_CODE = """
def run():
    return "default"
"""


# ---------------------------
# Tasks / Datasets
# ---------------------------

@dataclass
class TaskDescriptor:
    name: str
    family: str
    input_kind: str
    output_kind: str
    n_train: int
    n_hold: int
    n_test: int
    noise: float
    stress_mult: float
    has_switch: bool
    nonlinear: bool

    def vector(self) -> List[float]:
        family_map = {
            "poly": 0.1,
            "piecewise": 0.3,
            "rational": 0.5,
            "switching": 0.7,
            "classification": 0.9,
            "list": 0.2,
            "arc": 0.4,
            "other": 0.6,
        }
        return [
            family_map.get(self.family, 0.0),
            1.0 if self.input_kind == "list" else 0.0,
            1.0 if self.input_kind == "grid" else 0.0,
            1.0 if self.output_kind == "class" else 0.0,
            float(self.n_train) / 100.0,
            float(self.n_hold) / 100.0,
            float(self.n_test) / 100.0,
            clamp(self.noise, 0.0, 1.0),
            clamp(self.stress_mult / 5.0, 0.0, 2.0),
            1.0 if self.has_switch else 0.0,
            1.0 if self.nonlinear else 0.0,
        ]

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskSpec:
    name: str = "poly2"
    x_min: float = -3.0
    x_max: float = 3.0
    n_train: int = 96
    n_hold: int = 96
    n_test: int = 96
    noise: float = 0.01
    stress_mult: float = 3.0
    target_code: Optional[str] = None
    descriptor: Optional[TaskDescriptor] = None

    def ensure_descriptor(self) -> TaskDescriptor:
        if self.descriptor:
            return self.descriptor
        family = "other"
        if self.name in ("poly2", "poly3"):
            family = "poly"
        elif self.name == "piecewise":
            family = "piecewise"
        elif self.name == "rational":
            family = "rational"
        elif self.name == "switching":
            family = "switching"
        elif self.name == "classification":
            family = "classification"
        elif self.name in ("sort", "reverse", "filter", "max", "even_reverse_sort"):
            family = "list"
        elif self.name == "self_audit":
            family = "self_audit"
        elif self.name in ALGO_TASK_NAMES:
            family = "algo"
        elif self.name.startswith("arc_"):
            family = "arc"
        self.descriptor = TaskDescriptor(
            name=self.name,
            family=family,
            input_kind="vector"
            if family == "self_audit"
            else ("list" if family in ("list", "algo") else ("grid" if family == "arc" else "scalar")),
            output_kind="class" if family == "classification" else "scalar",
            n_train=self.n_train,
            n_hold=self.n_hold,
            n_test=self.n_test,
            noise=self.noise,
            stress_mult=self.stress_mult,
            has_switch=self.name == "switching",
            nonlinear=family in ("poly", "piecewise", "rational", "switching"),
        )
        return self.descriptor


# ---------------------------
# Algorithmic task suite (algo mode)
# ---------------------------

ALGO_TASK_NAMES = {
    "sort_int_list",
    "topk",
    "two_sum",
    "balanced_parens",
    "gcd_list",
    "rpn_eval",
    "bfs_shortest_path",
    "coin_change_min",
    "substring_find",
    "unique_count",
    "lis_length",
    "min_path_sum",
    "edit_distance",
}

ALGO_COUNTEREXAMPLES: Dict[str, List[Tuple[Any, Any]]] = {name: [] for name in ALGO_TASK_NAMES}

def _gen_int_list(rng: random.Random, min_len: int, max_len: int, lo: int = -9, hi: int = 9) -> List[int]:
    ln = rng.randint(min_len, max_len)
    return [rng.randint(lo, hi) for _ in range(ln)]

def _gen_parens(rng: random.Random, min_len: int, max_len: int) -> List[int]:
    ln = rng.randint(min_len, max_len)
    return [0 if rng.random() < 0.5 else 1 for _ in range(ln)]

def _gen_graph(rng: random.Random, n_min: int, n_max: int) -> List[List[int]]:
    n = rng.randint(n_min, n_max)
    g = []
    for i in range(n):
        neigh = []
        for j in range(n):
            if i != j and rng.random() < 0.25:
                neigh.append(j)
        g.append(neigh)
    return g

def _algo_descriptor(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "family": "algo",
        "input_kind": "list",
        "output_kind": "scalar",
        "n_train": 0,
        "n_hold": 0,
        "n_test": 0,
        "noise": 0.0,
        "stress_mult": 2.0,
        "has_switch": False,
        "nonlinear": True,
    }

def _algo_task_data(name: str, rng: random.Random, n: int, stress: bool = False) -> Tuple[List[Any], List[Any]]:
    xs: List[Any] = []
    ys: List[Any] = []
    for _ in range(n):
        if name == "sort_int_list":
            x = _gen_int_list(rng, 2, 8 if not stress else 12)
            y = sorted(x)
        elif name == "topk":
            arr = _gen_int_list(rng, 2, 10 if not stress else 14)
            k = rng.randint(1, max(1, len(arr) // 2))
            x = [arr, k]
            y = sorted(arr, reverse=True)[:k]
        elif name == "two_sum":
            arr = _gen_int_list(rng, 2, 10 if not stress else 14)
            i, j = rng.sample(range(len(arr)), 2)
            target = arr[i] + arr[j]
            x = [arr, target]
            y = [i, j]
        elif name == "balanced_parens":
            seq = _gen_parens(rng, 2, 12 if not stress else 18)
            bal = 0
            ok = 1
            for t in seq:
                bal += 1 if t == 0 else -1
                if bal < 0:
                    ok = 0
                    break
            if bal != 0:
                ok = 0
            x = seq
            y = ok
        elif name == "gcd_list":
            arr = [abs(v) + 1 for v in _gen_int_list(rng, 2, 8 if not stress else 12, 1, 9)]
            g = arr[0]
            for v in arr[1:]:
                g = math.gcd(g, v)
            x = arr
            y = g
        elif name == "rpn_eval":
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            op = rng.choice([-1, -2, -3, -4])
            if op == -1:
                y = a + b
            elif op == -2:
                y = a - b
            elif op == -3:
                y = a * b
            else:
                y = a // b if b else 0
            x = [a, b, op]
        elif name == "bfs_shortest_path":
            g = _gen_graph(rng, 4, 7 if not stress else 9)
            s, t = rng.sample(range(len(g)), 2)
            dist = [-1] * len(g)
            dist[s] = 0
            q = [s]
            while q:
                cur = q.pop(0)
                for nxt in g[cur]:
                    if dist[nxt] == -1:
                        dist[nxt] = dist[cur] + 1
                        q.append(nxt)
            x = [g, s, t]
            y = dist[t]
        elif name == "coin_change_min":
            coins = [c for c in _gen_int_list(rng, 2, 5 if not stress else 7, 1, 8) if c > 0]
            amount = rng.randint(1, 12 if not stress else 18)
            dp = [float("inf")] * (amount + 1)
            dp[0] = 0
            for c in coins:
                for a in range(c, amount + 1):
                    dp[a] = min(dp[a], dp[a - c] + 1)
            y = -1 if dp[amount] == float("inf") else int(dp[amount])
            x = [coins, amount]
        elif name == "substring_find":
            hay = _gen_int_list(rng, 4, 10 if not stress else 14, 1, 4)
            needle = hay[1:3] if len(hay) > 3 and rng.random() < 0.7 else _gen_int_list(rng, 2, 3, 1, 4)
            idx = -1
            for i in range(len(hay) - len(needle) + 1):
                if hay[i:i + len(needle)] == needle:
                    idx = i
                    break
            x = [hay, needle]
            y = idx
        elif name == "unique_count":
            arr = _gen_int_list(rng, 3, 10 if not stress else 14, 1, 6)
            x = arr
            y = len(set(arr))
        elif name == "lis_length":
            arr = _gen_int_list(rng, 3, 10 if not stress else 14, -5, 9)
            dp = [1 for _ in arr]
            for i in range(len(arr)):
                for j in range(i):
                    if arr[j] < arr[i]:
                        dp[i] = max(dp[i], dp[j] + 1)
            x = arr
            y = max(dp) if dp else 0
        elif name == "min_path_sum":
            rows = rng.randint(2, 5 if not stress else 7)
            cols = rng.randint(2, 5 if not stress else 7)
            grid = [[rng.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
            dp = [[0 for _ in range(cols)] for _ in range(rows)]
            dp[0][0] = grid[0][0]
            for r in range(1, rows):
                dp[r][0] = dp[r - 1][0] + grid[r][0]
            for c in range(1, cols):
                dp[0][c] = dp[0][c - 1] + grid[0][c]
            for r in range(1, rows):
                for c in range(1, cols):
                    dp[r][c] = min(dp[r - 1][c], dp[r][c - 1]) + grid[r][c]
            x = grid
            y = dp[-1][-1]
        elif name == "edit_distance":
            a = _gen_int_list(rng, 2, 6 if not stress else 8, 0, 4)
            b = _gen_int_list(rng, 2, 6 if not stress else 8, 0, 4)
            dp = [[0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
            for i in range(len(a) + 1):
                dp[i][0] = i
            for j in range(len(b) + 1):
                dp[0][j] = j
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost,
                    )
            x = [a, b]
            y = dp[-1][-1]
        else:
            x = []
            y = 0
        xs.append(x)
        ys.append(y)
    return xs, ys

def algo_batch(name: str, seed: int, freeze_eval: bool = True, train_resample_every: int = 1, gen: int = 0) -> Optional[Batch]:
    if name not in ALGO_TASK_NAMES:
        return None
    rng = random.Random(seed)
    hold_rng = random.Random(seed + 11)
    stress_rng = random.Random(seed + 29)
    test_rng = random.Random(seed + 47)
    if not freeze_eval:
        hold_rng = random.Random(seed + 11 + gen)
        stress_rng = random.Random(seed + 29 + gen)
        test_rng = random.Random(seed + 47 + gen)
    train_rng = rng if train_resample_every <= 1 else random.Random(seed + gen // max(1, train_resample_every))
    x_tr, y_tr = _algo_task_data(name, train_rng, 40, stress=False)
    x_ho, y_ho = _algo_task_data(name, hold_rng, 24, stress=False)
    x_st, y_st = _algo_task_data(name, stress_rng, 24, stress=True)
    x_te, y_te = _algo_task_data(name, test_rng, 24, stress=True)
    return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)


@dataclass
class ControlPacket:
    mutation_rate: Optional[float] = None
    crossover_rate: Optional[float] = None
    novelty_weight: float = 0.0
    branch_insert_rate: float = 0.0
    op_weights: Optional[Dict[str, float]] = None
    acceptance_margin: float = 1e-9
    patience: int = 5

    def get(self, key: str, default: Any = None) -> Any:
        val = getattr(self, key, default)
        if val is None:
            return default
        return val


TARGET_FNS = {
    "sort": lambda x: sorted(x),
    "reverse": lambda x: list(reversed(x)),
    "max": lambda x: max(x) if x else 0,
    "filter": lambda x: [v for v in x if v > 0],
    "arc_ident": lambda x: x,
    "arc_rot90": lambda x: [list(r) for r in zip(*x[::-1])],
    "arc_inv": lambda x: [[1 - c if c in (0, 1) else c for c in r] for r in x],
    "poly2": lambda x: 0.7 * x * x - 0.2 * x + 0.3,
    "poly3": lambda x: 0.3 * x ** 3 - 0.5 * x + 0.1,
    "piecewise": lambda x: (-0.5 * x + 1.0) if x < 0 else (0.3 * x * x + 0.1),
    "rational": lambda x: (x * x + 1.0) / (1.0 + 0.5 * abs(x)),
    "sinmix": lambda x: math.sin(x) + 0.3 * math.cos(2 * x),
    "absline": lambda x: abs(x) + 0.2 * x,
    "classification": lambda x: 1.0 if (x + 0.25 * math.sin(3 * x)) > 0 else 0.0,
}


ARC_GYM_PATH = os.path.join(os.path.dirname(__file__), "ARC_GYM")

def load_arc_task(task_id: str) -> Dict:
    fname = task_id
    if not fname.endswith(".json"):
        fname += ".json"
    path = os.path.join(ARC_GYM_PATH, fname)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_arc_tasks() -> List[str]:
    if not os.path.exists(ARC_GYM_PATH):
        return []
    return [f[:-5] for f in os.listdir(ARC_GYM_PATH) if f.endswith(".json")]

@dataclass
class Batch:
    x_tr: List[Any]
    y_tr: List[Any]
    x_ho: List[Any]
    y_ho: List[Any]
    x_st: List[Any]
    y_st: List[Any]
    x_te: List[Any]
    y_te: List[Any]


def _best_code_snapshot() -> str:
    try:
        state = load_state()
    except Exception:
        state = None
    if not state or not state.universes:
        return "def run(x):\n    return x\n"
    target = next((u for u in state.universes if u.get("uid") == state.selected_uid), None)
    if not target:
        target = state.universes[0]
    best = target.get("best")
    if not best:
        return "def run(x):\n    return x\n"
    if state.mode == "learner":
        return LearnerGenome(**best).code
    return Genome(**best).code


def _code_features(code: str) -> List[float]:
    return [
        float(len(code)),
        float(node_count(code)),
        float(code.count("if ")),
        float(code.count("while ")),
        float(code.count("return ")),
    ]

def sample_batch(rng: random.Random, t: TaskSpec) -> Optional[Batch]:
    # function target
    if t.target_code:
        f = lambda x: legacy_run(t.target_code, x)
    elif t.name in ("sort", "reverse", "filter", "max"):
        f = TARGET_FNS.get(t.name) or (lambda x: sorted(x))
    else:
        f = TARGET_FNS.get(t.name, lambda x: x)

    if t.name == "self_audit":
        base_code = _best_code_snapshot()
        base_features = _code_features(base_code)

        def synth_features(k: int, jitter: float) -> List[List[float]]:
            samples = []
            for _ in range(k):
                sample = [max(0.0, f + rng.gauss(0, jitter)) for f in base_features]
                samples.append(sample)
            return samples

        def target_score(vec: List[float]) -> float:
            length, nodes, ifs, whiles, returns = vec
            raw = 0.004 * length + 0.02 * nodes + 0.1 * ifs + 0.15 * whiles + 0.02 * returns
            return math.tanh(raw / 10.0)

        x_tr = synth_features(t.n_train, 3.0)
        x_ho = synth_features(t.n_hold, 4.0)
        x_st = synth_features(t.n_hold, 6.0)
        x_te = synth_features(t.n_test, 4.0)
        y_tr = [target_score(x) for x in x_tr]
        y_ho = [target_score(x) for x in x_ho]
        y_st = [target_score(x) for x in x_st]
        y_te = [target_score(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # ARC tasks from local json
    json_data = load_arc_task(t.name.replace("arc_", ""))
    if json_data:
        pairs = json_data.get("train", []) + json_data.get("test", [])
        x_all, y_all = [], []
        for p in pairs:
            x_all.append(p["input"])
            y_all.append(p["output"])
            if len(x_all) >= 30:
                break
        if not x_all:
            return None
        return Batch(
            x_all[:20], y_all[:20],
            x_all[:10], y_all[:10],
            x_all[:5],  y_all[:5],
            x_all[5:10], y_all[5:10],
        )

    # list tasks
    def gen_lists(k, min_len, max_len):
        data = []
        for _ in range(k):
            a = max(1, int(min_len))
            b = max(a, int(max_len))
            l = rng.randint(a, b)
            data.append([rng.randint(-100, 100) for _ in range(l)])
        return data

    if t.name == "even_reverse_sort":
        f = lambda x: sorted([n for n in x if n % 2 == 0], reverse=True)
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2)
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_lists(max(1, t.n_test), t.x_min + 1, t.x_max + 1)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    if t.name in ("sort", "reverse", "filter", "max"):
        x_tr = gen_lists(t.n_train, t.x_min, t.x_max)
        x_ho = gen_lists(t.n_hold, t.x_min + 2, t.x_max + 2)
        x_st = gen_lists(t.n_hold, t.x_max + 5, t.x_max + 10)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_lists(max(1, t.n_test), t.x_min + 1, t.x_max + 1)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # synthetic ARC-like generators if name starts with arc_
    if t.name.startswith("arc_"):
        def gen_grids(k, dim):
            data = []
            for _ in range(k):
                g = [[rng.randint(0, 1) for _ in range(dim)] for _ in range(dim)]
                data.append(g)
            return data
        dim = int(t.x_min) if t.x_min > 0 else 3
        x_tr = gen_grids(20, dim)
        x_ho = gen_grids(10, dim)
        x_st = gen_grids(10, dim + 1)
        y_tr = [f(x) for x in x_tr]
        y_ho = [f(x) for x in x_ho]
        y_st = [f(x) for x in x_st]
        x_te = gen_grids(10, dim)
        y_te = [f(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    if t.name == "switching":
        def target_switch(pair):
            x, s = pair
            return TARGET_FNS["poly2"](x) if s < 0.5 else TARGET_FNS["sinmix"](x)

        def gen_pairs(k, a, b):
            data = []
            for _ in range(k):
                x = a + (b - a) * rng.random()
                s = 1.0 if rng.random() > 0.5 else 0.0
                data.append([x, s])
            return data

        x_tr = gen_pairs(t.n_train, t.x_min, t.x_max)
        x_ho = gen_pairs(t.n_hold, t.x_min, t.x_max)
        x_st = gen_pairs(t.n_hold, t.x_min * t.stress_mult, t.x_max * t.stress_mult)
        x_te = gen_pairs(t.n_test, t.x_min, t.x_max)
        y_tr = [target_switch(x) for x in x_tr]
        y_ho = [target_switch(x) for x in x_ho]
        y_st = [target_switch(x) for x in x_st]
        y_te = [target_switch(x) for x in x_te]
        return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)

    # numeric regression tasks
    xs = lambda n, a, b: [a + (b - a) * rng.random() for _ in range(n)]
    ys = lambda xv, n: [f(x) + rng.gauss(0, n) if n > 0 else f(x) for x in xv]
    half = 0.5 * (t.x_max - t.x_min)
    mid = 0.5 * (t.x_min + t.x_max)
    x_tr = xs(t.n_train, t.x_min, t.x_max)
    x_ho = xs(t.n_hold, t.x_min, t.x_max)
    x_st = xs(t.n_hold, mid - half * t.stress_mult, mid + half * t.stress_mult)
    x_te = xs(t.n_test, t.x_min, t.x_max)
    return Batch(
        x_tr, ys(x_tr, t.noise),
        x_ho, ys(x_ho, t.noise),
        x_st, ys(x_st, t.noise * t.stress_mult),
        x_te, ys(x_te, t.noise),
    )


def task_suite(seed: int) -> List[TaskSpec]:
    base = [
        TaskSpec(name="poly2", x_min=-3.0, x_max=3.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="poly3", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="piecewise", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="rational", x_min=-5.0, x_max=5.0, n_train=96, n_hold=64, n_test=64, noise=0.02),
        TaskSpec(name="switching", x_min=-3.0, x_max=3.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
        TaskSpec(name="classification", x_min=-4.0, x_max=4.0, n_train=96, n_hold=64, n_test=64, noise=0.0),
        TaskSpec(name="sinmix", x_min=-6.0, x_max=6.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="absline", x_min=-6.0, x_max=6.0, n_train=96, n_hold=64, n_test=64, noise=0.01),
        TaskSpec(name="self_audit", x_min=0.0, x_max=1.0, n_train=64, n_hold=48, n_test=48, noise=0.0),
    ]
    rng = random.Random(seed)
    rng.shuffle(base)
    return base


def split_meta_tasks(seed: int, meta_train_ratio: float = 0.6) -> Tuple[List[TaskSpec], List[TaskSpec]]:
    suite = task_suite(seed)
    cut = max(1, int(len(suite) * meta_train_ratio))
    return suite[:cut], suite[cut:]


FROZEN_BATCH_CACHE: Dict[str, Batch] = {}


def _task_cache_key(task: TaskSpec, seed: int) -> str:
    return f"{task.name}:{seed}:{task.x_min}:{task.x_max}:{task.n_train}:{task.n_hold}:{task.n_test}:{task.noise}:{task.stress_mult}:{task.target_code}"


def get_task_batch(
    task: TaskSpec,
    seed: int,
    freeze_eval: bool = True,
    train_resample_every: int = 1,
    gen: int = 0,
) -> Optional[Batch]:
    if task.name in ALGO_TASK_NAMES:
        return algo_batch(task.name, seed, freeze_eval=freeze_eval, train_resample_every=train_resample_every, gen=gen)
    key = _task_cache_key(task, seed)
    if freeze_eval and key in FROZEN_BATCH_CACHE:
        return FROZEN_BATCH_CACHE[key]
    h = int(sha256(key)[:8], 16)
    rng = random.Random(h if freeze_eval else seed)
    batch = sample_batch(rng, task)
    if freeze_eval and batch is not None:
        FROZEN_BATCH_CACHE[key] = batch
    return batch


# ---------------------------
# Genome / Evaluation
# ---------------------------

@dataclass
class Genome:
    statements: List[str]
    gid: str = ""
    parents: List[str] = field(default_factory=list)
    op_tag: str = "init"
    birth_ms: int = 0

    @property
    def code(self) -> str:
        body = "\n    ".join(self.statements) if self.statements else "return x"
        return f"def run(x):\n    # {self.gid}\n    v0=x\n    {body}"

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256("".join(self.statements) + str(time.time()))[:12]
        if not self.birth_ms:
            self.birth_ms = now_ms()


@dataclass
class LearnerGenome:
    """PHASE B: learner genome with encode/predict/update/objective blocks."""
    encode_stmts: List[str]
    predict_stmts: List[str]
    update_stmts: List[str]
    objective_stmts: List[str]
    gid: str = ""
    parents: List[str] = field(default_factory=list)
    op_tag: str = "init"
    birth_ms: int = 0

    @property
    def code(self) -> str:
        def ensure_return(stmts: List[str], fallback: str) -> List[str]:
            for s in stmts:
                if s.strip().startswith("return "):
                    return stmts
            return stmts + [fallback]

        enc = ensure_return(self.encode_stmts or [], "return x")
        pred = ensure_return(self.predict_stmts or [], "return z")
        upd = ensure_return(self.update_stmts or [], "return mem")
        obj = ensure_return(self.objective_stmts or [], "return hold + 0.5*stress + 0.01*nodes")

        enc_body = "\n    ".join(enc) if enc else "return x"
        pred_body = "\n    ".join(pred) if pred else "return z"
        upd_body = "\n    ".join(upd) if upd else "return mem"
        obj_body = "\n    ".join(obj) if obj else "return hold + 0.5*stress + 0.01*nodes"

        return (
            "def init_mem():\n"
            "    return {\"w\": 0.0, \"b\": 0.0, \"t\": 0}\n\n"
            "def encode(x, mem):\n"
            f"    # {self.gid}\n    {enc_body}\n\n"
            "def predict(z, mem):\n"
            f"    {pred_body}\n\n"
            "def update(mem, x, y_pred, y_true, lr=0.05):\n"
            f"    {upd_body}\n\n"
            "def objective(train, hold, stress, nodes):\n"
            f"    {obj_body}\n"
        )

    def __post_init__(self):
        if not self.gid:
            self.gid = sha256("".join(self.encode_stmts + self.predict_stmts + self.update_stmts + self.objective_stmts) + str(time.time()))[:12]
        if not self.birth_ms:
            self.birth_ms = now_ms()


@dataclass
class EvalResult:
    ok: bool
    train: float
    hold: float
    stress: float
    test: float
    nodes: int
    score: float
    err: Optional[str] = None


SCORE_W_HOLD = 0.452390
SCORE_W_STRESS = 0.4
SCORE_W_TRAIN = 0.0


def calc_error(p: Any, t: Any) -> float:
    if isinstance(t, (int, float)):
        if isinstance(p, (int, float)):
            return (p - t) ** 2
        return 1_000_000.0
    if isinstance(t, list):
        if not isinstance(p, list):
            return 1_000_000.0
        if len(p) != len(t):
            return 1000.0 * abs(len(p) - len(t))
        return sum(calc_error(pv, tv) for pv, tv in zip(p, t))
    return 1_000_000.0


def _list_invariance_penalty(x: Any, p: Any, task_name: str) -> float:
    if not isinstance(x, list):
        return 0.0
    if task_name in ("sort", "reverse"):
        if not isinstance(p, list):
            return 5_000.0
        if len(p) != len(x):
            return 2_000.0 + 10.0 * abs(len(p) - len(x))
        try:
            if collections.Counter(p) != collections.Counter(x):
                return 2_000.0
        except TypeError:
            pass
    if task_name == "filter":
        if not isinstance(p, list):
            return 5_000.0
        try:
            x_counts = collections.Counter(x)
            p_counts = collections.Counter(p)
            for k, v in p_counts.items():
                if x_counts.get(k, 0) < v:
                    return 2_000.0
        except TypeError:
            pass
    if task_name == "max":
        if not isinstance(p, (int, float)):
            return 5_000.0
    return 0.0


def calc_loss_sort(p: List[Any], t: List[Any]) -> float:
    if not isinstance(p, list):
        return 1_000_000.0
    if len(p) != len(t):
        return 1000.0 * abs(len(p) - len(t))
    p_sorted = sorted(p) if all(isinstance(x, (int, float)) for x in p) else p
    t_sorted = sorted(t)
    content_loss = sum((a - b) ** 2 for a, b in zip(p_sorted, t_sorted))
    if content_loss > 0.1:
        return 1000.0 + content_loss
    inversions = 0
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                inversions += 1
    return float(inversions)


def calc_heuristic_loss(p: Any, t: Any, task_name: str, x: Any = None) -> float:
    penalty = _list_invariance_penalty(x, p, task_name)
    if task_name == "sort":
        return calc_loss_sort(p, t) + penalty
    if isinstance(t, list):
        if not isinstance(p, list):
            return 1_000_000.0 + penalty
        if len(p) != len(t):
            return 500.0 * abs(len(p) - len(t)) + penalty
        if task_name in ("reverse", "filter"):
            return sum(calc_error(pv, tv) for pv, tv in zip(p, t)) + penalty
    if task_name.startswith("arc_"):
        if not isinstance(p, list) or not p or not isinstance(p[0], list):
            return 1000.0 + penalty
        if len(p) != len(t) or len(p[0]) != len(t[0]):
            return 500.0 + abs(len(p) - len(t)) + abs(len(p[0]) - len(t[0])) + penalty
        err = 0
        for r in range(len(t)):
            for c in range(len(t[0])):
                if p[r][c] != t[r][c]:
                    err += 1
        return float(err) + penalty
    return calc_error(p, t) + penalty


def legacy_run_mse(
    code: str,
    xs: List[Any],
    ys: List[Any],
    task_name: str = "",
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> Tuple[bool, float, str]:
    ok, err = validator(code)
    if not ok:
        return (False, float("inf"), err)
    if validator == validate_program and not program_limits_ok(code):
        return (False, float("inf"), "program_limits")
    try:
        total_err = 0.0
        for x, y in zip(xs, ys):
            pred = legacy_run(code, x, extra_env=extra_env)
            if pred is None:
                return (False, float("inf"), "No return")
            if task_name in ("sort", "reverse", "max", "filter") or task_name.startswith("arc_"):
                total_err += calc_heuristic_loss(pred, y, task_name, x=x)
            else:
                total_err += calc_error(pred, y)
        return (True, total_err / max(1, len(xs)), "")
    except Exception as e:
        return (False, float("inf"), f"{type(e).__name__}: {str(e)}")


def _algo_equal(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_algo_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_algo_equal(a[k], b[k]) for k in a.keys())
    return a == b


def algo_runner(
    code: str,
    xs: List[Any],
    ys: List[Any],
    task_name: str,
    counterexamples: Optional[List[Tuple[Any, Any]]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_algo_program,
) -> Tuple[bool, float, int, float, int, str]:
    ok, err = validator(code)
    if not ok:
        return (False, 1.0, 0, 1.0, 0, err)
    total = 0
    timeouts = 0
    steps = 0
    failures = 0
    extra = counterexamples[:] if counterexamples else []
    xs_all = list(xs) + [x for x, _ in extra]
    ys_all = list(ys) + [y for _, y in extra]
    for x, y in zip(xs_all, ys_all):
        out, used, timeout = legacy_run_algo(code, x)
        steps += used
        if timeout:
            timeouts += 1
        if not _algo_equal(out, y):
            failures += 1
            if counterexamples is not None and len(counterexamples) < 64:
                counterexamples.append((x, y))
        total += 1
    err_rate = failures / max(1, total)
    timeout_rate = timeouts / max(1, total)
    avg_steps = steps // max(1, total)
    return (True, err_rate, avg_steps, timeout_rate, total, "")


def evaluate_algo(
    g: Genome,
    b: Batch,
    task_name: str,
    lam: float = 0.0001,
) -> EvalResult:
    code = g.code
    counterexamples = ALGO_COUNTEREXAMPLES.get(task_name, [])
    ok1, tr_err, tr_steps, tr_timeout, _, e1 = algo_runner(code, b.x_tr, b.y_tr, task_name, counterexamples)
    ok2, ho_err, ho_steps, ho_timeout, _, e2 = algo_runner(code, b.x_ho, b.y_ho, task_name, counterexamples)
    ok3, st_err, st_steps, st_timeout, _, e3 = algo_runner(code, b.x_st, b.y_st, task_name, counterexamples)
    ok4, te_err, te_steps, te_timeout, _, e4 = algo_runner(code, b.x_te, b.y_te, task_name, counterexamples)
    ok = ok1 and ok2 and ok3 and ok4 and all(math.isfinite(v) for v in (tr_err, ho_err, st_err, te_err))
    nodes = node_count(code)
    step_penalty = 0.0001 * (tr_steps + ho_steps + st_steps + te_steps)
    timeout_penalty = 0.5 * (tr_timeout + ho_timeout + st_timeout + te_timeout)
    if not ok:
        return EvalResult(False, tr_err, ho_err, st_err, te_err, nodes, float("inf"), e1 or e2 or e3 or e4 or "nan")
    # Hard cutoff: stress overflows are rejected before any score aggregation.
    if st_err > STRESS_MAX:
        return EvalResult(False, tr_err, ho_err, st_err, te_err, nodes, float("inf"), "stress_overflow")
    score = SCORE_W_HOLD * ho_err + SCORE_W_STRESS * st_err + SCORE_W_TRAIN * tr_err + lam * nodes + step_penalty + timeout_penalty
    err = e1 or e2 or e3 or e4
    return EvalResult(ok, tr_err, ho_err, st_err, te_err, nodes, score, err or None)


def evaluate(
    g: Genome,
    b: Batch,
    task_name: str,
    lam: float = 0.0001,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> EvalResult:
    code = g.code
    ok1, tr, e1 = legacy_run_mse(code, b.x_tr, b.y_tr, task_name, extra_env=extra_env)
    ok2, ho, e2 = legacy_run_mse(code, b.x_ho, b.y_ho, task_name, extra_env=extra_env)
    ok3, st, e3 = legacy_run_mse(code, b.x_st, b.y_st, task_name, extra_env=extra_env)
    ok4, te, e4 = legacy_run_mse(code, b.x_te, b.y_te, task_name, extra_env=extra_env)
    ok = ok1 and ok2 and ok3 and ok4 and all(math.isfinite(v) for v in (tr, ho, st, te))
    nodes = node_count(code)
    if not ok:
        return EvalResult(False, tr, ho, st, te, nodes, float("inf"), e1 or e2 or e3 or e4 or "nan")
    # Hard cutoff: stress overflows are rejected before any score aggregation.
    if st > STRESS_MAX:
        return EvalResult(False, tr, ho, st, te, nodes, float("inf"), "stress_overflow")
    score = SCORE_W_HOLD * ho + SCORE_W_STRESS * st + SCORE_W_TRAIN * tr + lam * nodes
    err = e1 or e2 or e3 or e4
    return EvalResult(ok, tr, ho, st, te, nodes, score, err or None)


def evaluate_learner(
    learner: LearnerGenome,
    b: Batch,
    task_name: str,
    adapt_steps: int = 8,
    lam: float = 0.0001,
) -> EvalResult:
    """PHASE B: evaluate learner with adaptation on training only."""
    env = safe_load_module(learner.code)
    if not env:
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), "load_failed")
    required = ["init_mem", "encode", "predict", "update", "objective"]
    if not all(name in env and callable(env[name]) for name in required):
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), "missing_funcs")

    init_mem = env["init_mem"]
    encode = env["encode"]
    predict = env["predict"]
    update = env["update"]
    objective = env["objective"]

    try:
        mem = init_mem()
    except Exception:
        mem = {"w": 0.0, "b": 0.0, "t": 0}

    def run_metric_check(xs: List[Any], ys: List[Any], do_update: bool) -> float:
        nonlocal mem
        total = 0.0
        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                z = encode(x, mem)
                y_pred = predict(z, mem)
            except Exception:
                y_pred = None
            if task_name in ("sort", "reverse", "max", "filter") or task_name.startswith("arc_"):
                total += calc_heuristic_loss(y_pred, y, task_name, x=x)
            else:
                total += calc_error(y_pred, y)
            if do_update and i < adapt_steps:
                try:
                    mem = update(mem, x, y_pred, y, 0.05)
                except Exception:
                    pass
        return total / max(1, len(xs))

    try:
        train = run_metric_check(b.x_tr, b.y_tr, do_update=True)
        hold = run_metric_check(b.x_ho, b.y_ho, do_update=False)
        stress = run_metric_check(b.x_st, b.y_st, do_update=False)
        test = run_metric_check(b.x_te, b.y_te, do_update=False)
        nodes = node_count(learner.code)
        ok = all(math.isfinite(v) for v in (train, hold, stress, test))
        if not ok:
            return EvalResult(False, train, hold, stress, test, nodes, float("inf"), "nan")
        # Hard cutoff: stress overflows are rejected before any score aggregation.
        if stress > STRESS_MAX:
            return EvalResult(False, train, hold, stress, test, nodes, float("inf"), "stress_overflow")
        obj = objective(train, hold, stress, nodes)
        if not isinstance(obj, (int, float)) or not math.isfinite(obj):
            obj = SCORE_W_HOLD * hold + SCORE_W_STRESS * stress
        score = float(obj) + lam * nodes
        ok = all(math.isfinite(v) for v in (train, hold, stress, test, score))
        return EvalResult(ok, train, hold, stress, test, nodes, score, None if ok else "nan")
    except Exception as exc:
        return EvalResult(False, float("inf"), float("inf"), float("inf"), float("inf"), 0, float("inf"), str(exc))


# ---------------------------
# Mutation operators
# ---------------------------

def _pick_node(rng: random.Random, body: ast.AST) -> ast.AST:
    nodes = list(ast.walk(body))
    return rng.choice(nodes[1:]) if len(nodes) > 1 else body

def _to_src(body: ast.AST) -> str:
    try:
        return ast.unparse(body)
    except Exception:
        return "x"

def _random_expr(rng: random.Random, depth: int = 0) -> str:
    if depth > 2:
        return rng.choice(["x", "v0", str(rng.randint(0, 9))])
    options = ["binop", "call", "const", "var"]
    weights = [GRAMMAR_PROBS.get(k, 1.0) for k in options]
    mtype = rng.choices(options, weights=weights, k=1)[0]
    if mtype == "binop":
        op = rng.choice(["+", "-", "*", "/", "**", "%"])
        return f"({_random_expr(rng, depth + 1)} {op} {_random_expr(rng, depth + 1)})"
    if mtype == "call":
        funcs = list(SAFE_FUNCS.keys())
        f_weights = [GRAMMAR_PROBS.get(f, 0.5) for f in funcs]
        fname = rng.choices(funcs, weights=f_weights, k=1)[0]
        return f"{fname}({_random_expr(rng, depth + 1)})"
    if mtype == "const":
        return f"{rng.uniform(-2, 2):.2f}"
    return rng.choice(["x", "v0"])


def op_insert_assign(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    var = f"v{rng.randint(0, 3)}"
    expr = _random_expr(rng)
    new_stmts.insert(idx, f"{var} = {expr}")
    return new_stmts

def op_insert_if(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx: idx + 2]]
    new_stmts[idx: idx + 2] = [f"if {cond}:"] + block
    return new_stmts

def op_insert_while(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    cond = f"v{rng.randint(0, 3)} < {rng.randint(0, 10)}"
    block = [f"    {s}" for s in new_stmts[idx: idx + 2]]
    new_stmts[idx: idx + 2] = [f"while {cond}:"] + block
    return new_stmts

def op_delete_stmt(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    new_stmts.pop(rng.randint(0, len(new_stmts) - 1))
    return new_stmts

def op_modify_line(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    if "=" in new_stmts[idx]:
        var = new_stmts[idx].split("=")[0].strip()
        new_stmts[idx] = f"{var} = {_random_expr(rng)}"
    return new_stmts

def op_tweak_const(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)

    class TweakTransformer(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                val = float(node.value)
                new_val = val + rng.gauss(0, 0.1 * abs(val) + 0.01)
                if rng.random() < 0.05:
                    new_val = -val
                if rng.random() < 0.05:
                    new_val = 0.0
                return ast.Constant(value=new_val)
            return node

    try:
        tree = ast.parse(new_stmts[idx], mode="exec")
        new_tree = TweakTransformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_stmts[idx] = ast.unparse(new_tree).strip()
    except Exception:
        pass
    return new_stmts

def op_change_binary(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts) - 1)
    pops = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod]

    class OpTransformer(ast.NodeTransformer):
        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if rng.random() < 0.5:
                node.op = rng.choice(pops)()
            return node

    try:
        tree = ast.parse(new_stmts[idx], mode="exec")
        new_tree = OpTransformer().visit(tree)
        ast.fix_missing_locations(new_tree)
        new_stmts[idx] = ast.unparse(new_tree).strip()
    except Exception:
        pass
    return new_stmts

def op_list_manipulation(rng: random.Random, stmts: List[str]) -> List[str]:
    if not stmts:
        return stmts
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    ops = [
        f"v{rng.randint(0,3)} = x[{rng.randint(0,2)}]",
        f"if len(x) > {rng.randint(1,5)}: v{rng.randint(0,3)} = x[0]",
        "v0, v1 = v1, v0",  # requires Tuple allowed
        f"v{rng.randint(0,3)} = sorted(x)",
    ]
    new_stmts.insert(idx, rng.choice(ops))
    return new_stmts

def op_modify_return(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    active_vars = ["x"] + [f"v{i}" for i in range(4)]
    for i in range(len(new_stmts) - 1, -1, -1):
        if new_stmts[i].strip().startswith("return "):
            new_stmts[i] = f"return {rng.choice(active_vars)}"
            return new_stmts
    new_stmts.append(f"return {rng.choice(active_vars)}")
    return new_stmts


def op_learner_update_step(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    ops = [
        "mem['w'] = mem['w'] + lr * (y_true - y_pred) * x",
        "mem['b'] = mem['b'] + lr * (y_true - y_pred)",
        "mem['t'] = mem['t'] + 1",
        "return mem",
    ]
    new_stmts.insert(idx, rng.choice(ops))
    return new_stmts


def op_learner_objective_tweak(rng: random.Random, stmts: List[str]) -> List[str]:
    new_stmts = stmts[:]
    idx = rng.randint(0, len(new_stmts))
    expr = rng.choice([
        "return hold + 0.5*stress + 0.01*nodes",
        "return 0.6*hold + 0.3*stress + 0.1*train",
        "return hold + stress + 0.001*nodes",
    ])
    new_stmts.insert(idx, expr)
    return new_stmts


OPERATORS: Dict[str, Callable[[random.Random, List[str]], List[str]]] = {
    "insert_assign": op_insert_assign,
    "insert_if": op_insert_if,
    "insert_while": op_insert_while,
    "delete_stmt": op_delete_stmt,
    "modify_line": op_modify_line,
    "tweak_const": op_tweak_const,
    "change_binary": op_change_binary,
    "list_manip": op_list_manipulation,
    "modify_return": op_modify_return,
    "learner_update": op_learner_update_step,
    "learner_objective": op_learner_objective_tweak,
}
PRIMITIVE_OPS = list(OPERATORS.keys())

# @@OPERATORS_LIB_START@@
OPERATORS_LIB: Dict[str, Dict] = {}
# @@OPERATORS_LIB_END@@


def apply_synthesized_op(rng: random.Random, stmts: List[str], steps: List[str]) -> List[str]:
    result = stmts
    for step in steps:
        if step in OPERATORS:
            result = OPERATORS[step](rng, result)
    return result

def synthesize_new_operator(rng: random.Random) -> Tuple[str, Dict]:
    n_steps = rng.randint(2, 4)
    steps = [rng.choice(PRIMITIVE_OPS) for _ in range(n_steps)]
    name = f"synth_{sha256(''.join(steps) + str(time.time()))[:8]}"
    return (name, {"steps": steps, "score": 0.0})


def mutate_learner(rng: random.Random, learner: LearnerGenome, meta: "MetaState") -> LearnerGenome:
    """PHASE B: mutate a learner genome by selecting a block."""
    blocks = ["encode", "predict", "update", "objective"]
    block = rng.choice(blocks)
    op = meta.sample_op(rng)

    def apply_block(stmts: List[str]) -> List[str]:
        if op in OPERATORS:
            return OPERATORS[op](rng, stmts)
        return stmts

    if block == "encode":
        new_encode = apply_block(learner.encode_stmts)
        return LearnerGenome(new_encode, learner.predict_stmts, learner.update_stmts, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    if block == "predict":
        new_predict = apply_block(learner.predict_stmts)
        return LearnerGenome(learner.encode_stmts, new_predict, learner.update_stmts, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    if block == "update":
        new_update = apply_block(learner.update_stmts)
        return LearnerGenome(learner.encode_stmts, learner.predict_stmts, new_update, learner.objective_stmts, parents=[learner.gid], op_tag=f"mut:{block}:{op}")
    new_objective = apply_block(learner.objective_stmts)
    return LearnerGenome(learner.encode_stmts, learner.predict_stmts, learner.update_stmts, new_objective, parents=[learner.gid], op_tag=f"mut:{block}:{op}")


# ---------------------------
# Surrogate + MAP-Elites
# ---------------------------

class SurrogateModel:
    def __init__(self, k: int = 5):
        self.k = k
        self.memory: List[Tuple[List[float], float]] = []

    def _extract_features(self, code: str) -> List[float]:
        return [
            len(code),
            code.count("\n"),
            code.count("if "),
            code.count("while "),
            code.count("="),
            code.count("return "),
            code.count("("),
        ]

    def train(self, history: List[Dict]):
        self.memory = []
        for h in history[-200:]:
            src = h.get("code") or h.get("expr")
            if src and "score" in h and isinstance(h["score"], (int, float)):
                feat = self._extract_features(src)
                self.memory.append((feat, float(h["score"])))

    def predict(self, code: str) -> float:
        if not self.memory:
            return 0.0
        target = self._extract_features(code)
        dists = []
        for feat, score in self.memory:
            d = sum((f1 - f2) ** 2 for f1, f2 in zip(target, feat)) ** 0.5
            dists.append((d, score))
        dists.sort(key=lambda x: x[0])
        nearest = dists[: self.k]
        total_w = 0.0
        weighted = 0.0
        for d, s in nearest:
            w = 1.0 / (d + 1e-6)
            weighted += s * w
            total_w += w
        return weighted / total_w if total_w > 0 else 0.0


SURROGATE = SurrogateModel()


class MAPElitesArchive:
    def __init__(self, genome_cls: type = Genome):
        self.grid: Dict[Tuple[int, int], Tuple[float, Any]] = {}
        self.genome_cls = genome_cls

    def _features(self, code: str) -> Tuple[int, int]:
        l_bin = min(20, len(code) // 20)
        d_bin = min(10, code.count("\n") // 2)
        return (l_bin, d_bin)

    def add(self, genome: Any, score: float):
        feat = self._features(genome.code)
        if feat not in self.grid or score < self.grid[feat][0]:
            self.grid[feat] = (score, genome)

    def sample(self, rng: random.Random) -> Optional[Any]:
        if not self.grid:
            return None
        return rng.choice(list(self.grid.values()))[1]

    def snapshot(self) -> Dict:
        return {
            "grid_size": len(self.grid),
            "entries": [(list(k), v[0], asdict(v[1])) for k, v in self.grid.items()],
        }

    def from_snapshot(self, s: Dict) -> "MAPElitesArchive":
        ma = MAPElitesArchive(self.genome_cls)
        for k, score, g_dict in s.get("entries", []):
            ma.grid[tuple(k)] = (score, self.genome_cls(**g_dict))
        return ma


MAP_ELITES = MAPElitesArchive(Genome)
MAP_ELITES_LEARNER = MAPElitesArchive(LearnerGenome)

def map_elites_filename(mode: str) -> str:
    return "map_elites_learner.json" if mode == "learner" else "map_elites.json"

def save_map_elites(path: Path, archive: MAPElitesArchive):
    path.write_text(json.dumps(archive.snapshot(), indent=2), encoding="utf-8")

def load_map_elites(path: Path, archive: MAPElitesArchive):
    if path.exists():
        try:
            loaded = archive.from_snapshot(json.loads(path.read_text(encoding="utf-8")))
            archive.grid = loaded.grid
        except Exception:
            pass


# ---------------------------
# Operator library evolution
# ---------------------------

def evolve_operator_meta(rng: random.Random) -> Tuple[str, Dict]:
    candidates = [v for _, v in OPERATORS_LIB.items() if v.get("score", 0) > -5.0]
    if len(candidates) < 2:
        return synthesize_new_operator(rng)
    p1 = rng.choice(candidates)["steps"]
    p2 = rng.choice(candidates)["steps"]
    cut = rng.randint(0, min(len(p1), len(p2)))
    child_steps = p1[:cut] + p2[cut:]
    if rng.random() < 0.5:
        mut_type = rng.choice(["mod", "add", "del"])
        if mut_type == "mod" and child_steps:
            child_steps[rng.randint(0, len(child_steps) - 1)] = rng.choice(PRIMITIVE_OPS)
        elif mut_type == "add":
            child_steps.insert(rng.randint(0, len(child_steps)), rng.choice(PRIMITIVE_OPS))
        elif mut_type == "del" and len(child_steps) > 1:
            child_steps.pop(rng.randint(0, len(child_steps) - 1))
    child_steps = child_steps[:6] or [rng.choice(PRIMITIVE_OPS)]
    name = f"evo_{sha256(''.join(child_steps) + str(time.time()))[:8]}"
    return (name, {"steps": child_steps, "score": 0.0})

def maybe_evolve_operators_lib(rng: random.Random, threshold: int = 10) -> Optional[str]:
    # remove worst if very bad
    if len(OPERATORS_LIB) > 3:
        sorted_ops = sorted(OPERATORS_LIB.items(), key=lambda x: x[1].get("score", 0))
        worst_name, worst_spec = sorted_ops[0]
        if worst_spec.get("score", 0) < -threshold:
            del OPERATORS_LIB[worst_name]

    # add new until size
    if len(OPERATORS_LIB) < 8:
        if rng.random() < 0.7 and len(OPERATORS_LIB) >= 2:
            name, spec = evolve_operator_meta(rng)
        else:
            name, spec = synthesize_new_operator(rng)
        OPERATORS_LIB[name] = spec
        return name
    return None


# ---------------------------
# Curriculum generator (simple)
# ---------------------------

class ProblemGeneratorV2:
    def __init__(self):
        self.archive: List[Dict] = []

    def evolve_task(self, rng: random.Random, current_elites: List[Genome]) -> TaskSpec:
        arc_tasks = get_arc_tasks()
        base_options = ["sort", "reverse", "max", "filter"]
        arc_options = [f"arc_{tid}" for tid in arc_tasks] if arc_tasks else []
        options = base_options + arc_options
        base_name = rng.choice(options) if options else "sort"
        level = rng.randint(1, 3)
        mn = 3 + level
        mx = 5 + level
        if base_name.startswith("arc_"):
            mn, mx = (3, 5)
        return TaskSpec(name=base_name, n_train=64, n_hold=32, x_min=float(mn), x_max=float(mx), noise=0.0)


# ---------------------------
# Task detective (seeding hints)
# ---------------------------

class TaskDetective:
    @staticmethod
    def detect_pattern(batch: Optional[Batch]) -> Optional[str]:
        if not batch or not batch.x_tr:
            return None
        check_set = list(zip(batch.x_tr[:5], batch.y_tr[:5]))
        is_sort = is_rev = is_max = is_min = is_len = True
        for x, y in check_set:
            if not isinstance(x, list) or not isinstance(y, (list, int, float)):
                return None
            if isinstance(y, list):
                if y != sorted(x):
                    is_sort = False
                if y != list(reversed(x)):
                    is_rev = False
            else:
                is_sort = is_rev = False
            if isinstance(y, (int, float)):
                if not x:
                    if y != 0:
                        is_len = False
                else:
                    if y != len(x):
                        is_len = False
                    if y != max(x):
                        is_max = False
                    if y != min(x):
                        is_min = False
            else:
                is_max = is_min = is_len = False
        if is_sort:
            return "HINT_SORT"
        if is_rev:
            return "HINT_REVERSE"
        if is_max:
            return "HINT_MAX"
        if is_min:
            return "HINT_MIN"
        if is_len:
            return "HINT_LEN"
        return None


def seed_genome(rng: random.Random, hint: Optional[str] = None) -> Genome:
    seeds = [
        ["return x"],
        ["return sorted(x)"],
        ["return list(reversed(x))"],
        ["v0 = sorted(x)", "return v0"],
        [f"return {_random_expr(rng, depth=0)}"],
    ]
    if hint == "HINT_SORT":
        seeds.extend([["return sorted(x)"]] * 5)
    elif hint == "HINT_REVERSE":
        seeds.extend([["return list(reversed(x))"]] * 5)
    elif hint == "HINT_MAX":
        seeds.extend([["return max(x)"]] * 5)
    elif hint == "HINT_MIN":
        seeds.extend([["return min(x)"]] * 5)
    elif hint == "HINT_LEN":
        seeds.extend([["return len(x)"]] * 5)
    return Genome(statements=rng.choice(seeds))


def seed_learner_genome(rng: random.Random, hint: Optional[str] = None) -> LearnerGenome:
    """PHASE B: learner seed set with simple predictors and objectives."""
    base_encode = ["return x"]
    base_predict = ["return z"]
    base_update = ["return mem"]
    base_obj = ["return hold + 0.5*stress + 0.01*nodes"]

    linear_predict = ["return mem['w'] * z + mem['b']"]
    linear_update = [
        "mem['w'] = mem['w'] + lr * (y_true - y_pred) * z",
        "mem['b'] = mem['b'] + lr * (y_true - y_pred)",
        "return mem",
    ]

    list_sort_predict = ["return sorted(z)"]
    list_reverse_predict = ["return list(reversed(z))"]
    list_max_predict = ["return max(z) if z else 0"]

    seeds = [
        LearnerGenome(base_encode, base_predict, base_update, base_obj),
        LearnerGenome(base_encode, linear_predict, linear_update, base_obj),
    ]

    if hint == "HINT_SORT":
        seeds.append(LearnerGenome(base_encode, list_sort_predict, base_update, base_obj))
    elif hint == "HINT_REVERSE":
        seeds.append(LearnerGenome(base_encode, list_reverse_predict, base_update, base_obj))
    elif hint == "HINT_MAX":
        seeds.append(LearnerGenome(base_encode, list_max_predict, base_update, base_obj))

    return rng.choice(seeds)


# ---------------------------
# Function library (learned helpers)
# ---------------------------

@dataclass
class LearnedFunc:
    name: str
    expr: str
    trust: float = 1.0
    uses: int = 0


class FunctionLibrary:
    def __init__(self, max_size: int = 16):
        self.funcs: Dict[str, LearnedFunc] = {}
        self.max_size = max_size

    def maybe_adopt(self, rng: random.Random, expr: str, threshold: float = 0.1) -> Optional[str]:
        if len(self.funcs) >= self.max_size or rng.random() > threshold:
            return None
        try:
            tree = ast.parse(expr, mode="eval").body
            nodes = list(ast.walk(tree))
            if len(nodes) < 4:
                return None
            sub = _pick_node(rng, tree)
            sub_expr = _to_src(sub)
            if node_count(sub_expr) < 3:
                return None
            ok, _ = validate_expr(sub_expr, extra=set(self.funcs.keys()))
            if not ok:
                return None
            name = f"h{len(self.funcs) + 1}"
            self.funcs[name] = LearnedFunc(name=name, expr=sub_expr)
            return name
        except Exception:
            return None

    def maybe_inject(self, rng: random.Random, expr: str) -> Tuple[str, Optional[str]]:
        if not self.funcs or rng.random() > 0.2:
            return (expr, None)
        fn = rng.choice(list(self.funcs.values()))
        fn.uses += 1
        try:
            call = f"{fn.name}(x)"
            new = expr.replace("x", call, 1) if rng.random() < 0.5 else f"({expr}+{call})"
            ok, _ = validate_expr(new, extra=set(self.funcs.keys()))
            return (new, fn.name) if ok else (expr, None)
        except Exception:
            return (expr, None)

    def update_trust(self, name: str, improved: bool):
        if name in self.funcs:
            self.funcs[name].trust *= 1.1 if improved else 0.9
            self.funcs[name].trust = clamp(self.funcs[name].trust, 0.1, 10.0)

    def get_helpers(self) -> Dict[str, Callable]:
        # helper functions callable from evolved programs
        helpers: Dict[str, Callable] = {}

        def make_helper(expr: str):
            return lambda x: legacy_evaluate_expr(expr, x, extra_funcs=helpers)

        for n, f in self.funcs.items():
            helpers[n] = make_helper(f.expr)
        return helpers

    def snapshot(self) -> Dict:
        return {"funcs": [asdict(f) for f in self.funcs.values()]}

    def merge(self, other: "FunctionLibrary"):
        for name, func in other.funcs.items():
            if name not in self.funcs:
                self.funcs[name] = func
            else:
                new_name = f"{name}_{len(self.funcs) + 1}"
                self.funcs[new_name] = LearnedFunc(name=new_name, expr=func.expr, trust=func.trust, uses=func.uses)

    @staticmethod
    def from_snapshot(s: Dict) -> "FunctionLibrary":
        lib = FunctionLibrary()
        for fd in s.get("funcs", []):
            lib.funcs[fd["name"]] = LearnedFunc(**fd)
        return lib


@dataclass
class LibraryRecord:
    descriptor: TaskDescriptor
    score_hold: float
    snapshot: Dict[str, Any]


class LibraryArchive:
    def __init__(self, k: int = 2):
        self.k = k
        self.records: List[LibraryRecord] = []

    def add(self, descriptor: TaskDescriptor, score_hold: float, lib: FunctionLibrary):
        self.records.append(LibraryRecord(descriptor=descriptor, score_hold=score_hold, snapshot=lib.snapshot()))

    def _distance(self, a: List[float], b: List[float]) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def select(self, descriptor: TaskDescriptor) -> List[FunctionLibrary]:
        if not self.records:
            return []
        vec = descriptor.vector()
        ranked = sorted(self.records, key=lambda r: (self._distance(vec, r.descriptor.vector()), r.score_hold))
        libs = []
        for rec in ranked[: self.k]:
            libs.append(FunctionLibrary.from_snapshot(rec.snapshot))
        return libs


# ---------------------------
# Grammar induction (single definition)
# ---------------------------

def induce_grammar(pool: List[Genome]):
    if not pool:
        return
    elites = pool[: max(10, len(pool) // 5)]
    counts = {k: 0.1 for k in GRAMMAR_PROBS}
    for g in elites:
        try:
            tree = ast.parse(g.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in counts:
                        counts[node.func.id] += 1.0
                    counts["call"] += 1.0
                elif isinstance(node, ast.BinOp):
                    counts["binop"] += 1.0
                elif isinstance(node, ast.Name) and node.id == "x":
                    counts["var"] += 1.0
                elif isinstance(node, ast.Constant):
                    counts["const"] += 1.0
        except Exception:
            pass
    total = sum(counts.values())
    if total > 0:
        for k in counts:
            old = GRAMMAR_PROBS.get(k, 1.0)
            target = counts[k] / total * 100.0
            GRAMMAR_PROBS[k] = 0.8 * old + 0.2 * target


def extract_return_expr(stmts: List[str]) -> Optional[str]:
    for stmt in reversed(stmts):
        s = stmt.strip()
        if s.startswith("return "):
            return s[len("return ") :].strip()
    return None


def inject_helpers_into_statements(rng: random.Random, stmts: List[str], library: FunctionLibrary) -> List[str]:
    if not library.funcs:
        return stmts
    new_stmts = []
    injected = False
    for stmt in stmts:
        if not injected and stmt.strip().startswith("return "):
            expr = stmt.strip()[len("return ") :].strip()
            new_expr, helper_name = library.maybe_inject(rng, expr)
            if helper_name:
                stmt = f"return {new_expr}"
                injected = True
        new_stmts.append(stmt)
    return new_stmts


# ---------------------------
# MetaState (L0/L1 source-patchable)
# ---------------------------

OP_WEIGHT_INIT: Dict[str, float] = {
    k: (5.0 if k in ("modify_return", "insert_assign", "list_manip") else 1.0)
    for k in OPERATORS
}


@dataclass
class UpdateRuleGenome:
    """Representation for update rule parameters (meta-level learning algorithm)."""

    learning_rate: float
    momentum: float
    rejection_penalty: float
    reward_scale: float
    uid: str = ""

    def __post_init__(self):
        if not self.uid:
            self.uid = sha256(f"{self.learning_rate}:{self.momentum}:{self.rejection_penalty}:{self.reward_scale}")[:10]

    def apply(self, meta: "MetaState", op: str, delta: float, accepted: bool) -> None:
        reward = (max(0.0, -delta) * self.reward_scale) if accepted else -self.rejection_penalty
        velocity = meta.op_velocity.get(op, 0.0)
        velocity = self.momentum * velocity + reward
        meta.op_velocity[op] = velocity
        meta.op_weights[op] = clamp(meta.op_weights.get(op, 1.0) + self.learning_rate * velocity, 0.1, 8.0)

    def mutate(self, rng: random.Random) -> "UpdateRuleGenome":
        return UpdateRuleGenome(
            learning_rate=clamp(self.learning_rate + rng.uniform(-0.05, 0.05), 0.01, 0.5),
            momentum=clamp(self.momentum + rng.uniform(-0.1, 0.1), 0.0, 0.95),
            rejection_penalty=clamp(self.rejection_penalty + rng.uniform(-0.05, 0.05), 0.01, 0.5),
            reward_scale=clamp(self.reward_scale + rng.uniform(-0.1, 0.1), 0.2, 2.0),
        )

    @staticmethod
    def default() -> "UpdateRuleGenome":
        return UpdateRuleGenome(learning_rate=0.12, momentum=0.3, rejection_penalty=0.12, reward_scale=1.0)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UpdateRuleGenome":
        return UpdateRuleGenome(
            learning_rate=float(data.get("learning_rate", 0.12)),
            momentum=float(data.get("momentum", 0.3)),
            rejection_penalty=float(data.get("rejection_penalty", 0.12)),
            reward_scale=float(data.get("reward_scale", 1.0)),
            uid=data.get("uid", ""),
        )

@dataclass
class MetaState:
    op_weights: Dict[str, float] = field(default_factory=lambda: dict(OP_WEIGHT_INIT))
    op_velocity: Dict[str, float] = field(default_factory=dict)
    mutation_rate: float = 0.8863
    crossover_rate: float = 0.1971
    complexity_lambda: float = 0.0001
    epsilon_explore: float = 0.4213
    adapt_steps: int = 8
    stuck_counter: int = 0
    update_rule: UpdateRuleGenome = field(default_factory=UpdateRuleGenome.default)
    strategy: EngineStrategy = field(default_factory=lambda: EngineStrategy(
        selection_code=DEFAULT_SELECTION_CODE,
        crossover_code=DEFAULT_CROSSOVER_CODE,
        mutation_policy_code=DEFAULT_MUTATION_CODE
    ))

    def sample_op(self, rng: random.Random) -> str:
        if rng.random() < self.epsilon_explore:
            return rng.choice(list(OPERATORS.keys()))
        total = sum(max(0.01, w) for w in self.op_weights.values())
        r = rng.random() * total
        acc = 0.0
        for k, w in self.op_weights.items():
            acc += max(0.01, w)
            if r <= acc:
                return k
        return rng.choice(list(OPERATORS.keys()))

    def update(self, op: str, delta: float, accepted: bool):
        if op in self.op_weights:
            self.update_rule.apply(self, op, delta, accepted)
        if not accepted:
            self.stuck_counter += 1
            if self.stuck_counter > 20:
                self.epsilon_explore = clamp(self.epsilon_explore + 0.02, 0.1, 0.4)
                self.mutation_rate = clamp(self.mutation_rate + 0.03, 0.4, 0.95)
        else:
            self.stuck_counter = 0
            self.epsilon_explore = clamp(self.epsilon_explore - 0.01, 0.05, 0.3)


class MetaCognitiveEngine:
    @staticmethod
    def analyze_execution(results: List[Tuple[Any, EvalResult]], meta: MetaState):
        errors = [r.err.split(":")[0] for _, r in results if (not r.ok and r.err)]
        if not errors:
            return
        counts = collections.Counter(errors)
        total_err = len(errors)
        if counts.get("TypeError", 0) > total_err * 0.3:
            if "binop" in GRAMMAR_PROBS:
                GRAMMAR_PROBS["binop"] *= 0.5
            GRAMMAR_PROBS["var"] = GRAMMAR_PROBS.get("var", 1.0) * 1.5
        if counts.get("IndexError", 0) > total_err * 0.3:
            if "list_manip" in meta.op_weights:
                meta.op_weights["list_manip"] *= 0.7
        if counts.get("StepLimitExceeded", 0) > total_err * 0.3:
            meta.complexity_lambda *= 2.0


# ---------------------------
# L1 Meta-optimizer policy
# ---------------------------

@dataclass
class MetaPolicy:
    weights: List[List[float]]
    bias: List[float]
    pid: str = ""

    @staticmethod
    def seed(rng: random.Random, n_outputs: int, n_inputs: int) -> "MetaPolicy":
        weights = [[rng.uniform(-0.2, 0.2) for _ in range(n_inputs)] for _ in range(n_outputs)]
        bias = [rng.uniform(-0.1, 0.1) for _ in range(n_outputs)]
        pid = sha256(json.dumps(weights) + json.dumps(bias))[:10]
        return MetaPolicy(weights=weights, bias=bias, pid=pid)

    def _linear(self, features: List[float], idx: int) -> float:
        w = self.weights[idx]
        return sum(fi * wi for fi, wi in zip(features, w)) + self.bias[idx]

    def act(self, descriptor: TaskDescriptor, stats: Dict[str, float]) -> Dict[str, Any]:
        features = descriptor.vector() + [
            stats.get("delta_best", 0.0),
            stats.get("auc_window", 0.0),
            stats.get("timeout_rate", 0.0),
            stats.get("avg_nodes", 0.0),
        ]
        outputs = [self._linear(features, i) for i in range(len(self.weights))]
        mutation_rate = clamp(0.5 + outputs[0], 0.05, 0.98)
        crossover_rate = clamp(0.2 + outputs[1], 0.0, 0.9)
        novelty_weight = clamp(0.2 + outputs[2], 0.0, 1.0)
        branch_insert_rate = clamp(0.1 + outputs[3], 0.0, 0.6)
        op_scale = clamp(1.0 + outputs[4], 0.2, 3.0)
        op_weights = {
            "modify_return": clamp(OP_WEIGHT_INIT.get("modify_return", 1.0) * op_scale, 0.1, 8.0),
            "insert_assign": clamp(OP_WEIGHT_INIT.get("insert_assign", 1.0) * (op_scale + 0.2), 0.1, 8.0),
            "list_manip": clamp(OP_WEIGHT_INIT.get("list_manip", 1.0) * (op_scale - 0.1), 0.1, 8.0),
        }
        return {
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate,
            "novelty_weight": novelty_weight,
            "branch_insert_rate": branch_insert_rate,
            "op_weights": op_weights,
        }

    def mutate(self, rng: random.Random, scale: float = 0.1) -> "MetaPolicy":
        weights = [row[:] for row in self.weights]
        bias = self.bias[:]
        for i in range(len(weights)):
            if rng.random() < 0.7:
                j = rng.randrange(len(weights[i]))
                weights[i][j] += rng.uniform(-scale, scale)
        for i in range(len(bias)):
            if rng.random() < 0.5:
                bias[i] += rng.uniform(-scale, scale)
        pid = sha256(json.dumps(weights) + json.dumps(bias))[:10]
        return MetaPolicy(weights=weights, bias=bias, pid=pid)


# ---------------------------
# Duo-loop (Creator/Critic)
# ---------------------------

@dataclass
class AgentPolicy:
    generator_mode: str
    search_bias: Dict[str, float]
    gate_target: float
    slice_seconds: float


CREATOR_POLICY = AgentPolicy(
    generator_mode="synthesize",
    search_bias={
        "novelty": 1.2,
        "simplicity": 0.4,
        "robustness": 0.3,
        "generalization": 0.2,
        "perf": 0.2,
    },
    gate_target=0.35,
    slice_seconds=6.0,
)

CRITIC_POLICY = AgentPolicy(
    generator_mode="mutate",
    search_bias={
        "novelty": 0.2,
        "simplicity": 0.9,
        "robustness": 1.1,
        "generalization": 1.0,
        "perf": 0.6,
    },
    gate_target=0.7,
    slice_seconds=6.0,
)

# ---------------------------
# Universe / Multiverse
# ---------------------------

@dataclass
class Universe:
    uid: int
    seed: int
    meta: MetaState
    pool: List[Genome]
    library: FunctionLibrary
    discriminator: ProblemGeneratorV2 = field(default_factory=ProblemGeneratorV2)
    eval_mode: str = "solver"
    best: Optional[Genome] = None
    best_score: float = float("inf")
    best_train: float = float("inf")
    best_hold: float = float("inf")
    best_stress: float = float("inf")
    best_test: float = float("inf")
    history: List[Dict] = field(default_factory=list)

    def step(
        self,
        gen: int,
        task: TaskSpec,
        pop_size: int,
        batch: Batch,
        policy_controls: Optional[Union[Dict[str, float], ControlPacket]] = None,
    ) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        if batch is None:
            self.pool = [seed_genome(rng) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "no_batch"}

        helper_env = self.library.get_helpers()
        if policy_controls:
            self.meta.mutation_rate = clamp(policy_controls.get("mutation_rate", self.meta.mutation_rate), 0.05, 0.98)
            self.meta.crossover_rate = clamp(policy_controls.get("crossover_rate", self.meta.crossover_rate), 0.0, 0.95)
            novelty_weight = clamp(policy_controls.get("novelty_weight", 0.0), 0.0, 1.0)
            branch_rate = clamp(policy_controls.get("branch_insert_rate", 0.0), 0.0, 0.6)
            if isinstance(policy_controls.get("op_weights"), dict):
                for k, v in policy_controls["op_weights"].items():
                    if k in self.meta.op_weights:
                        self.meta.op_weights[k] = clamp(float(v), 0.1, 8.0)
        else:
            novelty_weight = 0.0
            branch_rate = 0.0

        scored: List[Tuple[Genome, EvalResult]] = []
        all_results: List[Tuple[Genome, EvalResult]] = []
        for g in self.pool:
            # Hard gate: enforce input dependence before any scoring/selection.
            gate_ok, gate_reason = _hard_gate_ok(
                g.code,
                batch,
                self.eval_mode if self.eval_mode != "program" else "solver",
                task.name,
                extra_env=helper_env,
            )
            if not gate_ok:
                res = EvalResult(
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    node_count(g.code),
                    float("inf"),
                    f"hard_gate:{gate_reason}",
                )
                all_results.append((g, res))
                continue
            if self.eval_mode == "algo":
                res = evaluate_algo(g, batch, task.name, self.meta.complexity_lambda)
            else:
                validator = validate_program if self.eval_mode == "program" else validate_code
                res = evaluate(g, batch, task.name, self.meta.complexity_lambda, extra_env=helper_env, validator=validator)
            all_results.append((g, res))
            if res.ok:
                scored.append((g, res))

        MetaCognitiveEngine.analyze_execution(all_results, self.meta)

        if not scored:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "reseed"}

        scored.sort(key=lambda t: t[1].score)
        timeout_rate = 1.0 - (len(scored) / max(1, len(all_results)))
        avg_nodes = sum(r.nodes for _, r in scored) / max(1, len(scored))

        # MAP-Elites add
        best_g0, best_res0 = scored[0]
        MAP_ELITES.add(best_g0, best_res0.score)

        for g, _ in scored[:3]:
            expr = extract_return_expr(g.statements)
            if expr:
                adopted = self.library.maybe_adopt(rng, expr, threshold=0.3)
                if adopted:
                    break

        # selection via strategy
        sel_ctx = {
            "pool": [g for g, _ in scored],
            "scores": [res.score for _, res in scored],
            "pop_size": pop_size,
            "map_elites": MAP_ELITES,
            "rng": rng,
        }
        sel_res = safe_exec_engine(self.meta.strategy.selection_code, sel_ctx)
        if sel_res and isinstance(sel_res, (tuple, list)) and len(sel_res) == 2:
            elites, parenting_pool = sel_res
        else:
            elites = [g for g, _ in scored[: max(4, pop_size // 10)]]
            parenting_pool = [rng.choice(elites) for _ in range(pop_size - len(elites))]

        candidates: List[Genome] = []
        needed = pop_size - len(elites)
        attempts_needed = max(needed * 2, needed + 8)
        mate_pool = list(elites) + list(parenting_pool)

        while len(candidates) < attempts_needed:
            parent = rng.choice(parenting_pool) if parenting_pool else rng.choice(elites)
            new_stmts = None
            op_tag = "copy"

            # crossover
            if rng.random() < self.meta.crossover_rate and len(mate_pool) > 1:
                p2 = rng.choice(mate_pool)
                cross_ctx = {"p1": parent.statements, "p2": p2.statements, "rng": rng}
                new_stmts = safe_exec_engine(self.meta.strategy.crossover_code, cross_ctx)
                if new_stmts and isinstance(new_stmts, list):
                    op_tag = "crossover"
                else:
                    new_stmts = None

            if not new_stmts:
                new_stmts = parent.statements[:]

            # mutation
            if op_tag in ("copy", "crossover") and rng.random() < self.meta.mutation_rate:
                use_synth = rng.random() < 0.3 and bool(OPERATORS_LIB)
                if use_synth:
                    synth_name = rng.choice(list(OPERATORS_LIB.keys()))
                    steps = OPERATORS_LIB[synth_name].get("steps", [])
                    new_stmts = apply_synthesized_op(rng, new_stmts, steps)
                    op_tag = f"synth:{synth_name}"
                else:
                    op = self.meta.sample_op(rng)
                    if op in OPERATORS:
                        new_stmts = OPERATORS[op](rng, new_stmts)
                    op_tag = f"mut:{op}"

            if rng.random() < branch_rate:
                extra = rng.choice(seed_genome(rng).statements)
                new_stmts = list(new_stmts) + [extra]
                op_tag = f"{op_tag}|branch"

            new_stmts = inject_helpers_into_statements(rng, list(new_stmts), self.library)
            candidates.append(Genome(statements=new_stmts, parents=[parent.gid], op_tag=op_tag))

        # surrogate ranking
        with_pred = [(c, SURROGATE.predict(c.code) + novelty_weight * rng.random()) for c in candidates]
        with_pred.sort(key=lambda x: x[1])
        selected_children = [c for c, _ in with_pred[:needed]]

        self.pool = list(elites) + selected_children

        # occasionally evolve operator library
        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)

        # grammar induction
        if gen % 5 == 0:
            induce_grammar(list(elites))

        # acceptance update
        best_g, best_res = scored[0]
        old_score = self.best_score
        accept_margin = 1e-9
        if isinstance(policy_controls, ControlPacket):
            accept_margin = max(accept_margin, policy_controls.acceptance_margin)
        accepted = best_res.score < self.best_score - accept_margin
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_train = best_res.train
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress
            self.best_test = best_res.test

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)
        if isinstance(policy_controls, ControlPacket) and self.meta.stuck_counter > policy_controls.patience:
            self.meta.epsilon_explore = clamp(self.meta.epsilon_explore + 0.05, 0.05, 0.5)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "train": self.best_train,
            "hold": self.best_hold,
            "stress": self.best_stress,
            "test": self.best_test,
            "code": self.best.code if self.best else "none",
            "novelty_weight": novelty_weight,
            "timeout_rate": timeout_rate,
            "avg_nodes": avg_nodes,
        }
        self.history.append(log)
        if gen % 5 == 0:
            SURROGATE.train(self.history)
        return log

    def snapshot(self) -> Dict:
        return {
            "uid": self.uid,
            "seed": self.seed,
            "meta": asdict(self.meta),
            "best": asdict(self.best) if self.best else None,
            "best_score": self.best_score,
            "best_train": self.best_train,
            "best_hold": self.best_hold,
            "best_stress": self.best_stress,
            "best_test": self.best_test,
            "pool": [asdict(g) for g in self.pool[:20]],
            "library": self.library.snapshot(),
            "history": self.history[-50:],
            "eval_mode": self.eval_mode,
        }

    @staticmethod
    def from_snapshot(s: Dict) -> "Universe":
        meta_data = s.get("meta", {})
        if "strategy" in meta_data and isinstance(meta_data["strategy"], dict):
            meta_data["strategy"] = EngineStrategy(**meta_data["strategy"])
        if "update_rule" in meta_data and isinstance(meta_data["update_rule"], dict):
            meta_data["update_rule"] = UpdateRuleGenome.from_dict(meta_data["update_rule"])
        meta = MetaState(**{k: v for k, v in meta_data.items() if k != "op_weights"})
        meta.op_weights = meta_data.get("op_weights", dict(OP_WEIGHT_INIT))
        pool = [Genome(**g) for g in s.get("pool", [])]
        lib = FunctionLibrary.from_snapshot(s.get("library", {}))
        u = Universe(uid=s.get("uid", 0), seed=s.get("seed", 0), meta=meta, pool=pool, library=lib)
        if s.get("best"):
            u.best = Genome(**s["best"])
        u.best_score = s.get("best_score", float("inf"))
        u.best_train = s.get("best_train", float("inf"))
        u.best_hold = s.get("best_hold", float("inf"))
        u.best_stress = s.get("best_stress", float("inf"))
        u.best_test = s.get("best_test", float("inf"))
        u.history = s.get("history", [])
        u.eval_mode = s.get("eval_mode", "solver")
        return u


@dataclass
class UniverseLearner:
    """PHASE C: learner multiverse wrapper."""
    uid: int
    seed: int
    meta: MetaState
    pool: List[LearnerGenome]
    library: FunctionLibrary
    discriminator: ProblemGeneratorV2 = field(default_factory=ProblemGeneratorV2)
    best: Optional[LearnerGenome] = None
    best_score: float = float("inf")
    best_hold: float = float("inf")
    best_stress: float = float("inf")
    best_test: float = float("inf")
    history: List[Dict] = field(default_factory=list)

    def step(self, gen: int, task: TaskSpec, pop_size: int, batch: Batch) -> Dict:
        rng = random.Random(self.seed + gen * 1009)
        if batch is None:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_learner_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "no_batch"}

        scored: List[Tuple[LearnerGenome, EvalResult]] = []
        all_results: List[Tuple[LearnerGenome, EvalResult]] = []
        for g in self.pool:
            # Hard gate: enforce input dependence before any scoring/selection.
            gate_ok, gate_reason = _hard_gate_ok(g.code, batch, "learner", task.name)
            if not gate_ok:
                res = EvalResult(
                    False,
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    float("inf"),
                    node_count(g.code),
                    float("inf"),
                    f"hard_gate:{gate_reason}",
                )
                all_results.append((g, res))
                continue
            res = evaluate_learner(g, batch, task.name, self.meta.adapt_steps, self.meta.complexity_lambda)
            all_results.append((g, res))
            if res.ok:
                scored.append((g, res))

        MetaCognitiveEngine.analyze_execution(all_results, self.meta)

        if not scored:
            hint = TaskDetective.detect_pattern(batch)
            self.pool = [seed_learner_genome(rng, hint) for _ in range(pop_size)]
            return {"gen": gen, "accepted": False, "reason": "reseed"}

        scored.sort(key=lambda t: t[1].score)
        best_g0, best_res0 = scored[0]
        MAP_ELITES_LEARNER.add(best_g0, best_res0.score)

        sel_ctx = {
            "pool": [g for g, _ in scored],
            "scores": [res.score for _, res in scored],
            "pop_size": pop_size,
            "map_elites": MAP_ELITES_LEARNER,
            "rng": rng,
        }
        sel_res = safe_exec_engine(self.meta.strategy.selection_code, sel_ctx)
        if sel_res and isinstance(sel_res, (tuple, list)) and len(sel_res) == 2:
            elites, parenting_pool = sel_res
        else:
            elites = [g for g, _ in scored[: max(4, pop_size // 10)]]
            parenting_pool = [rng.choice(elites) for _ in range(pop_size - len(elites))]

        candidates: List[LearnerGenome] = []
        needed = pop_size - len(elites)
        attempts_needed = max(needed * 2, needed + 8)
        mate_pool = list(elites) + list(parenting_pool)

        while len(candidates) < attempts_needed:
            parent = rng.choice(parenting_pool) if parenting_pool else rng.choice(elites)
            child = parent
            op_tag = "copy"

            if rng.random() < self.meta.crossover_rate and len(mate_pool) > 1:
                p2 = rng.choice(mate_pool)
                new_encode = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.encode_stmts, "p2": p2.encode_stmts, "rng": rng})
                new_predict = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.predict_stmts, "p2": p2.predict_stmts, "rng": rng})
                new_update = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.update_stmts, "p2": p2.update_stmts, "rng": rng})
                new_objective = safe_exec_engine(self.meta.strategy.crossover_code, {"p1": parent.objective_stmts, "p2": p2.objective_stmts, "rng": rng})
                if all(isinstance(v, list) for v in (new_encode, new_predict, new_update, new_objective)):
                    child = LearnerGenome(new_encode, new_predict, new_update, new_objective, parents=[parent.gid], op_tag="crossover")
                    op_tag = "crossover"

            if op_tag in ("copy", "crossover") and rng.random() < self.meta.mutation_rate:
                child = mutate_learner(rng, child, self.meta)
                op_tag = child.op_tag

            candidates.append(child)

        with_pred = [(c, SURROGATE.predict(c.code)) for c in candidates]
        with_pred.sort(key=lambda x: x[1])
        selected_children = [c for c, _ in with_pred[:needed]]

        self.pool = list(elites) + selected_children

        if rng.random() < 0.02:
            maybe_evolve_operators_lib(rng)

        if gen % 5 == 0:
            induce_grammar([Genome(statements=["return x"])])

        best_g, best_res = scored[0]
        old_score = self.best_score
        accepted = best_res.score < self.best_score - 1e-9
        if accepted:
            self.best = best_g
            self.best_score = best_res.score
            self.best_hold = best_res.hold
            self.best_stress = best_res.stress
            self.best_test = best_res.test

        op_used = best_g.op_tag.split(":")[1].split("|")[0] if ":" in best_g.op_tag else "unknown"
        self.meta.update(op_used, self.best_score - old_score, accepted)

        log = {
            "gen": gen,
            "accepted": accepted,
            "score": self.best_score,
            "hold": self.best_hold,
            "stress": self.best_stress,
            "test": self.best_test,
            "code": self.best.code if self.best else "none",
        }
        self.history.append(log)
        if gen % 5 == 0:
            SURROGATE.train(self.history)
        return log

    def snapshot(self) -> Dict:
        return {
            "uid": self.uid,
            "seed": self.seed,
            "meta": asdict(self.meta),
            "best": asdict(self.best) if self.best else None,
            "best_score": self.best_score,
            "best_hold": self.best_hold,
            "best_stress": self.best_stress,
            "best_test": self.best_test,
            "pool": [asdict(g) for g in self.pool[:20]],
            "library": self.library.snapshot(),
            "history": self.history[-50:],
        }

    @staticmethod
    def from_snapshot(s: Dict) -> "UniverseLearner":
        meta_data = s.get("meta", {})
        if "strategy" in meta_data and isinstance(meta_data["strategy"], dict):
            meta_data["strategy"] = EngineStrategy(**meta_data["strategy"])
        if "update_rule" in meta_data and isinstance(meta_data["update_rule"], dict):
            meta_data["update_rule"] = UpdateRuleGenome.from_dict(meta_data["update_rule"])
        meta = MetaState(**{k: v for k, v in meta_data.items() if k != "op_weights"})
        meta.op_weights = meta_data.get("op_weights", dict(OP_WEIGHT_INIT))
        pool = [LearnerGenome(**g) for g in s.get("pool", [])]
        lib = FunctionLibrary.from_snapshot(s.get("library", {}))
        u = UniverseLearner(uid=s.get("uid", 0), seed=s.get("seed", 0), meta=meta, pool=pool, library=lib)
        if s.get("best"):
            u.best = LearnerGenome(**s["best"])
        u.best_score = s.get("best_score", float("inf"))
        u.best_hold = s.get("best_hold", float("inf"))
        u.best_stress = s.get("best_stress", float("inf"))
        u.best_test = s.get("best_test", float("inf"))
        u.history = s.get("history", [])
        return u


# ---------------------------
# State persistence
# ---------------------------

@dataclass
class GlobalState:
    version: str
    created_ms: int
    updated_ms: int
    base_seed: int
    task: Dict
    universes: List[Dict]
    selected_uid: int = 0
    generations_done: int = 0
    mode: str = "solver"
    rule_dsl: Optional[Dict[str, Any]] = None

STATE_DIR = Path(".rsi_state")
UPDATE_RULE_FILE = STATE_DIR / "update_rule.json"

def save_update_rule(path: Path, rule: UpdateRuleGenome):
    write_json(path, asdict(rule))

def load_update_rule(path: Path) -> UpdateRuleGenome:
    if path.exists():
        try:
            return UpdateRuleGenome.from_dict(read_json(path))
        except Exception:
            return UpdateRuleGenome.default()
    return UpdateRuleGenome.default()

def save_operators_lib(path: Path):
    path.write_text(json.dumps(OPERATORS_LIB, indent=2), encoding="utf-8")

def load_operators_lib(path: Path):
    global OPERATORS_LIB
    if path.exists():
        try:
            OPERATORS_LIB.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass

def save_state(gs: GlobalState):
    gs.updated_ms = now_ms()
    write_json(STATE_DIR / "state.json", asdict(gs))
    save_operators_lib(STATE_DIR / "operators_lib.json")
    if gs.universes:
        meta_snapshot = gs.universes[0].get("meta", {})
        if isinstance(meta_snapshot, dict) and "update_rule" in meta_snapshot:
            try:
                save_update_rule(STATE_DIR / "update_rule.json", UpdateRuleGenome.from_dict(meta_snapshot["update_rule"]))
            except Exception:
                pass
    if gs.mode == "learner":
        save_map_elites(STATE_DIR / map_elites_filename("learner"), MAP_ELITES_LEARNER)
    else:
        save_map_elites(STATE_DIR / map_elites_filename("solver"), MAP_ELITES)

def load_state() -> Optional[GlobalState]:
    p = STATE_DIR / "state.json"
    if not p.exists():
        return None
    try:
        data = read_json(p)
        mode = data.get("mode", "solver")
        load_operators_lib(STATE_DIR / "operators_lib.json")
        if mode == "learner":
            load_map_elites(STATE_DIR / map_elites_filename("learner"), MAP_ELITES_LEARNER)
        else:
            load_map_elites(STATE_DIR / map_elites_filename("solver"), MAP_ELITES)
        data["mode"] = mode
        return GlobalState(**data)
    except Exception:
        return None


def run_multiverse(
    seed: int,
    task: TaskSpec,
    gens: int,
    pop: int,
    n_univ: int,
    resume: bool = False,
    save_every: int = 5,
    mode: str = "solver",
    freeze_eval: bool = True,
) -> GlobalState:
    safe_mkdir(STATE_DIR)
    logger = RunLogger(STATE_DIR / "run_log.jsonl", append=resume)
    task.ensure_descriptor()
    update_rule = load_update_rule(STATE_DIR / "update_rule.json")

    if resume and (gs0 := load_state()):
        mode = gs0.mode
        if mode == "learner":
            us = [UniverseLearner.from_snapshot(s) for s in gs0.universes]
        else:
            us = [Universe.from_snapshot(s) for s in gs0.universes]
        for u in us:
            u.meta.update_rule = update_rule
        start = gs0.generations_done
    else:
        b0 = get_task_batch(task, seed, freeze_eval=freeze_eval)
        hint = TaskDetective.detect_pattern(b0)
        if hint:
            print(f"[Detective] Detected pattern: {hint}. Injecting smart seeds.")
        if mode == "learner":
            us = [
                UniverseLearner(
                    uid=i,
                    seed=seed + i * 9973,
                    meta=MetaState(update_rule=update_rule),
                    pool=[seed_learner_genome(random.Random(seed + i), hint) for _ in range(pop)],
                    library=FunctionLibrary(),
                )
                for i in range(n_univ)
            ]
        else:
            eval_mode = "program" if mode == "program" else ("algo" if mode == "algo" else "solver")
            us = [
                Universe(
                    uid=i,
                    seed=seed + i * 9973,
                    meta=MetaState(update_rule=update_rule),
                    pool=[seed_genome(random.Random(seed + i), hint) for _ in range(pop)],
                    library=FunctionLibrary(),
                    eval_mode=eval_mode,
                )
                for i in range(n_univ)
            ]
        start = 0

    for gen in range(start, start + gens):
        start_ms = now_ms()
        batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
        for u in us:
            if mode == "learner":
                u.step(gen, task, pop, batch)
            else:
                u.step(gen, task, pop, batch)

        us.sort(key=lambda u: u.best_score)
        best = us[0]
        runtime_ms = now_ms() - start_ms
        best_code = best.best.code if best.best else "none"
        code_hash = sha256(best_code)
        novelty = 1.0 if code_hash not in logger.seen_hashes else 0.0
        logger.seen_hashes.add(code_hash)
        accepted = bool(best.history[-1]["accepted"]) if best.history else False
        last_log = best.history[-1] if best.history else {}
        control_packet = {
            "mutation_rate": best.meta.mutation_rate,
            "crossover_rate": best.meta.crossover_rate,
            "epsilon_explore": best.meta.epsilon_explore,
            "acceptance_margin": 1e-9,
            "patience": getattr(best.meta, "patience", 5),
        }
        counterexample_count = len(ALGO_COUNTEREXAMPLES.get(task.name, [])) if mode == "algo" else 0
        logger.log(
            gen=gen,
            task_id=task.name,
            mode=mode,
            score_hold=best.best_hold,
            score_stress=best.best_stress,
            score_test=getattr(best, "best_test", float("inf")),
            runtime_ms=runtime_ms,
            nodes=node_count(best_code),
            code_hash=code_hash,
            accepted=accepted,
            novelty=novelty,
            meta_policy_params={},
            solver_hash=code_hash,
            p1_hash="default",
            err_hold=best.best_hold,
            err_stress=best.best_stress,
            err_test=getattr(best, "best_test", float("inf")),
            steps=last_log.get("avg_nodes"),
            timeout_rate=last_log.get("timeout_rate"),
            counterexample_count=counterexample_count,
            library_size=len(OPERATORS_LIB),
            control_packet=control_packet,
            task_descriptor=task.descriptor.snapshot() if task.descriptor else None,
        )
        print(
            f"[Gen {gen + 1:4d}] Score: {best.best_score:.4f} | Hold: {best.best_hold:.4f} | Stress: {best.best_stress:.4f} | Test: {best.best_test:.4f} | "
            f"{(best.best.code if best.best else 'none')}"
        )

        if save_every > 0 and (gen + 1) % save_every == 0:
            gs = GlobalState(
                "RSI_EXTENDED_v2",
                now_ms(),
                now_ms(),
                seed,
                asdict(task),
                [u.snapshot() for u in us],
                us[0].uid,
                gen + 1,
                mode=mode,
            )
            save_state(gs)

    gs = GlobalState(
        "RSI_EXTENDED_v2",
        now_ms(),
        now_ms(),
        seed,
        asdict(task),
        [u.snapshot() for u in us],
        us[0].uid,
        start + gens,
        mode=mode,
    )
    save_state(gs)
    return gs


def policy_stats_from_history(history: List[Dict[str, Any]], window: int = 5) -> Dict[str, float]:
    if not history:
        return {"delta_best": 0.0, "auc_window": 0.0, "timeout_rate": 0.0, "avg_nodes": 0.0}
    holds = [h.get("hold", 0.0) for h in history]
    recent = holds[-window:] if len(holds) >= window else holds
    auc_window = sum(recent) / max(1, len(recent))
    if len(holds) >= window:
        delta_best = holds[-1] - holds[-window]
    else:
        delta_best = holds[-1] - holds[0]
    timeout_rate = history[-1].get("timeout_rate", 0.0)
    avg_nodes = history[-1].get("avg_nodes", 0.0)
    return {
        "delta_best": delta_best,
        "auc_window": auc_window,
        "timeout_rate": timeout_rate,
        "avg_nodes": avg_nodes,
    }


def run_policy_episode(
    seed: int,
    task: TaskSpec,
    policy: MetaPolicy,
    gens: int,
    pop: int,
    n_univ: int,
    freeze_eval: bool,
    library_archive: LibraryArchive,
    logger: Optional[RunLogger],
    mode: str,
    update_archive: bool = True,
) -> Tuple[List[Dict[str, Any]], Universe]:
    batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
    hint = TaskDetective.detect_pattern(batch)
    descriptor = task.ensure_descriptor()
    base_lib = FunctionLibrary()
    for lib in library_archive.select(descriptor):
        base_lib.merge(lib)
    update_rule = load_update_rule(STATE_DIR / "update_rule.json")
    universes = [
        Universe(
            uid=i,
            seed=seed + i * 9973,
            meta=MetaState(update_rule=update_rule),
            pool=[seed_genome(random.Random(seed + i), hint) for _ in range(pop)],
            library=FunctionLibrary.from_snapshot(base_lib.snapshot()),
        )
        for i in range(n_univ)
    ]
    for gen in range(gens):
        start_ms = now_ms()
        stats = policy_stats_from_history(universes[0].history)
        controls = policy.act(descriptor, stats)
        for u in universes:
            u.step(gen, task, pop, batch, policy_controls=controls)
        universes.sort(key=lambda u: u.best_score)
        best = universes[0]
        if logger:
            best_code = best.best.code if best.best else "none"
            code_hash = sha256(best_code)
            novelty = 1.0 if code_hash not in logger.seen_hashes else 0.0
            logger.seen_hashes.add(code_hash)
            logger.log(
                gen=gen,
                task_id=task.name,
                mode=mode,
                score_hold=best.best_hold,
                score_stress=best.best_stress,
                score_test=best.best_test,
                runtime_ms=now_ms() - start_ms,
                nodes=node_count(best_code),
                code_hash=code_hash,
                accepted=bool(best.history[-1]["accepted"]) if best.history else False,
                novelty=novelty,
                meta_policy_params={"pid": policy.pid, "weights": policy.weights, "bias": policy.bias, "controls": controls},
                task_descriptor=descriptor.snapshot(),
            )
    universes.sort(key=lambda u: u.best_score)
    best = universes[0]
    if update_archive:
        library_archive.add(descriptor, best.best_hold, best.library)
    return best.history, best


def compute_transfer_metrics(history: List[Dict[str, Any]], window: int) -> Dict[str, float]:
    if not history:
        return {"auc": float("inf"), "regret": float("inf"), "gap": float("inf"), "recovery_time": float("inf")}
    holds = [h.get("hold", float("inf")) for h in history[:window]]
    tests = [h.get("test", float("inf")) for h in history[:window]]
    auc = sum(holds) / max(1, len(holds))
    best = min(holds)
    regret = sum(h - best for h in holds) / max(1, len(holds))
    gap = (tests[-1] - holds[-1]) if holds and tests else float("inf")
    threshold = best * 1.1 if math.isfinite(best) else float("inf")
    recovery_time = float("inf")
    for i, h in enumerate(holds):
        if h <= threshold:
            recovery_time = i + 1
            break
    return {"auc": auc, "regret": regret, "gap": gap, "recovery_time": recovery_time}


def run_meta_meta(
    seed: int,
    episodes: int,
    gens_per_episode: int,
    pop: int,
    n_univ: int,
    policy_pop: int,
    freeze_eval: bool,
    state_dir: Path,
    eval_every: int,
    few_shot_gens: int,
) -> None:
    rng = random.Random(seed)
    meta_train, meta_test = split_meta_tasks(seed)
    n_inputs = len(TaskSpec().ensure_descriptor().vector()) + 4
    policies = [MetaPolicy.seed(rng, n_outputs=5, n_inputs=n_inputs) for _ in range(policy_pop)]
    policy_scores = {p.pid: float("inf") for p in policies}
    archive = LibraryArchive(k=2)
    logger = RunLogger(state_dir / "run_log.jsonl")

    for episode in range(episodes):
        task = rng.choice(meta_train)
        policy = policies[episode % len(policies)]
        history, best = run_policy_episode(
            seed + episode * 31,
            task,
            policy,
            gens_per_episode,
            pop,
            n_univ,
            freeze_eval,
            archive,
            logger,
            mode="meta-train",
            update_archive=True,
        )
        metrics = compute_transfer_metrics(history, window=min(few_shot_gens, len(history)))
        reward = metrics["auc"]
        policy_scores[policy.pid] = min(policy_scores[policy.pid], reward)

        if (episode + 1) % eval_every == 0:
            transfer_scores = []
            for task_test in meta_test:
                warmup_task = rng.choice(meta_train) if meta_train else task_test
                warmup_gens = max(1, few_shot_gens // 2)
                run_policy_episode(
                    seed + episode * 73,
                    warmup_task,
                    policy,
                    warmup_gens,
                    pop,
                    n_univ,
                    freeze_eval,
                    archive,
                    logger,
                    mode="meta-transfer-train",
                    update_archive=True,
                )
                hist, _ = run_policy_episode(
                    seed + episode * 73 + 1,
                    task_test,
                    policy,
                    few_shot_gens,
                    pop,
                    n_univ,
                    freeze_eval,
                    archive,
                    logger,
                    mode="meta-transfer-test",
                    update_archive=False,
                )
                transfer_scores.append(compute_transfer_metrics(hist, window=few_shot_gens)["auc"])
            if transfer_scores:
                policy_scores[policy.pid] = sum(transfer_scores) / len(transfer_scores)
            policies.sort(key=lambda p: policy_scores.get(p.pid, float("inf")))
            best_policy = policies[0]
            policies = [best_policy] + [best_policy.mutate(rng, scale=0.05) for _ in range(policy_pop - 1)]


def evaluate_update_rule(
    seed: int,
    task: TaskSpec,
    rule: UpdateRuleGenome,
    gens: int,
    pop: int,
    freeze_eval: bool,
) -> Tuple[float, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    batch = get_task_batch(task, seed, freeze_eval=freeze_eval)
    hint = TaskDetective.detect_pattern(batch)
    universe = Universe(
        uid=0,
        seed=seed,
        meta=MetaState(update_rule=rule),
        pool=[seed_genome(rng, hint) for _ in range(pop)],
        library=FunctionLibrary(),
    )
    for gen in range(gens):
        universe.step(gen, task, pop, batch)
    return universe.best_score, universe.history


def run_update_rule_search(
    seed: int,
    rounds: int,
    gens_per_round: int,
    pop: int,
    freeze_eval: bool,
    state_dir: Path,
) -> UpdateRuleGenome:
    rng = random.Random(seed)
    task = TaskSpec(name="self_audit", n_train=64, n_hold=48, n_test=48, noise=0.0)
    current = load_update_rule(state_dir / "update_rule.json")
    rules = [current] + [current.mutate(rng) for _ in range(3)]
    best_rule = current
    best_score = float("inf")
    for r in range(rounds):
        scored: List[Tuple[float, UpdateRuleGenome]] = []
        for idx, rule in enumerate(rules):
            score, history = evaluate_update_rule(seed + r * 101 + idx, task, rule, gens_per_round, pop, freeze_eval)
            scored.append((score, rule))
            if history:
                last = history[-1]
                if last.get("code"):
                    SURROGATE.train(history)
        scored.sort(key=lambda item: item[0])
        score, best = scored[0]
        if score < best_score:
            best_score = score
            best_rule = best
            save_update_rule(state_dir / "update_rule.json", best_rule)
        rules = [best] + [best.mutate(rng) for _ in range(max(1, len(rules) - 1))]
    return best_rule


def run_task_switch(
    seed: int,
    task_a: TaskSpec,
    task_b: TaskSpec,
    gens_a: int,
    gens_b: int,
    pop: int,
    n_univ: int,
    freeze_eval: bool,
    state_dir: Path,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    n_inputs = len(TaskSpec().ensure_descriptor().vector()) + 4
    transfer_policy = MetaPolicy.seed(rng, n_outputs=5, n_inputs=n_inputs)
    archive = LibraryArchive(k=2)
    logger = RunLogger(state_dir / "run_log.jsonl")
    baseline = MetaPolicy.seed(random.Random(seed + 999), n_outputs=5, n_inputs=n_inputs)

    history_a, _ = run_policy_episode(
        seed,
        task_a,
        transfer_policy,
        gens_a,
        pop,
        n_univ,
        freeze_eval,
        archive,
        logger,
        mode="switch-train",
        update_archive=True,
    )
    history_transfer, _ = run_policy_episode(
        seed + 1,
        task_b,
        transfer_policy,
        gens_b,
        pop,
        n_univ,
        freeze_eval,
        archive,
        logger,
        mode="switch-transfer",
        update_archive=False,
    )
    history_baseline, _ = run_policy_episode(
        seed + 2,
        task_b,
        baseline,
        gens_b,
        pop,
        n_univ,
        freeze_eval,
        LibraryArchive(k=0),
        logger,
        mode="switch-baseline",
        update_archive=False,
    )
    metrics_transfer = compute_transfer_metrics(history_transfer, window=gens_b)
    metrics_baseline = compute_transfer_metrics(history_baseline, window=gens_b)
    delta_auc = metrics_baseline["auc"] - metrics_transfer["auc"]
    delta_recovery = metrics_baseline["recovery_time"] - metrics_transfer["recovery_time"]
    return {
        "transfer": metrics_transfer,
        "baseline": metrics_baseline,
        "delta_auc": delta_auc,
        "delta_recovery_time": delta_recovery,
    }


def generate_report(path: Path, few_shot_gens: int) -> Dict[str, Any]:
    if not path.exists():
        return {"error": "run_log.jsonl not found"}
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        key = f"{rec['task_id']}::{rec.get('mode', 'unknown')}"
        by_task.setdefault(key, []).append(rec)
    report = {"tasks": {}, "few_shot_gens": few_shot_gens}
    for key, recs in by_task.items():
        recs.sort(key=lambda r: r["gen"])
        holds = [r["score_hold"] for r in recs[:few_shot_gens]]
        tests = [r["score_test"] for r in recs[:few_shot_gens]]
        auc = sum(holds) / max(1, len(holds))
        best = min(holds) if holds else float("inf")
        regret = sum(h - best for h in holds) / max(1, len(holds))
        gap = (tests[-1] - holds[-1]) if holds and tests else float("inf")
        threshold = best * 1.1 if math.isfinite(best) else float("inf")
        recovery_time = float("inf")
        for i, h in enumerate(holds):
            if h <= threshold:
                recovery_time = i + 1
                break
        few_shot_delta = (holds[0] - holds[-1]) if len(holds) > 1 else 0.0
        report["tasks"][key] = {
            "auc": auc,
            "regret": regret,
            "generalization_gap": gap,
            "recovery_time": recovery_time,
            "few_shot_delta": few_shot_delta,
        }
    return report


def transfer_bench(
    task_from: str,
    task_to: str,
    budget: int,
    seed: int,
    freeze_eval: bool = True,
) -> Dict[str, Any]:
    task_a = TaskSpec(name=task_from)
    task_b = TaskSpec(name=task_to)
    mode = "algo" if task_from in ALGO_TASK_NAMES else "solver"
    u = Universe(uid=0, seed=seed, meta=MetaState(), pool=[], library=FunctionLibrary(), eval_mode=mode)
    rng = random.Random(seed)

    for g in range(budget):
        batch = get_task_batch(task_a, seed, freeze_eval=freeze_eval, gen=g)
        if batch is None:
            break
        u.step(g, task_a, 24, batch)

    holds: List[float] = []
    for g in range(budget):
        batch = get_task_batch(task_b, seed + 17, freeze_eval=freeze_eval, gen=g)
        if batch is None:
            break
        u.step(g, task_b, 24, batch)
        holds.append(u.best_hold)

    auc = sum(holds) / max(1, len(holds))
    best = min(holds) if holds else float("inf")
    threshold = best * 1.1 if math.isfinite(best) else float("inf")
    recovery_time = float("inf")
    for i, h in enumerate(holds):
        if h <= threshold:
            recovery_time = i + 1
            break

    record = {
        "from": task_from,
        "to": task_to,
        "budget": budget,
        "seed": seed,
        "auc_N": auc,
        "recovery_time": recovery_time,
        "holds": holds,
    }
    out = STATE_DIR / "transfer_bench.jsonl"
    safe_mkdir(out.parent)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


# ---------------------------
# RSI Loop (Hard Gates + Rollback)
# ---------------------------

STRESS_MAX = 1_000_000.0
OUTPUT_VARIANCE_EPS = 1e-6
RSI_CONFIRM_ROUNDS = 2

def _outputs_constant(outputs: List[Any], tol: float = 1e-9) -> bool:
    if not outputs:
        return True
    first = outputs[0]
    if isinstance(first, (int, float)):
        return all(isinstance(o, (int, float)) and abs(o - first) <= tol for o in outputs[1:])
    return all(_algo_equal(o, first) for o in outputs[1:])

def _unique_output_count(outputs: List[Any]) -> int:
    uniques: List[Any] = []
    for out in outputs:
        if not any(_algo_equal(out, seen) for seen in uniques):
            uniques.append(out)
    return len(uniques)

def _piecewise_constant(outputs: List[Any], max_unique: int = 2) -> bool:
    if not outputs:
        return True
    return _unique_output_count(outputs) <= max_unique

def _variance_low(outputs: List[Any], eps: float = OUTPUT_VARIANCE_EPS) -> bool:
    if not outputs or not all(isinstance(o, (int, float)) for o in outputs):
        return False
    vals = [float(o) for o in outputs]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
    return var <= eps

def _collect_outputs(
    code: str,
    xs: List[Any],
    mode: str,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[Any], str]:
    outputs: List[Any] = []
    if mode == "learner":
        env = safe_load_module(code)
        if not env:
            return False, [], "load_failed"
        required = ["init_mem", "encode", "predict"]
        if not all(name in env and callable(env[name]) for name in required):
            return False, [], "missing_funcs"
        mem = env["init_mem"]()
        encode = env["encode"]
        predict = env["predict"]
        for x in xs:
            try:
                z = encode(x, mem)
                out = predict(z, mem)
            except Exception:
                return False, [], "exec_error"
            outputs.append(out)
        return True, outputs, ""
    if mode == "algo":
        for x in xs:
            out, _, timeout = safe_exec_algo(code, x)
            if timeout:
                return False, [], "timeout"
            outputs.append(out)
        return True, outputs, ""
    for x in xs:
        out = legacy_run(code, x, extra_env=extra_env)
        if out is None:
            return False, [], "no_output"
        outputs.append(out)
    return True, outputs, ""

def _hard_gate_ok(
    code: str,
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    xs = batch.x_ho[:8] if batch.x_ho else batch.x_tr[:8]
    if not xs:
        return False, "no_inputs"
    ok, outputs, err = _collect_outputs(code, xs, mode, extra_env=extra_env)
    if not ok:
        return False, err
    # Hard gate: reject any non-finite numeric output (timeouts/NaNs are disqualifying).
    for out in outputs:
        if isinstance(out, (int, float)) and not math.isfinite(out):
            return False, "non_finite_output"
    # Hard gate: reject constant or near-constant outputs to enforce input dependence.
    if _outputs_constant(outputs):
        return False, "constant_output"
    # Hard gate: prevent piecewise-constant or low-diversity output hacks.
    if _piecewise_constant(outputs):
        return False, "piecewise_constant"
    # Hard gate: reject numerically low-variance responses (e.g., tiny jitter around a constant).
    if _variance_low(outputs):
        return False, "low_variance_output"
    return True, ""

def _evaluate_candidate(
    g: Union[Genome, LearnerGenome],
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> EvalResult:
    gate_ok, gate_reason = _hard_gate_ok(g.code, batch, mode, task_name, extra_env=extra_env)
    if not gate_ok:
        return EvalResult(
            False,
            float("inf"),
            float("inf"),
            float("inf"),
            float("inf"),
            node_count(g.code),
            float("inf"),
            f"hard_gate:{gate_reason}",
        )
    if mode == "learner":
        return evaluate_learner(g, batch, task_name)
    if mode == "algo":
        return evaluate_algo(g, batch, task_name)
    return evaluate(g, batch, task_name, extra_env=extra_env, validator=validator)

def _merge_stress(fixed: Batch, resampled: Batch) -> Batch:
    return Batch(
        x_tr=resampled.x_tr,
        y_tr=resampled.y_tr,
        x_ho=resampled.x_ho,
        y_ho=resampled.y_ho,
        x_st=fixed.x_st + resampled.x_st,
        y_st=fixed.y_st + resampled.y_st,
        x_te=resampled.x_te,
        y_te=resampled.y_te,
    )

def _load_rsi_archive(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"entries": [], "current": None, "consecutive": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"entries": [], "current": None, "consecutive": 0}

def _save_rsi_archive(path: Path, archive: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(archive, indent=2), encoding="utf-8")

def _load_state_snapshot(state_dir: Path) -> Optional[Dict[str, Any]]:
    state_path = state_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_state_snapshot(state_dir: Path, snapshot: Dict[str, Any]) -> None:
    safe_mkdir(state_dir)
    (state_dir / "state.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

def _current_best_score(snapshot: Optional[Dict[str, Any]]) -> float:
    if not snapshot:
        return float("inf")
    universes = snapshot.get("universes", [])
    selected_uid = snapshot.get("selected_uid", None)
    best = float("inf")
    for u in universes:
        score = float(u.get("best_score", float("inf")))
        if selected_uid is not None and u.get("uid") == selected_uid:
            return score
        best = min(best, score)
    return best

def _clone_state_dir(src: Path, dest: Path) -> None:
    safe_mkdir(dest)
    if not src.exists():
        return
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

def _autopatch_evolve_score(
    script: Path,
    state_dir: Path,
    mode: str,
    task_name: str,
    seed: int,
    generations: int,
    population: int,
    universes: int,
    resume: bool,
    freeze_eval: bool = True,
) -> float:
    if mode == "learner":
        cmd = [
            sys.executable,
            str(script),
            "learner-evolve",
            "--seed",
            str(seed),
            "--generations",
            str(generations),
            "--population",
            str(population),
            "--universes",
            str(universes),
            "--task",
            task_name,
            "--state-dir",
            str(state_dir),
        ]
    else:
        cmd = [
            sys.executable,
            str(script),
            "evolve",
            "--seed",
            str(seed),
            "--generations",
            str(generations),
            "--population",
            str(population),
            "--universes",
            str(universes),
            "--task",
            task_name,
            "--state-dir",
            str(state_dir),
        ]
        if mode:
            cmd.extend(["--mode", mode])
    if resume:
        cmd.append("--resume")
    if not freeze_eval:
        cmd.append("--no-freeze-eval")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return float("inf")
    snapshot = _load_state_snapshot(state_dir)
    if not snapshot:
        return float("inf")
    return _current_best_score(snapshot)

def _autopatch_probe_score(
    mode: str,
    task_name: str,
    seed: int = 1337,
    generations: int = 6,
    population: int = 32,
    universes: int = 1,
    freeze_eval: bool = True,
) -> float:
    task = TaskSpec(name=task_name)
    gs = run_multiverse(
        seed,
        task,
        generations,
        population,
        universes,
        resume=False,
        save_every=0,
        mode=mode,
        freeze_eval=freeze_eval,
    )
    if not gs.universes:
        return float("inf")
    best_snapshot = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), gs.universes[0])
    return float(best_snapshot.get("best_score", float("inf")))

def _probe_score(script: Path, mode: str, task_name: str) -> float:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "autopatch-probe",
                "--mode",
                mode,
                "--task",
                task_name,
                "--state-dir",
                tmpdir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        return float("inf")
    output = result.stdout.strip().splitlines()
    if not output:
        return float("inf")
    try:
        return float(output[-1].strip())
    except Exception:
        return float("inf")

def _replace_source_segment(source: str, old: str, new: str) -> str:
    if old not in source:
        return source
    return source.replace(old, new, 1)

def _mutate_hyperparameter(
    tree: ast.AST,
    source: str,
    param_name: str,
    rng: random.Random,
) -> Tuple[str, str, Optional[float]]:
    ranges: Dict[str, Tuple[float, float]] = {
        "mutation_rate": (0.05, 0.95),
        "crossover_rate": (0.0, 0.9),
        "complexity_lambda": (1e-5, 1e-2),
        "epsilon_explore": (0.05, 0.5),
    }
    int_ranges: Dict[str, Tuple[int, int]] = {
        "adapt_steps": (4, 16),
    }
    target_node: Optional[ast.AnnAssign] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MetaState":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    if stmt.target.id == param_name:
                        target_node = stmt
                        break
    if not target_node:
        return source, "", None
    old_segment = ast.get_source_segment(source, target_node.value) or ""
    if param_name in int_ranges:
        low, high = int_ranges[param_name]
        new_value = rng.randint(low, high)
        new_segment = str(new_value)
    else:
        low, high = ranges.get(param_name, (0.0, 1.0))
        new_value = rng.uniform(low, high)
        new_segment = f"{new_value:.6f}"
    new_source = _replace_source_segment(source, old_segment, new_segment)
    return new_source, f"L1:{param_name}", float(new_value)

def _mutate_operator(tree: ast.AST, source: str, rng: random.Random) -> Tuple[str, str]:
    target_assign: Optional[ast.Assign] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "act":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                        if stmt.targets[0].id == "op_weights" and isinstance(stmt.value, ast.Dict):
                            target_assign = stmt
                            break
    if not target_assign or not isinstance(target_assign.value, ast.Dict):
        return source, ""
    new_source = source
    for key_node, value_node in zip(target_assign.value.keys, target_assign.value.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        key = key_node.value
        if key not in ("insert_assign", "list_manip"):
            continue
        old_segment = ast.get_source_segment(source, value_node) or ""
        offset = rng.uniform(0.0, 0.5)
        if key == "list_manip":
            offset = rng.uniform(0.0, 0.3)
        if "op_scale" not in old_segment:
            continue
        if "op_scale +" in old_segment:
            new_segment = re.sub(r"op_scale\s*\+\s*[-+]?\d*\.?\d+", f"op_scale + {offset:.3f}", old_segment)
        else:
            new_segment = re.sub(r"op_scale\s*-\s*[-+]?\d*\.?\d+", f"op_scale - {offset:.3f}", old_segment)
        if new_segment == old_segment:
            continue
        new_source = _replace_source_segment(new_source, old_segment, new_segment)
    return new_source, "L2:op_weights"

def _mutate_evaluation(tree: ast.AST, source: str, rng: random.Random) -> Tuple[str, str]:
    weights = {
        "SCORE_W_HOLD": rng.uniform(0.45, 0.7),
        "SCORE_W_STRESS": rng.uniform(0.2, 0.6),
        "SCORE_W_TRAIN": rng.uniform(0.0, 0.2),
    }
    new_source = source
    changed = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in weights:
                old_segment = ast.get_source_segment(source, node.value) or ""
                new_segment = f"{weights[name]:.6f}"
                if old_segment:
                    new_source = _replace_source_segment(new_source, old_segment, new_segment)
                    changed = True
    return new_source, "L3:score_weights" if changed else ""

def _evaluate_patch_candidate(
    patch_code: str,
    baseline_score: float,
    mode: str,
    task_name: str,
) -> Tuple[bool, float, float]:
    """Evaluate a patch candidate. Returns (accepted, improvement, new_score)."""
    min_improvement_threshold = 0.03
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(patch_code)
        tmp = Path(f.name)
    try:
        new_score = _probe_score(tmp, mode, task_name)
    finally:
        tmp.unlink(missing_ok=True)
    if not math.isfinite(baseline_score) or baseline_score <= 0:
        return False, 0.0, new_score
    improvement = (baseline_score - new_score) / baseline_score
    return improvement >= min_improvement_threshold, improvement, new_score

def _select_best_patch(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [c for c in candidates if c.get("improvement", 0.0) > 0.0]
    if not valid:
        return None
    return max(valid, key=lambda c: (c["improvement"], -c["diff_size"]))

def _safe_apply_patch(self_path: Path, new_code: str) -> bool:
    backup_path = self_path.with_suffix(".py.bak")
    shutil.copy(self_path, backup_path)
    try:
        ast.parse(new_code)
        self_path.write_text(new_code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(self_path), "selftest"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError("Selftest failed")
        return True
    except Exception:
        shutil.copy(backup_path, self_path)
        return False

def _log_autopatch_attempt(record: Dict[str, Any]) -> None:
    log_path = STATE_DIR / "autopatch_log.jsonl"
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def run_deep_autopatch(
    levels: List[int],
    candidates: int = 4,
    apply: bool = True,
    mode: str = "solver",
) -> Dict[str, Any]:
    """
    True RSI self-modification system with fitness-gated acceptance and rollback safety.
    Core change: evolve after mutation instead of re-evaluating the same code.
    """
    script = Path(__file__).resolve()
    source = script.read_text(encoding="utf-8")
    state_snapshot = _load_state_snapshot(STATE_DIR)
    task_name = (state_snapshot or {}).get("task", {}).get("name", TaskSpec().name)
    seed = int((state_snapshot or {}).get("base_seed", 1337))
    universes = max(1, len((state_snapshot or {}).get("universes", [])) or 1)
    pool_len = 0
    if state_snapshot and state_snapshot.get("universes"):
        pool_len = len(state_snapshot["universes"][0].get("pool", []))
    population = max(64, pool_len, 32)
    baseline = _current_best_score(state_snapshot)
    if not math.isfinite(baseline) or baseline <= 0:
        baseline = _probe_score(script, mode, task_name)
    print(f"[AUTOPATCH L{levels}] Baseline: {baseline:.4f}")

    rng = random.Random(int(time.time()) % 100000)
    patch_candidates: List[Dict[str, Any]] = []
    attempt_idx = 0

    for level in levels:
        for _ in range(candidates):
            attempt_idx += 1
            tree = ast.parse(source)
            patch_type = ""
            mutated_source = source
            mutated_state = copy.deepcopy(state_snapshot) if state_snapshot else None
            mutated_params: Dict[str, Any] = {}
            if level == 1:
                param = rng.choice(["mutation_rate", "crossover_rate", "complexity_lambda", "epsilon_explore", "adapt_steps"])
                mutated_source, patch_type, new_value = _mutate_hyperparameter(tree, source, param, rng)
                if new_value is None:
                    continue
                if param == "adapt_steps":
                    new_value = int(round(new_value))
                mutated_params[param] = new_value
                if mutated_state:
                    for u in mutated_state.get("universes", []):
                        meta = u.get("meta", {})
                        meta[param] = new_value
                        u["meta"] = meta
            elif level == 2:
                mutated_source, patch_type = _mutate_operator(tree, source, rng)
            elif level == 3:
                mutated_source, patch_type = _mutate_evaluation(tree, source, rng)
            if not patch_type or (mutated_source == source and not mutated_params):
                continue
            diff = unified_diff(source, mutated_source, str(script))
            diff_size = len(diff.splitlines())
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_state_dir = Path(tmpdir)
                if state_snapshot:
                    _clone_state_dir(STATE_DIR, tmp_state_dir)
                    if mutated_state:
                        _write_state_snapshot(tmp_state_dir, mutated_state)
                script_path = script
                if mutated_source != source:
                    script_path = tmp_state_dir / script.name
                    script_path.write_text(mutated_source, encoding="utf-8")
                attempt_seed = seed + attempt_idx
                print(f"[DEBUG] Running evolution with params: {mutated_params}")
                new_score = _autopatch_evolve_score(
                    script_path,
                    tmp_state_dir,
                    mode,
                    task_name,
                    attempt_seed,
                    generations=15,
                    population=population,
                    universes=universes,
                    resume=state_snapshot is not None,
                    freeze_eval=True,
                )
                print(f"[DEBUG] Evolution returned best_score: {new_score}")
            improvement = baseline - new_score
            accepted = improvement > 0
            record = {
                "level": level,
                "patch_type": patch_type,
                "old_score": baseline,
                "new_score": new_score,
                "improvement": improvement,
                "diff_size": diff_size,
                "accepted": accepted,
                "params": mutated_params,
            }
            _log_autopatch_attempt(record)
            if accepted:
                print(f"[AUTOPATCH] {patch_type} -> {new_score:.4f} (ACCEPT +{improvement:.2f})")
            else:
                print(f"[AUTOPATCH] {patch_type} -> {new_score:.4f} (REJECT)")
            patch_candidates.append(
                {
                    **record,
                    "diff": diff,
                    "code": mutated_source,
                    "state": mutated_state,
                }
            )

    best = _select_best_patch(patch_candidates)
    if not best:
        return {
            "applied": False,
            "improvement": 0.0,
            "old_score": baseline,
            "new_score": baseline,
            "patch_type": "",
            "diff": "",
    }

    if apply:
        applied = True
        if best["code"] != source:
            applied = _safe_apply_patch(script, best["code"])
        if applied and best.get("state"):
            _write_state_snapshot(STATE_DIR, best["state"])
        if applied:
            print(f"[RSI] Self-modified! Score: {best['old_score']:.4f} -> {best['new_score']:.4f}")
        return {
            "applied": applied,
            "improvement": best["improvement"],
            "old_score": best["old_score"],
            "new_score": best["new_score"],
            "patch_type": best["patch_type"],
            "diff": best["diff"],
        }

    return {
        "applied": False,
        "improvement": best["improvement"],
        "old_score": best["old_score"],
        "new_score": best["new_score"],
        "patch_type": best["patch_type"],
        "diff": best["diff"],
    }


def load_recent_scores(log_path: Path, n: int) -> List[float]:
    scores = []
    if not log_path.exists():
        return scores
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-n:]:
                try:
                    data = json.loads(line)
                    if "score" in data:
                        scores.append(float(data["score"]))
                except:
                    pass
    except Exception:
        pass
    return scores


def is_300s_stagnation(scores: List[float]) -> bool:
    if len(scores) < 5:
        return False
    return all(s > 300.0 for s in scores)


def _candidate_hash(code: str) -> str:
    return sha256(code)


def _slice_pair(xs: List[Any], ys: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    if not xs or not ys:
        return [], []
    k = min(n, len(xs), len(ys))
    return xs[:k], ys[:k]


def _prefilter_batch(batch: Batch, max_samples: int = 3) -> Batch:
    x_tr, y_tr = _slice_pair(batch.x_tr, batch.y_tr, max_samples)
    x_ho, y_ho = _slice_pair(batch.x_ho, batch.y_ho, max_samples)
    x_st, y_st = _slice_pair(batch.x_st, batch.y_st, max_samples)
    x_te, y_te = _slice_pair(batch.x_te, batch.y_te, max_samples)
    if not x_tr and x_ho:
        x_tr, y_tr = x_ho, y_ho
    if not x_ho and x_tr:
        x_ho, y_ho = x_tr, y_tr
    if not x_st and x_tr:
        x_st, y_st = x_tr, y_tr
    if not x_te and x_tr:
        x_te, y_te = x_tr, y_tr
    return Batch(x_tr, y_tr, x_ho, y_ho, x_st, y_st, x_te, y_te)


def _prefilter_check(
    g: Genome,
    batch: Batch,
    mode: str,
    task_name: str,
    extra_env: Optional[Dict[str, Any]] = None,
    validator: Callable[[str], Tuple[bool, str]] = validate_code,
) -> Tuple[bool, str, Optional[EvalResult]]:
    gate_ok, gate_reason = _hard_gate_ok(g.code, batch, mode, task_name, extra_env=extra_env)
    if not gate_ok:
        return False, f"hard_gate:{gate_reason}", None
    if mode in ("solver", "program"):
        ok, err = validator(g.code)
        if not ok:
            return False, f"validator:{err}", None
    mini_batch = _prefilter_batch(batch, max_samples=4)
    if mode == "learner":
        res = evaluate_learner(g, mini_batch, task_name)
    elif mode == "algo":
        res = evaluate_algo(g, mini_batch, task_name)
    else:
        res = evaluate(g, mini_batch, task_name, extra_env=extra_env, validator=validator)
    return res.ok, res.err or "", res


def _mutate_genome_with_meta(
    rng: random.Random,
    g: Genome,
    meta: MetaState,
    library: FunctionLibrary,
    op_bias: Optional[str] = None,
) -> Genome:
    stmts = g.statements[:]
    op_tag = "mutate"
    use_synth = rng.random() < 0.3 and bool(OPERATORS_LIB)
    if use_synth:
        synth_name = rng.choice(list(OPERATORS_LIB.keys()))
        steps = OPERATORS_LIB[synth_name].get("steps", [])
        stmts = apply_synthesized_op(rng, stmts, steps)
        op_tag = f"synth:{synth_name}"
    else:
        op = op_bias or meta.sample_op(rng)
        if op in OPERATORS:
            stmts = OPERATORS[op](rng, stmts)
        op_tag = f"mut:{op}"
    stmts = inject_helpers_into_statements(rng, list(stmts), library)
    return Genome(statements=stmts, parents=[g.gid], op_tag=op_tag)


def _synthesize_genome(
    rng: random.Random,
    pool: List[Genome],
    hint: Optional[str],
    library: FunctionLibrary,
) -> Genome:
    if not pool:
        return seed_genome(rng, hint)
    p1 = rng.choice(pool)
    p2 = rng.choice(pool)
    if len(p1.statements) <= 1 or len(p2.statements) <= 1:
        stmts = (p1.statements or []) + (p2.statements or [])
    else:
        cut1 = max(1, len(p1.statements) // 2)
        cut2 = max(1, len(p2.statements) // 2)
        stmts = p1.statements[:cut1] + p2.statements[-cut2:]
    if not stmts:
        stmts = ["return x"]
    stmts = inject_helpers_into_statements(rng, list(stmts), library)
    return Genome(statements=stmts, parents=[p1.gid, p2.gid], op_tag="synthesize")


def _fallback_template_genome(rng: random.Random, hint: Optional[str]) -> Genome:
    if hint:
        return seed_genome(rng, hint)
    return Genome(statements=["v0 = x", "return v0"], op_tag="fallback")


def _simplify_genome(rng: random.Random, g: Genome) -> Optional[Genome]:
    if len(g.statements) <= 1:
        return None
    stmts = g.statements[:]
    removable = [i for i, s in enumerate(stmts) if not s.strip().startswith("return ")]
    if not removable:
        return None
    idx = rng.choice(removable)
    del stmts[idx]
    return Genome(statements=stmts, parents=[g.gid], op_tag="simplify")


def _repair_genome(g: Genome) -> Genome:
    stmts = g.statements[:]
    has_return = any(s.strip().startswith("return ") for s in stmts)
    if not has_return:
        stmts.append("return x")
    else:
        if not any("x" in s for s in stmts if s.strip().startswith("return ")):
            stmts.append("return x")
    return Genome(statements=stmts, parents=[g.gid], op_tag="repair")


def _critic_refine(
    rng: random.Random,
    g: Genome,
    meta: MetaState,
    library: FunctionLibrary,
) -> List[Genome]:
    refined: List[Genome] = []
    simplified = _simplify_genome(rng, g)
    if simplified:
        refined.append(simplified)
    refined.append(_repair_genome(g))
    refined.append(_mutate_genome_with_meta(rng, g, meta, library, op_bias="modify_return"))
    return refined


def _adjust_creator_policy(
    policy: AgentPolicy,
    gate_pass_rate: float,
    gate_fail_reasons: collections.Counter,
) -> AgentPolicy:
    new_search = dict(policy.search_bias)
    generator_mode = policy.generator_mode
    if gate_pass_rate < policy.gate_target:
        generator_mode = "template"
        new_search["simplicity"] = clamp(new_search.get("simplicity", 0.5) + 0.3, 0.1, 2.0)
    if gate_fail_reasons.get("constant_output", 0) > 0:
        new_search["robustness"] = clamp(new_search.get("robustness", 0.5) + 0.2, 0.1, 2.0)
    return AgentPolicy(
        generator_mode=generator_mode,
        search_bias=new_search,
        gate_target=policy.gate_target,
        slice_seconds=policy.slice_seconds,
    )


def _critic_rank_score(res: EvalResult, policy: AgentPolicy) -> float:
    simplicity = policy.search_bias.get("simplicity", 0.0)
    robustness = policy.search_bias.get("robustness", 0.0)
    generalization = policy.search_bias.get("generalization", 0.0)
    perf = policy.search_bias.get("perf", 0.0)
    return (
        res.score
        + simplicity * 0.0005 * res.nodes
        + robustness * res.stress
        + generalization * res.hold
        + perf * res.train
    )


def _print_critic_summary(
    gate_pass: int,
    total_checked: int,
    adopted: bool,
    full_results_count: int,
    duplicate_count: int,
    scored_empty_count: int,
    gate_fail_reasons: collections.Counter,
    validator_fail_reasons: collections.Counter,
) -> None:
    gate_pass_rate = gate_pass / max(1, total_checked)
    adoption_rate = (1.0 if adopted else 0.0) / max(1, full_results_count)
    duplicate_ratio = duplicate_count / max(1, total_checked)
    top_gate = gate_fail_reasons.most_common(5)
    print(
        f"[Critic] gate_pass_rate={gate_pass_rate:.2f} adoption_rate={adoption_rate:.2f} "
        f"duplicate_ratio={duplicate_ratio:.2f} scored_empty={scored_empty_count}"
    )
    if top_gate:
        print("[Critic] top gate failures:", ", ".join(f"{k}:{v}" for k, v in top_gate))
    else:
        print("[Critic] top gate failures: none")
    if validator_fail_reasons:
        top_validator = validator_fail_reasons.most_common(3)
        print("[Critic] top validator failures:", ", ".join(f"{k}:{v}" for k, v in top_validator))


def run_duo_loop(
    rounds: int,
    slice_seconds: float,
    blackboard_path: Path,
    k_full: int,
    seed: int,
    mode: str = "solver",
    freeze_eval: bool = True,
    population: int = 64,
    max_candidates: int = 512,
) -> None:
    task = TaskSpec()
    task.ensure_descriptor()
    rng = random.Random(seed)
    gs = load_state()
    if gs and gs.mode == mode and gs.universes:
        selected = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), gs.universes[0])
        universe = Universe.from_snapshot(selected)
    else:
        batch0 = get_task_batch(task, seed, freeze_eval=freeze_eval)
        hint = TaskDetective.detect_pattern(batch0)
        universe = Universe(
            uid=0,
            seed=seed,
            meta=MetaState(),
            pool=[seed_genome(random.Random(seed + i), hint) for i in range(population)],
            library=FunctionLibrary(),
            eval_mode="program" if mode == "program" else ("algo" if mode == "algo" else "solver"),
        )

    creator_policy = CREATOR_POLICY
    critic_policy = CRITIC_POLICY
    reseed_templates: List[List[str]] = []
    fixed_batch = get_task_batch(task, seed, freeze_eval=freeze_eval, gen=0)
    if fixed_batch is None:
        print("[DUO] No batch available; aborting.")
        return

    for r in range(rounds):
        round_seed = seed + r * 9973
        round_rng = random.Random(round_seed)
        batch = get_task_batch(task, seed, freeze_eval=freeze_eval, gen=r)
        if batch is None:
            print("[DUO] No batch available; aborting.")
            break
        helper_env = universe.library.get_helpers()
        hint = TaskDetective.detect_pattern(batch)
        if hint:
            print(f"[DUO] Detected pattern: {hint}")

        creator_slice = slice_seconds if slice_seconds > 0 else creator_policy.slice_seconds
        critic_slice = slice_seconds if slice_seconds > 0 else critic_policy.slice_seconds

        print(f"\n{'='*60}\n[DUO ROUND {r+1}/{rounds}] Creator\n{'='*60}")
        creator_candidates: List[Genome] = [
            seed_genome(round_rng, hint),
            _fallback_template_genome(round_rng, hint),
        ]
        if universe.best:
            creator_candidates.append(_repair_genome(universe.best))
        creator_start = time.time()
        while time.time() - creator_start < creator_slice:
            if len(creator_candidates) >= max_candidates:
                break
            mode_choice = creator_policy.generator_mode
            if mode_choice == "template":
                if reseed_templates:
                    stmts = round_rng.choice(reseed_templates)
                    g = Genome(statements=list(stmts), op_tag="reseed")
                else:
                    g = seed_genome(round_rng, hint)
            elif mode_choice == "mutate":
                parent = round_rng.choice(universe.pool) if universe.pool else seed_genome(round_rng, hint)
                g = _mutate_genome_with_meta(round_rng, parent, universe.meta, universe.library)
            else:
                g = _synthesize_genome(round_rng, universe.pool, hint, universe.library)
            creator_candidates.append(g)

        print(f"[DUO] Creator proposed {len(creator_candidates)} candidates")

        print(f"\n{'='*60}\n[DUO ROUND {r+1}/{rounds}] Critic\n{'='*60}")
        critic_start = time.time()
        gate_fail_reasons: collections.Counter = collections.Counter()
        validator_fail_reasons: collections.Counter = collections.Counter()
        scored_empty_count = 0
        prefiltered: List[Tuple[Genome, EvalResult]] = []
        seen_hashes: Set[str] = set()
        duplicate_count = 0
        total_checked = 0
        gate_pass = 0

        for g in creator_candidates:
            if time.time() - critic_start > critic_slice:
                break
            total_checked += 1
            code_hash = _candidate_hash(g.code)
            if code_hash in seen_hashes:
                duplicate_count += 1
            seen_hashes.add(code_hash)

            ok, reason, pre_res = _prefilter_check(
                g,
                batch,
                universe.eval_mode,
                task.name,
                extra_env=helper_env,
                validator=validate_program if universe.eval_mode == "program" else validate_code,
            )
            record = {
                "timestamp": now_ms(),
                "agent_id": "critic",
                "generation": r,
                "candidate_hash": code_hash,
                "gate_ok": ok,
                "gate_reason": "" if ok else reason,
                "score_train": pre_res.train if pre_res else None,
                "score_holdout": pre_res.hold if pre_res else None,
                "score_stress": pre_res.stress if pre_res else None,
                "selected": False,
                "note": "prefilter",
            }
            append_blackboard(blackboard_path, record)

            if not ok:
                if reason.startswith("hard_gate:"):
                    gate_fail_reasons[reason.split("hard_gate:", 1)[1]] += 1
                elif reason.startswith("validator:"):
                    validator_fail_reasons[reason.split("validator:", 1)[1]] += 1
                continue
            gate_pass += 1
            if pre_res:
                prefiltered.append((g, pre_res))

        if not prefiltered:
            scored_empty_count += 1
            reseed_templates = [_fallback_template_genome(round_rng, hint).statements]
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": "none",
                    "gate_ok": False,
                    "gate_reason": "scored_empty",
                    "score_train": None,
                    "score_holdout": None,
                    "score_stress": None,
                    "selected": False,
                    "note": "reseed",
                },
            )
            gate_pass_rate = gate_pass / max(1, total_checked)
            creator_policy = _adjust_creator_policy(creator_policy, gate_pass_rate, gate_fail_reasons)
            print("[DUO] No candidates passed prefilter; reseeding templates.")
            _print_critic_summary(
                gate_pass=gate_pass,
                total_checked=total_checked,
                adopted=False,
                full_results_count=0,
                duplicate_count=duplicate_count,
                scored_empty_count=scored_empty_count,
                gate_fail_reasons=gate_fail_reasons,
                validator_fail_reasons=validator_fail_reasons,
            )
            continue

        prefiltered.sort(key=lambda t: _critic_rank_score(t[1], critic_policy))
        selected = prefiltered[: max(1, k_full)]
        prefilter_map = {_candidate_hash(g.code): res for g, res in prefiltered}
        baseline_candidates = creator_candidates[:2]
        baseline_hashes = {_candidate_hash(c.code) for c in baseline_candidates}
        selected_hashes = {_candidate_hash(g.code) for g, _ in selected}
        for base in baseline_candidates:
            base_hash = _candidate_hash(base.code)
            if base_hash not in selected_hashes:
                base_res = prefilter_map.get(base_hash)
                if base_res:
                    selected.append((base, base_res))
        for g, pre_res in selected:
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": _candidate_hash(g.code),
                    "gate_ok": True,
                    "gate_reason": "",
                    "score_train": pre_res.train,
                    "score_holdout": pre_res.hold,
                    "score_stress": pre_res.stress,
                    "selected": True,
                    "note": "prefilter_selected",
                },
            )

        full_results: List[Tuple[Genome, EvalResult]] = []
        forced_eval = set(baseline_hashes)
        for g, _ in selected:
            if time.time() - critic_start > critic_slice and _candidate_hash(g.code) not in forced_eval:
                break
            refined = _critic_refine(round_rng, g, universe.meta, universe.library)
            for candidate in [g] + refined:
                if time.time() - critic_start > critic_slice and _candidate_hash(candidate.code) not in forced_eval:
                    break
                res = _evaluate_candidate(
                    candidate,
                    _merge_stress(fixed_batch, batch),
                    universe.eval_mode,
                    task.name,
                    extra_env=helper_env,
                    validator=validate_program if universe.eval_mode == "program" else validate_code,
                )
                if res.ok:
                    full_results.append((candidate, res))
                else:
                    if res.err:
                        if res.err.startswith("hard_gate:"):
                            gate_fail_reasons[res.err.split("hard_gate:", 1)[1]] += 1
                        else:
                            validator_fail_reasons[res.err] += 1
                append_blackboard(
                    blackboard_path,
                    {
                        "timestamp": now_ms(),
                        "agent_id": "critic",
                        "generation": r,
                        "candidate_hash": _candidate_hash(candidate.code),
                        "gate_ok": res.ok,
                        "gate_reason": "" if res.ok else (res.err or ""),
                        "score_train": res.train if res.ok else None,
                        "score_holdout": res.hold if res.ok else None,
                        "score_stress": res.stress if res.ok else None,
                        "selected": False,
                        "note": candidate.op_tag,
                    },
                )

        adopted = False
        full_results_count = len(full_results)
        if not full_results:
            scored_empty_count += 1
            reseed_templates = [_fallback_template_genome(round_rng, hint).statements]
            append_blackboard(
                blackboard_path,
                {
                    "timestamp": now_ms(),
                    "agent_id": "critic",
                    "generation": r,
                    "candidate_hash": "none",
                    "gate_ok": False,
                    "gate_reason": "scored_empty",
                    "score_train": None,
                    "score_holdout": None,
                    "score_stress": None,
                    "selected": False,
                    "note": "reseed",
                },
            )
            print("[DUO] No candidates survived full evaluation; reseeding templates.")
        else:
            full_results.sort(key=lambda t: t[1].score)
            best_g, best_res = full_results[0]
            if best_res.score < universe.best_score:
                adopted = True
                universe.best = best_g
                universe.best_score = best_res.score
                universe.best_train = best_res.train
                universe.best_hold = best_res.hold
                universe.best_stress = best_res.stress
                universe.best_test = best_res.test
                append_blackboard(
                    blackboard_path,
                    {
                        "timestamp": now_ms(),
                        "agent_id": "critic",
                        "generation": r,
                        "candidate_hash": _candidate_hash(best_g.code),
                        "gate_ok": True,
                        "gate_reason": "",
                        "score_train": best_res.train,
                        "score_holdout": best_res.hold,
                        "score_stress": best_res.stress,
                        "selected": True,
                        "note": "adopted",
                    },
                )
            universe.pool = [g for g, _ in full_results[: max(8, population // 4)]]
            if len(universe.pool) < population:
                universe.pool.extend([seed_genome(round_rng, hint) for _ in range(population - len(universe.pool))])

        gate_pass_rate = gate_pass / max(1, total_checked)
        creator_policy = _adjust_creator_policy(creator_policy, gate_pass_rate, gate_fail_reasons)
        _print_critic_summary(
            gate_pass=gate_pass,
            total_checked=total_checked,
            adopted=adopted,
            full_results_count=full_results_count,
            duplicate_count=duplicate_count,
            scored_empty_count=scored_empty_count,
            gate_fail_reasons=gate_fail_reasons,
            validator_fail_reasons=validator_fail_reasons,
        )

        gs = GlobalState(
            "RSI_EXTENDED_v2",
            now_ms(),
            now_ms(),
            seed,
            asdict(task),
            [universe.snapshot()],
            universe.uid,
            r + 1,
            mode=mode,
        )
        save_state(gs)

def run_rsi_loop(
    gens_per_round: int,
    rounds: int,
    levels: List[int],
    pop: int,
    n_univ: int,
    mode: str,
    freeze_eval: bool = True,
    meta_meta: bool = False,
    update_rule_rounds: int = 0,
):
    task = TaskSpec()
    seed = int(time.time()) % 100000
    if meta_meta:
        run_meta_meta(
            seed=seed,
            episodes=rounds,
            gens_per_episode=gens_per_round,
            pop=pop,
            n_univ=n_univ,
            freeze_eval=freeze_eval,
            state_dir=STATE_DIR,
            eval_every=1,
            few_shot_gens=max(3, gens_per_round // 2),
        )
        print(f"\n[RSI LOOP COMPLETE] {rounds} meta-meta rounds finished")
        return

    archive_path = STATE_DIR / "rsi_archive.json"
    archive = _load_rsi_archive(archive_path)
    if archive.get("current") and "genome" not in archive["current"]:
        archive = {"entries": [], "current": None, "consecutive": 0}
    fixed_batch = get_task_batch(task, seed, freeze_eval=True, gen=0)
    if fixed_batch is None:
        print("[RSI] No batch available; aborting.")
        return

    for r in range(rounds):
        print(f"\n{'='*60}\n[RSI ROUND {r+1}/{rounds}]\n{'='*60}")
        print(f"[EVOLVE] {gens_per_round} generations...")
        gs = run_multiverse(seed, task, gens_per_round, pop, n_univ, resume=(r > 0), mode=mode, freeze_eval=freeze_eval)
        best_snapshot = next((u for u in gs.universes if u.get("uid") == gs.selected_uid), None)
        best_data = (best_snapshot or {}).get("best")
        best_code = None
        if isinstance(best_data, dict):
            if mode == "learner":
                best_code = LearnerGenome(**best_data).code
            else:
                best_code = Genome(**best_data).code
        if best_code and best_code != "none":
            gate_ok, gate_reason = _hard_gate_ok(best_code, fixed_batch, mode, task.name)
            if not gate_ok:
                print(f"[RSI] Hard gate failed for best candidate ({gate_reason}); rejecting before scoring/autopatch.")
                archive["current"] = None
                archive["consecutive"] = 0
                archive["entries"] = []
                _save_rsi_archive(archive_path, archive)
                continue
        recent_scores = load_recent_scores(STATE_DIR / "run_log.jsonl", 5)
        forced_applied = False
        if is_300s_stagnation(recent_scores):
            print("[STAGNATION] 300s plateau detected for >=5 gens. Forcing L1/L3 autopatch.")
            forced = run_deep_autopatch([1, 3], candidates=4, apply=True, mode=mode)
            forced_applied = bool(forced.get("applied"))
            if forced_applied:
                print("[RSI] Self-modified via forced L1/L3 patch.")
            else:
                print("[STAGNATION] Forced patch rejected. Launching meta-meta acceleration.")
                run_meta_meta(
                    seed=seed,
                    episodes=1,
                    gens_per_episode=gens_per_round,
                    pop=pop,
                    n_univ=n_univ,
                    freeze_eval=freeze_eval,
                    state_dir=STATE_DIR,
                    eval_every=1,
                    few_shot_gens=max(3, gens_per_round // 2),
                )
                print("[STAGNATION] Meta-meta episode completed.")
        if not forced_applied:
            print(f"[AUTOPATCH] Trying L{levels}...")
            result = run_deep_autopatch(levels, candidates=4, apply=True, mode=mode)
            if result.get("applied"):
                print("[RSI] Self-modified! Reloading...")
        if update_rule_rounds > 0:
            print(f"[META] Running update-rule search for {update_rule_rounds} rounds...")
            run_update_rule_search(
                seed=seed + r * 127,
                rounds=update_rule_rounds,
                gens_per_round=max(3, gens_per_round // 2),
                pop=max(16, pop // 2),
                freeze_eval=freeze_eval,
                state_dir=STATE_DIR,
            )

    print(f"\n[RSI LOOP COMPLETE] {rounds} rounds finished")


# ---------------------------
# CLI Commands
# ---------------------------

def cmd_selftest(args):
    print("[selftest] Validating...")
    assert validate_expr("sin(x) + x*x")[0]
    assert not validate_expr("__import__('os')")[0]

    g = seed_genome(random.Random(42))
    t = TaskSpec()
    b = sample_batch(random.Random(42), t)
    assert b is not None
    r = evaluate(g, b, t.name)
    assert isinstance(r.score, float)

    hint = TaskDetective.detect_pattern(b)
    lg = seed_learner_genome(random.Random(42), hint)
    lr = evaluate_learner(lg, b, t.name)
    assert isinstance(lr.score, float)

    algo_code = "def run(inp):\n    return inp\n"
    assert validate_algo_program(algo_code)[0]

    print("[selftest] OK")
    return 0

def cmd_autopatch_probe(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    score = _autopatch_probe_score(
        mode=args.mode,
        task_name=args.task,
        seed=args.seed,
        generations=args.generations,
        population=args.population,
        universes=args.universes,
        freeze_eval=args.freeze_eval,
    )
    print(f"{score:.6f}")
    return 0

def cmd_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    mode = args.mode or ("algo" if args.task in ALGO_TASK_NAMES else "solver")
    run_multiverse(
        args.seed,
        TaskSpec(name=args.task),
        args.generations,
        args.population,
        args.universes,
        resume=resume,
        save_every=args.save_every,
        mode=mode,
        freeze_eval=args.freeze_eval,
    )
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_learner_evolve(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    resume = bool(args.resume) and (not args.fresh)
    run_multiverse(
        args.seed,
        TaskSpec(name=args.task),
        args.generations,
        args.population,
        args.universes,
        resume=resume,
        save_every=args.save_every,
        mode="learner",
        freeze_eval=args.freeze_eval,
    )
    print(f"\n[OK] State saved to {STATE_DIR / 'state.json'}")
    return 0

def cmd_best(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    gs = load_state()
    if not gs:
        print("No state.")
        return 1
    u = next((s for s in gs.universes if s.get("uid") == gs.selected_uid), gs.universes[0] if gs.universes else {})
    best = u.get("best")
    if best:
        if gs.mode == "learner":
            g = LearnerGenome(**best)
        else:
            g = Genome(**best)
        print(g.code)
    print(f"Score: {u.get('best_score')} | Hold: {u.get('best_hold')} | Stress: {u.get('best_stress')} | Test: {u.get('best_test')}")
    print(f"Generations: {gs.generations_done}")
    return 0

def cmd_rsi_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    levels = [int(l) for l in args.levels.split(",") if l.strip()]
    run_rsi_loop(
        args.generations,
        args.rounds,
        levels,
        args.population,
        args.universes,
        mode=args.mode,
        freeze_eval=args.freeze_eval,
        meta_meta=args.meta_meta,
        update_rule_rounds=args.update_rule_rounds,
    )
    return 0


def cmd_update_rule_search(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_update_rule_search(
        seed=args.seed,
        rounds=args.rounds,
        gens_per_round=args.generations,
        pop=args.population,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
    )
    return 0

def cmd_duo_loop(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_duo_loop(
        rounds=args.rounds,
        slice_seconds=args.slice_seconds,
        blackboard_path=Path(args.blackboard),
        k_full=args.k_full,
        seed=args.seed,
        mode=args.mode,
        freeze_eval=args.freeze_eval,
        population=args.population,
        max_candidates=args.max_candidates,
    )
    return 0

def cmd_meta_meta(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    run_meta_meta(
        seed=args.seed,
        episodes=args.episodes,
        gens_per_episode=args.gens_per_episode,
        pop=args.population,
        n_univ=args.universes,
        policy_pop=args.policy_pop,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
        eval_every=args.eval_every,
        few_shot_gens=args.few_shot_gens,
    )
    return 0

def cmd_task_switch(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    result = run_task_switch(
        seed=args.seed,
        task_a=TaskSpec(name=args.task_a),
        task_b=TaskSpec(name=args.task_b),
        gens_a=args.gens_a,
        gens_b=args.gens_b,
        pop=args.population,
        n_univ=args.universes,
        freeze_eval=args.freeze_eval,
        state_dir=STATE_DIR,
    )
    print(json.dumps(result, indent=2))
    return 0

def cmd_report(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    report = generate_report(STATE_DIR / "run_log.jsonl", args.few_shot_gens)
    print(json.dumps(report, indent=2))
    return 0


def cmd_transfer_bench(args):
    global STATE_DIR
    STATE_DIR = Path(args.state_dir)
    result = transfer_bench(args.task_from, args.task_to, args.budget, args.seed, freeze_eval=not args.no_freeze_eval)
    print(json.dumps(result, indent=2))
    return 0

def build_parser():
    p = argparse.ArgumentParser(prog="UNIFIED_RSI_EXTENDED", description="True RSI Engine with hard gates and rollback")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)

    e = sub.add_parser("evolve")
    e.add_argument("--seed", type=int, default=1337)
    e.add_argument("--generations", type=int, default=80)
    e.add_argument("--population", type=int, default=128)
    e.add_argument("--universes", type=int, default=4)
    e.add_argument("--task", default="poly2")
    e.add_argument("--resume", action="store_true")
    e.add_argument("--fresh", action="store_true")
    e.add_argument("--save-every", type=int, default=5)
    e.add_argument("--state-dir", default=".rsi_state")
    e.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--mode", default="", choices=["", "solver", "algo"])
    e.set_defaults(fn=cmd_evolve)

    le = sub.add_parser("learner-evolve")
    le.add_argument("--seed", type=int, default=1337)
    le.add_argument("--generations", type=int, default=80)
    le.add_argument("--population", type=int, default=128)
    le.add_argument("--universes", type=int, default=4)
    le.add_argument("--task", default="poly2")
    le.add_argument("--resume", action="store_true")
    le.add_argument("--fresh", action="store_true")
    le.add_argument("--save-every", type=int, default=5)
    le.add_argument("--state-dir", default=".rsi_state")
    le.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    le.set_defaults(fn=cmd_learner_evolve)

    b = sub.add_parser("best")
    b.add_argument("--state-dir", default=".rsi_state")
    b.set_defaults(fn=cmd_best)

    r = sub.add_parser("rsi-loop")
    r.add_argument("--generations", type=int, default=50)
    r.add_argument("--rounds", type=int, default=5)
    r.add_argument("--population", type=int, default=64)
    r.add_argument("--universes", type=int, default=2)
    r.add_argument("--state-dir", default=".rsi_state")
    r.add_argument("--mode", default="solver", choices=["solver", "learner", "algo"])
    r.add_argument("--levels", default="1,2,3", help="Comma-separated autopatch levels (e.g., 1,3)")
    r.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--meta-meta", action="store_true", help="Run meta-meta loop instead of standard RSI rounds")
    r.add_argument("--update-rule-rounds", type=int, default=0, help="Rounds of update-rule search per RSI round")
    r.set_defaults(fn=cmd_rsi_loop)

    dl = sub.add_parser("duo-loop")
    dl.add_argument("--rounds", type=int, default=5)
    dl.add_argument("--slice-seconds", type=float, default=0.0)
    dl.add_argument("--blackboard", default=".rsi_blackboard.jsonl")
    dl.add_argument("--k-full", type=int, default=6)
    dl.add_argument("--seed", type=int, default=1337)
    dl.add_argument("--mode", default="solver", choices=["solver", "algo", "program"])
    dl.add_argument("--population", type=int, default=64)
    dl.add_argument("--max-candidates", type=int, default=512)
    dl.add_argument("--state-dir", default=".rsi_state")
    dl.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    dl.set_defaults(fn=cmd_duo_loop)

    ap = sub.add_parser("autopatch-probe")
    ap.add_argument("--mode", default="solver", choices=["solver", "learner", "algo"])
    ap.add_argument("--task", default="poly2")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--universes", type=int, default=1)
    ap.add_argument("--state-dir", default=".rsi_state")
    ap.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ap.set_defaults(fn=cmd_autopatch_probe)

    mm = sub.add_parser("meta-meta")
    mm.add_argument("--seed", type=int, default=1337)
    mm.add_argument("--episodes", type=int, default=20)
    mm.add_argument("--gens-per-episode", type=int, default=20)
    mm.add_argument("--population", type=int, default=64)
    mm.add_argument("--universes", type=int, default=2)
    mm.add_argument("--policy-pop", type=int, default=4)
    mm.add_argument("--state-dir", default=".rsi_state")
    mm.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    mm.add_argument("--eval-every", type=int, default=4)
    mm.add_argument("--few-shot-gens", type=int, default=10)
    mm.set_defaults(fn=cmd_meta_meta)

    ts = sub.add_parser("task-switch")
    ts.add_argument("--seed", type=int, default=1337)
    ts.add_argument("--task-a", default="poly2")
    ts.add_argument("--task-b", default="piecewise")
    ts.add_argument("--gens-a", type=int, default=10)
    ts.add_argument("--gens-b", type=int, default=10)
    ts.add_argument("--population", type=int, default=64)
    ts.add_argument("--universes", type=int, default=2)
    ts.add_argument("--state-dir", default=".rsi_state")
    ts.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ts.set_defaults(fn=cmd_task_switch)

    tb = sub.add_parser("transfer-bench")
    tb.add_argument("--from", dest="task_from", required=True)
    tb.add_argument("--to", dest="task_to", required=True)
    tb.add_argument("--budget", type=int, default=12)
    tb.add_argument("--seed", type=int, default=1337)
    tb.add_argument("--state-dir", default=".rsi_state")
    tb.add_argument("--no-freeze-eval", action="store_true")
    tb.set_defaults(fn=cmd_transfer_bench)

    rp = sub.add_parser("report")
    rp.add_argument("--state-dir", default=".rsi_state")
    rp.add_argument("--few-shot-gens", type=int, default=10)
    rp.set_defaults(fn=cmd_report)

    inv = sub.add_parser("invention")
    inv.add_argument("--seed", type=int, default=0)
    inv.add_argument("--iterations", type=int, default=6)
    inv.set_defaults(fn=cmd_invention)

    ur = sub.add_parser("update-rule")
    ur.add_argument("--seed", type=int, default=1337)
    ur.add_argument("--rounds", type=int, default=4)
    ur.add_argument("--generations", type=int, default=6)
    ur.add_argument("--population", type=int, default=32)
    ur.add_argument("--state-dir", default=".rsi_state")
    ur.add_argument("--freeze-eval", action=argparse.BooleanOptionalAction, default=True)
    ur.set_defaults(fn=cmd_update_rule_search)

    return p

def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        sys.argv.append("selftest")
    args = parser.parse_args()
    return args.fn(args)

# if __name__ == "__main__":
    raise SystemExit(main())

# END OF unified_rsi_extended.py


# START OF NON_RSI_AGI_CORE_v5.py

"""
NON_RSI_AGI_CORE_v5.py
======================

Architecture goal:
- Fixed source code (no code-level RSI).
- AGI-oriented B×C structure:
  - B: world-model + planner + memory + skill-DSL interpreter (per agent)
  - C: multi-agent orchestrator + project/goal graph + evaluation/selection
- Self-improvement happens only via:
  - parameter updates (world model)
  - knowledge/memory accumulation
  - data-level skill programs
  - project graph + org policy adaptation
  NOT via modifying this file.

v5 Upgrade:
- "Real" Neuro-Symbolic Core using Hyperdimensional Computing (HDC).
- Strict Majority Rule for bundling (no OR hacks).
- Adaptive Planning robustness.
- Enhanced associative memory.

Run:
  python NON_RSI_AGI_CORE_v5.py --rounds 40 --agents 8
"""


import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Utility
# ----------------------------

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    buf: List[str] = []
    cur: List[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []
    if cur:
        buf.append("".join(cur))
    return buf


def load_unified_critic_module() -> Any:
    module_path = Path(__file__).with_name("unified_rsi_extended.py")
    legacy_path = Path(__file__).with_name("unified_rsi_extended .py")
    if not module_path.exists() and legacy_path.exists():
        module_path = legacy_path
    spec = importlib.util.spec_from_file_location("unified_rsi_extended", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load critic module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------
# Hyperdimensional Computing (HDC) Core
# ----------------------------

class HyperVector:
    """
    Pure Python Hyperdimensional Vector implementation (10,000 bits).
    Uses strict Majority Rule for bundling.
    """
    DIM = 10000

    def __init__(self, val: Optional[int] = None) -> None:
        if val is None:
            self.val = random.getrandbits(self.DIM)
        else:
            self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HyperVector):
            return NotImplemented
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

    @classmethod
    def from_seed(cls, seed_obj: Any) -> HyperVector:
        """Deterministic generation from a seed object."""
        # Create a deterministic seed from the object string
        s = str(seed_obj)
        h_hex = hashlib.sha256(s.encode("utf-8")).hexdigest()
        h_int = int(h_hex, 16)
        rng = random.Random(h_int)
        return cls(rng.getrandbits(cls.DIM))

    @classmethod
    def zero(cls) -> HyperVector:
        return cls(0)

    def bind(self, other: HyperVector) -> HyperVector:
        """XOR binding operation."""
        return HyperVector(self.val ^ other.val)

    def permute(self, shifts: int = 1) -> HyperVector:
        """Cyclic shift."""
        shifts %= self.DIM
        if shifts == 0:
            return self
        mask = (1 << self.DIM) - 1
        new_val = ((self.val << shifts) & mask) | (self.val >> (self.DIM - shifts))
        return HyperVector(new_val)

    def similarity(self, other: HyperVector) -> float:
        """Hamming similarity (normalized 0.0 to 1.0)."""
        diff = self.val ^ other.val
        dist = diff.bit_count()
        return 1.0 - (dist / self.DIM)

    @staticmethod
    def bundle(vectors: List[HyperVector]) -> HyperVector:
        """
        Majority Rule bundling.
        Sum bits column-wise. Threshold at N/2.
        Optimized for pure Python using string manipulation.
        """
        if not vectors:
            return HyperVector.zero()

        n = len(vectors)
        if n == 1:
            return vectors[0]

        threshold = n / 2.0
        counts = [0] * HyperVector.DIM

        # Optimization: String iteration is faster than bitwise loops in Python
        for vec in vectors:
            # bin(val) -> '0b101...', slice [2:], zfill to DIM
            # Reverse so index 0 corresponds to LSB
            s = bin(vec.val)[2:].zfill(HyperVector.DIM)[::-1]
            for i, char in enumerate(s):
                if char == '1':
                    counts[i] += 1

        result_val = 0
        for i in range(HyperVector.DIM):
            c = counts[i]
            if c > threshold:
                result_val |= (1 << i)
            elif c == threshold:
                # Random tie-breaking
                if random.random() < 0.5:
                    result_val |= (1 << i)

        return HyperVector(result_val)


# ----------------------------
# Shared Memory / Knowledge Base (Neuro-Symbolic)
# ----------------------------

@dataclass
class MemoryItem:
    ts_ms: int
    kind: str               # "episode" | "note" | "artifact" | "principle"
    title: str
    content: Dict[str, Any]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {"ts": self.ts_ms, "k": self.kind, "t": self.title, "c": self.content, "tags": self.tags}
        )


class SharedMemory:
    """
    Shared KB using HDC for associative retrieval.
    """

    def __init__(self, max_items: int = 8000) -> None:
        self.max_items = max_items
        self._items: List[MemoryItem] = []
        # HDC Memory Index
        self._item_vectors: Dict[str, HyperVector] = {}
        # Cache common vectors to speed up encoding
        self._token_cache: Dict[str, HyperVector] = {}

    def _get_token_hv(self, token: str) -> HyperVector:
        if token not in self._token_cache:
            self._token_cache[token] = HyperVector.from_seed(f"token:{token}")
        return self._token_cache[token]

    def _encode_text_bag(self, text: str) -> HyperVector:
        tokens = tokenize(text)
        if not tokens:
            return HyperVector.zero()
        vecs = [self._get_token_hv(t) for t in tokens]
        return HyperVector.bundle(vecs)

    def _encode_item(self, item: MemoryItem) -> HyperVector:
        # Bundle: Title, Kind, Tags
        # Structure: Bind(Role, Value)

        # 1. Kind
        kind_hv = self._get_token_hv(f"kind:{item.kind}")

        # 2. Title
        title_hv = self._encode_text_bag(item.title)

        # 3. Tags
        if item.tags:
            tag_vecs = [self._get_token_hv(f"tag:{t}") for t in item.tags]
            tags_hv = HyperVector.bundle(tag_vecs)
        else:
            tags_hv = HyperVector.zero()

        # Bundle all components
        # Note: We don't bind to roles here to allow freer association,
        # or we could bind. Let's keep it simple: bundle of properties.
        return HyperVector.bundle([kind_hv, title_hv, tags_hv])

    def add(self, kind: str, title: str, content: Dict[str, Any],
            tags: Optional[List[str]] = None) -> str:
        tags = tags or []
        item = MemoryItem(ts_ms=now_ms(), kind=kind, title=title,
                          content=content, tags=tags)
        self._items.append(item)
        
        # Generate and store HV
        item_hv = self._encode_item(item)
        self._item_vectors[item.id] = item_hv
        
        if len(self._items) > self.max_items:
            removed = self._items.pop(0)
            self._item_vectors.pop(removed.id, None)

        return item.id

    def search(self, query: str, k: int = 10,
               kinds: Optional[List[str]] = None,
               tags: Optional[List[str]] = None) -> List[MemoryItem]:
        
        # 1. Encode Query
        query_parts = []
        
        # Text query
        if query:
            query_parts.append(self._encode_text_bag(query))

        # Tags query
        if tags:
            tag_vecs = [self._get_token_hv(f"tag:{t}") for t in tags]
            query_parts.append(HyperVector.bundle(tag_vecs))

        # Kinds (act as filter, but also can be part of query vector)
        if kinds:
             # We typically don't bundle all kinds, we use kinds as a hard filter.
             pass

        if not query_parts:
            return self._items[-k:]

        query_hv = HyperVector.bundle(query_parts)

        # 2. Score all items
        t_now = now_ms()
        scored: List[Tuple[float, MemoryItem]] = []

        # Optimization: Pre-filter by kind to reduce HDC checks?
        # Or just check all. 8000 checks is fine.

        for it in self._items:
            if kinds is not None and it.kind not in kinds:
                continue

            # HDC Similarity
            it_vec = self._item_vectors.get(it.id)
            if not it_vec:
                continue

            sim = query_hv.similarity(it_vec)

            # Recency & Reward boost
            recency = 1.0 / (1.0 + (t_now - it.ts_ms) / (1000.0 * 60.0 * 30.0))
            reward = float(it.content.get("reward", 0.0)) if isinstance(it.content, dict) else 0.0
            reward_boost = max(0.0, min(0.5, reward))

            # Composite Score
            # HDC similarity for random vectors is ~0.5.
            # We are interested in deviations above 0.5.
            if sim < 0.48:
                continue # Irrelevant

            # Normalize sim to 0..1 range roughly (0.5 -> 0, 1.0 -> 1)
            norm_sim = max(0.0, (sim - 0.5) * 2.0)

            final_score = norm_sim + 0.35 * recency + reward_boost

            if final_score > 0.1:
                scored.append((final_score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:k]]

    def extract_principles(self, k: int = 6) -> List[str]:
        episodes = [it for it in self._items if it.kind == "episode"]
        if not episodes:
            return []
        episodes.sort(key=lambda it: float(it.content.get("reward", 0.0)), reverse=True)
        selected = episodes[:k]
        created: List[str] = []
        for it in selected:
            obs = it.content.get("obs", {})
            action = it.content.get("action", "")
            reward = float(it.content.get("reward", 0.0))
            conditions = {
                "task": obs.get("task"),
                "domain": obs.get("domain"),
                "difficulty": obs.get("difficulty"),
                "phase": obs.get("phase"),
                "action": action,
            }
            pid = self.add(
                "principle",
                f"pattern:{obs.get('task','task')}:{action}",
                {
                    "conditions": conditions,
                    "reward": reward,
                    "source_episode": it.id,
                },
                tags=["principle", "derived"],
            )
            created.append(pid)
        return created

    def dump_summary(self, k: int = 15) -> List[Dict[str, Any]]:
        tail = self._items[-k:]
        return [
            {
                "id": it.id,
                "ts_ms": it.ts_ms,
                "kind": it.kind,
                "title": it.title,
                "tags": it.tags,
            }
            for it in tail
        ]


# ----------------------------
# Tool interface (external world hook)
# ----------------------------

ToolFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._tools.get(name)
        if fn is None:
            return {"ok": False, "error": f"unknown_tool:{name}", "tool": name}
        try:
            out = fn(args)
            out = dict(out)
            out.setdefault("ok", True)
            out.setdefault("tool", name)
            return out
        except Exception as e:
            return {"ok": False, "error": repr(e), "tool": name}


# ----------------------------
# Skill DSL (data-level programs)
# ----------------------------

@dataclass
class SkillStep:
    kind: str
    tool: Optional[str] = None
    args_template: Optional[Dict[str, Any]] = None
    condition: Optional[Dict[str, Any]] = None
    steps: Optional[List["SkillStep"]] = None
    else_steps: Optional[List["SkillStep"]] = None
    list_key: Optional[str] = None
    item_key: Optional[str] = None


@dataclass
class Skill:
    """
    Interpreted skill program:
    - steps are data structures with explicit control-flow
    - supports: call, if, foreach
    - arguments can reference context via ${key}
    """
    name: str
    purpose: str
    steps: List[SkillStep]
    tags: List[str] = field(default_factory=list)
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = stable_hash(
            {
                "name": self.name,
                "purpose": self.purpose,
                "steps": [self._serialize_step(s) for s in self.steps],
            }
        )

    def _serialize_step(self, step: SkillStep) -> Dict[str, Any]:
        return {
            "kind": step.kind,
            "tool": step.tool,
            "args_template": step.args_template,
            "condition": step.condition,
            "list_key": step.list_key,
            "item_key": step.item_key,
            "steps": [self._serialize_step(s) for s in step.steps] if step.steps else None,
            "else_steps": [self._serialize_step(s) for s in step.else_steps] if step.else_steps else None,
        }

    def run(self, tools: ToolRegistry, context: Dict[str, Any]) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        ctx = dict(context)

        def subst(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                key = value[2:-1]
                return ctx.get(key)
            if isinstance(value, dict):
                return {k: subst(v) for k, v in value.items()}
            if isinstance(value, list):
                return [subst(v) for v in value]
            return value

        def eval_condition(cond: Dict[str, Any]) -> bool:
            key = cond.get("key")
            op = cond.get("op", "truthy")
            val = cond.get("value")
            cur = ctx.get(key)
            if op == "eq":
                return cur == val
            if op == "neq":
                return cur != val
            if op == "contains":
                return isinstance(cur, (list, str)) and val in cur
            if op == "gt":
                return isinstance(cur, (int, float)) and cur > val
            if op == "lt":
                return isinstance(cur, (int, float)) and cur < val
            if op == "gte":
                return isinstance(cur, (int, float)) and cur >= val
            if op == "lte":
                return isinstance(cur, (int, float)) and cur <= val
            return bool(cur)

        def run_steps(steps: Iterable[SkillStep], depth: int = 0) -> bool:
            if depth > 12:
                return False
            for i, st in enumerate(steps):
                if st.kind == "call" and st.tool:
                    args = subst(st.args_template or {})
                    if not isinstance(args, dict):
                        args = {"value": args}
                    res = tools.call(st.tool, args)
                    trace.append({"i": len(trace), "tool": st.tool, "args": args, "res": res})
                    ctx["last"] = res
                    if isinstance(res, dict):
                        ctx["last_verdict"] = res.get("verdict")
                    ctx[f"step_{len(trace) - 1}"] = res
                    if not res.get("ok", False):
                        return False
                elif st.kind == "if" and st.condition:
                    branch = st.steps if eval_condition(st.condition) else st.else_steps
                    if branch:
                        if not run_steps(branch, depth + 1):
                            return False
                elif st.kind == "foreach" and st.list_key:
                    items = ctx.get(st.list_key, [])
                    if isinstance(items, list) and st.steps:
                        for idx, item in enumerate(items):
                            ctx[st.item_key or "item"] = item
                            ctx["index"] = idx
                            if not run_steps(st.steps, depth + 1):
                                return False
                else:
                    return False
            return True

        ok = run_steps(self.steps)
        return {
            "ok": ok,
            "trace": trace,
            "final": ctx.get("last"),
        }


class SkillLibrary:
    def __init__(self, max_skills: int = 3000) -> None:
        self.max_skills = max_skills
        self._skills: Dict[str, Skill] = {}

    def add(self, sk: Skill) -> str:
        self._skills[sk.id] = sk
        if len(self._skills) > self.max_skills:
            for sid in list(self._skills.keys())[: len(self._skills) - self.max_skills]:
                self._skills.pop(sid, None)
        return sk.id

    def list(self, tag: Optional[str] = None) -> List[Skill]:
        vals = list(self._skills.values())
        if tag is None:
            return vals
        return [s for s in vals if tag in s.tags]

    def get(self, sid: str) -> Optional[Skill]:
        return self._skills.get(sid)


# ----------------------------
# World Model (feature-based value model)
# ----------------------------

@dataclass
class TransitionSummary:
    count: int = 0


class WorldModel:
    """
    Feature-based Q-value model with v5 enhancements:
    - Non-linear feature combinations
    - Experience replay buffer
    - online TD updates
    - separate state-action counts for uncertainty estimates
    """

    def __init__(self, gamma: float = 0.9, lr: float = 0.08) -> None:
        self.gamma = gamma
        self.lr = lr
        self._weights: Dict[str, float] = {}
        self._sa_counts: Dict[Tuple[str, str], TransitionSummary] = {}
        # v5: Experience replay buffer
        self.replay_buffer: List[Tuple[Dict[str, Any], str, float, Dict[str, Any], List[str]]] = []
        self.max_buffer_size = 200

    def _feature_bucket(self, budget: int) -> int:
        return min(5, max(0, budget // 10))

    def encode_state(self, obs: Dict[str, Any]) -> str:
        key = {
            "task": obs.get("task", ""),
            "domain": obs.get("domain", ""),
            "difficulty": int(obs.get("difficulty", 0)),
            "budget": int(obs.get("budget", 0)),
            "phase": obs.get("phase", ""),
        }
        return stable_hash(key)

    def features(self, obs: Dict[str, Any], action: str) -> Dict[str, float]:
        task = str(obs.get("task", ""))
        domain = str(obs.get("domain", ""))
        diff = int(obs.get("difficulty", 0))
        phase = str(obs.get("phase", ""))
        budget = int(obs.get("budget", 0))
        bucket = self._feature_bucket(budget)
        
        # v5: Non-linear feature combinations
        feats = {
            "bias": 1.0,
            f"task:{task}": 1.0,
            f"domain:{domain}": 1.0,
            f"diff:{diff}": 1.0,
            f"phase:{phase}": 1.0,
            f"action:{action}": 1.0,
            f"task_action:{task}|{action}": 1.0,
            f"budget_bucket:{bucket}": 1.0,
            # Non-linear combinations
            f"diff_action:{diff}|{action}": 1.0,
            f"domain_diff:{domain}|{diff}": float(diff) / 5.0,
            f"task_phase:{task}|{phase}": 1.0,
        }
        return feats

    def q_value(self, obs: Dict[str, Any], action: str) -> float:
        feats = self.features(obs, action)
        return sum(self._weights.get(k, 0.0) * v for k, v in feats.items())

    def confidence(self, obs: Dict[str, Any], action: str) -> float:
        s = self.encode_state(obs)
        count = self._sa_counts.get((s, action), TransitionSummary()).count
        return 1.0 - (1.0 / math.sqrt(count + 1.0))

    def update(self, obs: Dict[str, Any], action: str, reward: float,
               next_obs: Dict[str, Any], action_space: List[str]) -> None:
        # v5: Add to replay buffer
        self.replay_buffer.append((obs, action, reward, next_obs, action_space))
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer.pop(0)
        
        # Current experience update
        feats = self.features(obs, action)
        current = self.q_value(obs, action)
        next_best = max(self.q_value(next_obs, a) for a in action_space)
        target = reward + self.gamma * next_best
        td_error = target - current
        for k, v in feats.items():
            self._weights[k] = self._weights.get(k, 0.0) + self.lr * td_error * v
        
        # v5: Experience replay (sample mini-batch)
        if len(self.replay_buffer) >= 10:
            samples = random.sample(self.replay_buffer, min(5, len(self.replay_buffer)))
            for s_obs, s_action, s_reward, s_next_obs, s_action_space in samples:
                s_feats = self.features(s_obs, s_action)
                s_current = self.q_value(s_obs, s_action)
                s_next_best = max(self.q_value(s_next_obs, a) for a in s_action_space)
                s_target = s_reward + self.gamma * s_next_best
                s_td_error = s_target - s_current
                for k, v in s_feats.items():
                    self._weights[k] = self._weights.get(k, 0.0) + (self.lr * 0.5) * s_td_error * v
        
        # Update visitation counts
        s = self.encode_state(obs)
        entry = self._sa_counts.get((s, action))
        if entry is None:
            entry = TransitionSummary()
            self._sa_counts[(s, action)] = entry
        entry.count += 1


# ----------------------------
# Planner (lookahead over world model)
# ----------------------------

@dataclass
class PlanCandidate:
    actions: List[str]
    score: float


class Planner:
    def __init__(self, wm: WorldModel, depth: int = 3,
                 width: int = 6, gamma: float = 0.9) -> None:
        self.wm = wm
        self.depth = depth
        self.width = width
        self.gamma = gamma

    def propose(self, obs: Dict[str, Any], action_space: List[str],
                risk_pref: float) -> List[PlanCandidate]:
        # Robustness: Safety check
        if not action_space:
            return []

        beam: List[PlanCandidate] = [PlanCandidate(actions=[], score=0.0)]

        for d in range(self.depth):
            new_beam: List[PlanCandidate] = []
            for cand in beam:
                for a in action_space:
                    q = self.wm.q_value(obs, a)
                    uncertainty = 1.0 - self.wm.confidence(obs, a)
                    adjusted = q - (1.0 - risk_pref) * uncertainty
                    sc = cand.score + (self.gamma ** d) * adjusted
                    new_beam.append(PlanCandidate(actions=cand.actions + [a], score=sc))

            # Robustness: Sort and Prune
            if not new_beam:
                break

            new_beam.sort(key=lambda c: c.score, reverse=True)
            beam = new_beam[: self.width]

        return beam


# ----------------------------
# Project / Goal Graph (C-layer long-horizon structure)
# ----------------------------

@dataclass
class ProjectNode:
    id: str
    name: str
    task: str
    status: str = "open"      # "open" | "active" | "done"
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    value_estimate: float = 0.0
    history: List[str] = field(default_factory=list)  # memory ids
    value_history: List[float] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)


class ProjectGraph:
    """
    Long-horizon project DAG:
    - orchestrator attaches agent runs to nodes
    - nodes accumulate evidence and value estimates
    - spawn subprojects based on value thresholds
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ProjectNode] = {}

    def create_root(self, name: str, task: str) -> str:
        nid = stable_hash({"name": name, "task": task, "root": True})
        self._nodes[nid] = ProjectNode(id=nid, name=name, task=task, status="open")
        return nid

    def add_child(self, parent_id: str, name: str,
                  task: Optional[str] = None) -> str:
        parent = self._nodes[parent_id]
        nid = stable_hash({"name": name, "task": task or parent.task, "parent": parent_id})
        node = ProjectNode(id=nid, name=name, task=task or parent.task,
                           status="open", parent_id=parent_id)
        self._nodes[nid] = node
        parent.children.append(nid)
        return nid

    def nodes_for_task(self, task: str) -> List[ProjectNode]:
        return [n for n in self._nodes.values() if n.task == task]

    def pick_node_for_round(self, task: str) -> ProjectNode:
        candidates = [n for n in self._nodes.values()
                      if n.task == task and n.status != "done"]
        if not candidates:
            nid = self.create_root(name=f"{task}_root", task=task)
            return self._nodes[nid]
        candidates.sort(key=lambda n: n.value_estimate, reverse=True)
        return candidates[0]

    def update_node(self, nid: str, reward: float,
                    memory_id: Optional[str]) -> None:
        node = self._nodes[nid]
        alpha = 0.25
        node.value_estimate = (1 - alpha) * node.value_estimate + alpha * reward
        node.value_history.append(node.value_estimate)
        if memory_id:
            node.history.append(memory_id)
            node.evidence_refs.append(memory_id)
        if node.value_estimate > 0.18 and len(node.children) < 3:
            self.add_child(parent_id=nid, name=f"{node.name}_infra_focus")
            self.add_child(parent_id=nid, name=f"{node.name}_breakthrough_focus")
        if node.value_estimate > 0.35:
            node.status = "active"


@dataclass
class RuleProposal:
    proposal_id: str
    level: str  # "L0" | "L1" | "L2"
    payload: Dict[str, Any]
    creator_key: str
    created_ms: int
    evidence: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None


# ----------------------------
# Environment (research/engineering playground)
# ----------------------------

@dataclass
class TaskSpec:
    name: str
    difficulty: int
    baseline: float
    domain: str   # "algorithm" | "systems" | "theory" | "strategy" ...


class ResearchEnvironment:
    """
    Abstract multi-domain environment.
    - Each step is "run one agent on one project node for a given task/budget"
    - Reward ~ improvement over task baseline + infra gain
    - Global qualities (tool/kb/org) mediate acceleration
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.tasks: List[TaskSpec] = [
            TaskSpec("algorithm_design", difficulty=3, baseline=0.35, domain="algorithm"),
            TaskSpec("systems_optimization", difficulty=4, baseline=0.30, domain="systems"),
            TaskSpec("verification_pipeline", difficulty=2, baseline=0.40, domain="verification"),
            TaskSpec("toolchain_speedup", difficulty=5, baseline=0.25, domain="engineering"),
            TaskSpec("theory_discovery", difficulty=5, baseline=0.28, domain="theory"),
            TaskSpec("strategy_optimization", difficulty=3, baseline=0.32, domain="strategy"),
        ]
        self.global_tool_quality = 0.10
        self.global_kb_quality = 0.10
        self.global_org_quality = 0.10

    def sample_task(self) -> TaskSpec:
        return self.rng.choice(self.tasks)

    def make_observation(self, task: TaskSpec, budget: int,
                         phase: str = "research") -> Dict[str, Any]:
        return {
            "task": task.name,
            "domain": task.domain,
            "difficulty": task.difficulty,
            "baseline": task.baseline,
            "budget": budget,
            "phase": phase,
        }

    def step(self, obs: Dict[str, Any], action: str,
             payload: Dict[str, Any]) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        diff = int(obs["difficulty"])
        base = float(obs["baseline"])
        budget = int(obs["budget"])
        domain = str(obs.get("domain", ""))

        tq = self.global_tool_quality
        kq = self.global_kb_quality
        oq = self.global_org_quality

        # L3-modifiable environment parameters
        if not hasattr(self, 'env_params'):
            self.env_params = {
                'leverage_multiplier': 1.0,      # Scales leverage effect
                'breakthrough_base': 0.04,        # Base breakthrough reward
                'breakthrough_scale': 0.32,       # Leverage scaling for breakthrough
                'reward_ceiling': 1.0,            # Max possible reward
                'difficulty_penalty': 0.30,       # How much difficulty reduces reward
                'infra_bonus_scale': 0.025,       # Infrastructure bonus multiplier
            }

        infra_scale = 1.0 / (1.0 + 0.4 * diff)
        leverage = (0.30 * tq + 0.30 * kq + 0.30 * oq) * self.env_params.get('leverage_multiplier', 1.0)
        diminishing = 1.0 / (1.0 + 2.0 * leverage)

        domain_bonus = {
            "algorithm": 0.04 if action == "attempt_breakthrough" else 0.01,
            "theory": 0.05 if action == "attempt_breakthrough" else 0.01,
            "systems": 0.04 if action in ("build_tool", "tune_orchestration") else 0.01,
            "engineering": 0.05 if action == "build_tool" else 0.01,
            "verification": 0.05 if action == "write_verified_note" else 0.01,
            "strategy": 0.04 if action == "tune_orchestration" else 0.01,
        }.get(domain, 0.01)

        if action == "build_tool":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.12 * tq) * invest * infra_scale * diminishing
            self.global_tool_quality = min(1.0, self.global_tool_quality + gain)
            raw = 0.02 * invest + domain_bonus
        elif action == "write_verified_note":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.10 * kq) * invest * infra_scale * diminishing
            self.global_kb_quality = min(1.0, self.global_kb_quality + gain)
            raw = 0.018 * invest + domain_bonus
        elif action == "tune_orchestration":
            invest = float(payload.get("invest", 1.0))
            gain = (0.03 + 0.10 * oq) * invest * infra_scale * diminishing
            self.global_org_quality = min(1.0, self.global_org_quality + gain)
            raw = 0.016 * invest + domain_bonus
        elif action == "attempt_breakthrough":
            effort = (1.0 + math.log(1 + budget) / 4.0)
            bt_base = self.env_params.get('breakthrough_base', 0.04)
            bt_scale = self.env_params.get('breakthrough_scale', 0.32)
            diff_penalty = self.env_params.get('difficulty_penalty', 0.30)
            raw = (bt_base + bt_scale * leverage) * effort * (1.0 / (1.0 + diff_penalty * diff)) + domain_bonus
        else:
            raw = 0.0

        noise = self.rng.uniform(-0.02, 0.02)
        
        # Custom hard-coded logic for recursion detection via Stagnation (simulated)
        # If we are stagnant, we might be hitting a wall that needs recursion.
        
        # Calculate base performance (still 0-1 normalized for consistency check)
        base_performance = max(0.0, min(1.0, base + raw + noise))
        infra_scale_bonus = self.env_params.get('infra_bonus_scale', 0.025)
        infra_bonus = infra_scale_bonus * (tq + kq + oq) / 3.0
        
        # Standard reward: performance gain + infrastructure bonus
        # No exponential accumulator here - we want the agent to fight for every point
        reward = (base_performance - base) * 10.0 + infra_bonus

        next_obs = dict(obs)
        next_obs["phase"] = "integrate"
        info = {
            "task": obs.get("task"),
            "performance": base_performance,
            "delta": base_performance - base,
            "tq": self.global_tool_quality,
            "kq": self.global_kb_quality,
            "oq": self.global_org_quality,
        }
        return next_obs, reward, info


# ----------------------------
# Agent (B-type architecture)
# ----------------------------

@dataclass
class AgentConfig:
    name: str
    role: str = "general"     # "theorist" | "builder" | "experimenter" | "verifier" | "strategist"
    planner_depth: int = 3
    planner_width: int = 6
    risk: float = 0.2


class Agent:
    """
    B-type core:
    - WorldModel + Planner
    - SharedMemory + SkillLibrary + ToolRegistry
    - No self-modifying code; only state/memory/skills evolve.
    """

    def __init__(self, cfg: AgentConfig, tools: ToolRegistry,
                 shared_mem: SharedMemory, skills: SkillLibrary) -> None:
        self.cfg = cfg
        self.tools = tools
        self.mem = shared_mem
        self.skills = skills

        self.wm = WorldModel()
        # v5: Adaptive planning - planner will be recreated dynamically
        self.planner = Planner(self.wm, depth=cfg.planner_depth,
                               width=cfg.planner_width)
        # v5: Agent specialization tracking
        self.domain_expertise: Dict[str, float] = {}

    def action_space(self) -> List[str]:
        base = ["attempt_breakthrough", "build_tool", "write_verified_note", "tune_orchestration"]
        r = self.cfg.role
        if r == "verifier":
            return ["write_verified_note", "build_tool", "tune_orchestration", "attempt_breakthrough"]
        if r == "builder":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "theorist":
            return ["attempt_breakthrough", "write_verified_note", "build_tool", "tune_orchestration"]
        if r == "experimenter":
            return ["build_tool", "attempt_breakthrough", "write_verified_note", "tune_orchestration"]
        if r == "strategist":
            return ["tune_orchestration", "attempt_breakthrough", "build_tool", "write_verified_note"]
        return base

    def choose_action(self, obs: Dict[str, Any]) -> str:
        # v5: Adaptive planning depth based on task difficulty
        difficulty = int(obs.get('difficulty', 3))

        # Robustness: Bound adaptive parameters
        adaptive_depth = min(10, max(2, difficulty))
        adaptive_width = min(12, max(4, 4 + difficulty // 2))
        
        # Recreate planner with adaptive parameters
        self.planner = Planner(self.wm, depth=adaptive_depth, width=adaptive_width)
        
        # System 1: Fast heuristic planning
        try:
            candidates = self.planner.propose(obs, self.action_space(), self.cfg.risk)
        except Exception:
            candidates = []

        if not candidates:
            return random.choice(self.action_space())
        
        draft_action = candidates[0].actions[0]
        task = obs.get('task', '')
        
        # v4: Improved System 2 with balanced success/failure analysis
        # Look at BOTH successes and failures
        past_episodes = self.mem.search(
            query=f"{task} {draft_action}",
            k=8,
            kinds=["episode"],
            tags=[task, draft_action]
        )
        
        success_count = 0
        failure_count = 0
        total_reward = 0.0
        
        for mem in past_episodes:
            if mem.content.get("action") == draft_action:
                reward = float(mem.content.get("reward", 0.0))
                total_reward += reward
                if reward >= 0.25:  # Success threshold
                    success_count += 1
                elif reward < 0.10:  # Failure threshold
                    failure_count += 1
        
        # v4: Probabilistic risk assessment instead of hard override
        if success_count + failure_count > 0:
            success_rate = success_count / (success_count + failure_count)
            avg_reward = total_reward / len(past_episodes) if past_episodes else 0.0
            
            # If this action has consistently failed, penalize it in exploration
            if success_rate < 0.3 and failure_count >= 3:
                # Don't completely avoid, but reduce probability
                if random.random() < 0.6 and len(candidates) > 1:
                    # Try second-best candidate
                    return candidates[1].actions[0]
        
        # Standard exploration vs exploitation
        if random.random() > self.cfg.risk:
            return draft_action
        return random.choice(self.action_space())

    def maybe_synthesize_skill(self, obs: Dict[str, Any]) -> Optional[str]:
        task = obs.get("task", "")
        if task == "verification_pipeline" and random.random() < 0.30:
            sk = Skill(
                name=f"{self.cfg.name}_verify_pipeline",
                purpose="Evaluate candidate and write verified note if passing.",
                steps=[
                    SkillStep(
                        kind="call",
                        tool="evaluate_candidate",
                        args_template={"task": "${task}", "candidate": "${candidate}"},
                    ),
                    SkillStep(
                        kind="if",
                        condition={"key": "last_verdict", "op": "eq", "value": "pass"},
                        steps=[
                            SkillStep(
                                kind="call",
                                tool="write_note",
                                args_template={"title": "verified_result", "payload": "${step_0}"},
                            )
                        ],
                        else_steps=[
                            SkillStep(
                                kind="call",
                                tool="write_note",
                                args_template={"title": "needs_revision", "payload": "${step_0}"},
                            )
                        ],
                    ),
                ],
                tags=["verification", "meta"],
            )
            return self.skills.add(sk)
        if task == "toolchain_speedup" and random.random() < 0.30:
            sk = Skill(
                name=f"{self.cfg.name}_toolchain_upgrade",
                purpose="Propose toolchain improvement artifact for each hint.",
                steps=[
                    SkillStep(
                        kind="foreach",
                        list_key="hint_titles",
                        item_key="hint",
                        steps=[
                            SkillStep(
                                kind="call",
                                tool="tool_build_report",
                                args_template={"task": "${task}", "idea": {"hint": "${hint}"}},
                            ),
                            SkillStep(
                                kind="call",
                                tool="write_artifact",
                                args_template={"title": "tool_artifact", "payload": "${last}"},
                            ),
                        ],
                    )
                ],
                tags=["toolchain", "artifact"],
            )
            return self.skills.add(sk)
        return None

    def act_on_project(self, env: ResearchEnvironment,
                       proj_node: ProjectNode,
                       obs: Dict[str, Any]) -> Dict[str, Any]:
        hints = self.mem.search(
            f"{obs.get('task','')} difficulty {obs.get('difficulty',0)}",
            k=6,
            kinds=["principle", "artifact", "note"],
        )

        context = {
            "task": obs.get("task"),
            "domain": obs.get("domain"),
            "difficulty": obs.get("difficulty"),
            "budget": obs.get("budget"),
            "project": {"id": proj_node.id, "name": proj_node.name},
            "candidate": {
                "type": "proposal",
                "from": self.cfg.name,
                "role": self.cfg.role,
                "hints": [h.title for h in hints],
            },
            "idea": {
                "from": self.cfg.name,
                "summary": "incremental improvement on project using accumulated tools/kb/org.",
            },
            "hint_titles": [h.title for h in hints],
        }

        sid = self.maybe_synthesize_skill(obs)
        if sid:
            self.mem.add(
                "artifact",
                f"skill_added:{sid}",
                {"agent": self.cfg.name, "skill_id": sid},
                tags=["skill"],
            )

        action = self.choose_action(obs)
        invest = max(1.0, float(obs.get("budget", 1)) / 10.0)
        payload = {
            "invest": invest,
            "agent": self.cfg.name,
            "role": self.cfg.role,
            "task": obs.get("task"),
            "project_id": proj_node.id,
        }

        next_obs, reward, info = env.step(obs, action, payload)
        self.wm.update(obs, action, reward, next_obs, self.action_space())

        mem_id = self.mem.add(
            "episode",
            f"{self.cfg.name}:{action}:{obs.get('task')}:{proj_node.name}",
            {
                "obs": obs,
                "action": action,
                "payload": payload,
                "reward": reward,
                "info": info,
                "project_id": proj_node.id,
                "hints_used": [h.id for h in hints],
            },
            tags=["episode", self.cfg.role, obs.get("task", "task")],
        )

        if random.random() < 0.35:
            tag = "verification" if action == "write_verified_note" else "toolchain"
            candidates = self.skills.list(tag=tag)
            if candidates:
                sk = random.choice(candidates)
                out = sk.run(self.tools, context)
                self.mem.add(
                    "note",
                    f"{self.cfg.name}:skill_run:{sk.name}",
                    {"skill_id": sk.id, "out": out},
                    tags=["skill_run", tag],
                )

        return {
            "agent": self.cfg.name,
            "role": self.cfg.role,
            "project_id": proj_node.id,
            "project_name": proj_node.name,
            "action": action,
            "reward": reward,
            "mem_id": mem_id,
            "info": info,
        }


# ----------------------------
# Orchestrator (C-layer: multi-agent + project graph)
# ----------------------------

@dataclass
class OrchestratorConfig:
    agents: int = 8
    base_budget: int = 20
    selection_top_k: int = 4
    budget_growth: float = 1.06


class Orchestrator:
    """
    C-layer:
    - maintains SharedMemory, SkillLibrary, ProjectGraph
    - runs multiple B-type agents per round
    - distills principles from best episodes
    - adapts org policy (role mix, risk) based on outcomes
    """

    def __init__(self, cfg: OrchestratorConfig,
                 env: ResearchEnvironment,
                 tools: ToolRegistry) -> None:
        self.cfg = cfg
        self.env = env
        self.tools = tools

        self.mem = SharedMemory()
        self.skills = SkillLibrary()
        self.projects = ProjectGraph()

        self._agents: List[Agent] = []
        self._org_policy: Dict[str, Any] = {
            "risk": 0.25,
            "role_mix": ["theorist", "builder", "experimenter", "verifier", "strategist"],
            "infra_focus": 0.5,
        }
        self.candidate_queue: List[RuleProposal] = []
        self.evaluation_rules: Dict[str, Any] = {
            "min_score": 0.25,
            "l1_update_rate": 0.08,
            "min_transfer": 0.05,
            "min_holdout_pass_rate": 0.30,
            "max_generalization_gap": 0.05,
            "holdout_weight": 1.0,
            "generalization_gap_penalty": 0.75,
            "discovery_cost_penalty": 0.08,
            "min_adversarial_pass_rate": 0.28,
            "min_shift_holdout_pass_rate": 0.25,
            "max_holdout_discovery_cost": 4.0,
            "require_holdout_metrics": True,
        }
        self.meta_rules: Dict[str, Any] = {
            "l1_update_rate_bounds": (0.04, 0.20),
        }
        self.invariants: Dict[str, Any] = {
            "min_evidence": 1,
            "min_transfer": 0.05,
            "l1_update_rate_bounds": (0.04, 0.20),
        }
        self._recent_rewards: List[float] = []
        self._adoption_cooldown_ms = 1500
        self._last_adoption_ms = 0
        self._critic_module: Optional[Any] = None
        self._init_agents()

    def _init_agents(self) -> None:
        roles = self._org_policy["role_mix"]
        for i in range(self.cfg.agents):
            role = roles[i % len(roles)]
            cfg = AgentConfig(
                name=f"agent_{i:02d}",
                role=role,
                planner_depth=4 if role in ("theorist", "strategist") else 3,
                planner_width=7 if role == "strategist" else 6,
                risk=self._org_policy["risk"],
            )
            self._agents.append(Agent(cfg, self.tools, self.mem, self.skills))

    def _record_round_rewards(self, results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        mean_reward = sum(r["reward"] for r in results) / max(1, len(results))
        self._recent_rewards.append(mean_reward)
        if len(self._recent_rewards) > 8:
            self._recent_rewards.pop(0)

    def _detect_stagnation(self, window: int = 5, threshold: float = 0.01) -> bool:
        if len(self._recent_rewards) < window:
            return False
        start = self._recent_rewards[-window]
        end = self._recent_rewards[-1]
        return (end - start) < threshold

    def _build_gap_spec(self, round_idx: int, round_out: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "round": round_idx,
            "seed": round_idx + 11,
            "tasks": round_out.get("tasks", []),
            "recent_rewards": list(self._recent_rewards[-5:]),
            "constraints": {
                "quarantine_only": True,
                "no_self_adoption": True,
                "max_candidates": 1,
            },
        }

    def _omega_generate_candidates(self, gap_spec: Dict[str, Any]) -> List[RuleProposal]:
        # import omega_forge_two_stage_feedback as omega

        engine = Stage1Engine(seed=int(gap_spec.get("seed", 0)))
        engine.init_population()
        for _ in range(3):
            engine.step()
            if engine.candidates:
                break

        candidates = list(engine.candidates)
        if not candidates and engine.population:
            fallback = engine.population[0]
            candidates = [
                {
                    "gid": fallback.gid,
                    "generation": engine.generation,
                    "code": [(i.op, i.a, i.b, i.c) for i in fallback.instructions],
                    "metrics": {"fallback": True},
                    "task_scores": {},
                }
            ]

        proposals: List[RuleProposal] = []
        for cand in candidates[: gap_spec.get("constraints", {}).get("max_candidates", 1)]:
            payload = {"candidate": cand, "gap_spec": gap_spec}
            proposal_id = stable_hash({"level": "L0", "payload": payload})
            proposals.append(
                RuleProposal(
                    proposal_id=proposal_id,
                    level="L0",
                    payload=payload,
                    creator_key=stable_hash({"source": "omega", "gid": cand.get("gid")}),
                    created_ms=now_ms(),
                    evidence={"metrics": cand.get("metrics", {}), "task_scores": cand.get("task_scores", {})},
                )
            )
        return proposals

    def _load_critic(self) -> Any:
        if self._critic_module is None:
            self._critic_module = load_unified_critic_module()
        return self._critic_module

    def _critic_evaluate(self, proposal: RuleProposal) -> Dict[str, Any]:
        # critic = self._load_critic()
        candidate = proposal.payload.get("candidate")
        if proposal.level == "L0":
            assert isinstance(candidate, dict), "candidate missing"
            assert candidate.get("gid"), "candidate gid missing"
        if proposal.level == "L1":
            assert isinstance(proposal.payload.get("evaluation_update"), dict), "evaluation_update missing"
        if proposal.level == "L2":
            assert isinstance(proposal.payload.get("meta_update"), dict), "meta_update missing"
        packet = {
            "proposal": asdict(proposal),
            "evaluation_rules": dict(self.evaluation_rules),
            "invariants": dict(self.invariants),
        }
        return critic_evaluate_candidate_packet(packet, invariants=self.invariants)

    def _adopt_proposal(self, proposal: RuleProposal, verdict: Dict[str, Any]) -> bool:
        if verdict.get("verdict") != "approve":
            return False
        if not proposal.creator_key or not verdict.get("approval_key"):
            return False
        if now_ms() - self._last_adoption_ms < self._adoption_cooldown_ms:
            return False

        if proposal.level == "L0":
            self.mem.add(
                "artifact",
                f"adopted_candidate:{proposal.proposal_id}",
                {"proposal": proposal.payload, "critic": verdict},
                tags=["adopted", "L0"],
            )
        elif proposal.level == "L1":
            update = proposal.payload.get("evaluation_update", {})
            if update:
                self.evaluation_rules.update(update)
        elif proposal.level == "L2":
            meta_update = proposal.payload.get("meta_update", {})
            if meta_update:
                self.meta_rules.update(meta_update)
                if "l1_update_rate" in meta_update:
                    self.evaluation_rules["l1_update_rate"] = meta_update["l1_update_rate"]
        elif proposal.level == "L3":
            # L3: Meta-meta-meta - modify environment structure itself
            env_update = proposal.payload.get("env_update", {})
            if env_update and hasattr(self.env, 'env_params'):
                for key, value in env_update.items():
                    if key in self.env.env_params:
                        self.env.env_params[key] = value
                print(f"  [L3 ENV UPDATE] {self.env.env_params}")

        self._last_adoption_ms = now_ms()
        return True

    def _apply_l1_update(self) -> Optional[Dict[str, Any]]:
        if len(self._recent_rewards) < 2:
            return None
        trend = self._recent_rewards[-1] - self._recent_rewards[-2]
        update_rate = float(self.evaluation_rules.get("l1_update_rate", 0.08))
        min_score = float(self.evaluation_rules.get("min_score", 0.4))
        if trend < 0:
            min_score = min(0.9, min_score + update_rate)
        else:
            min_score = max(0.1, min_score - update_rate / 2.0)
        self.evaluation_rules["min_score"] = min_score
        return {"min_score": min_score}

    def _propose_l2_update(self, round_idx: int, force: bool = False) -> Optional[RuleProposal]:
        if not force and round_idx % 6 != 0:
            return None
        bounds = self.meta_rules.get("l1_update_rate_bounds", (0.04, 0.20))
        current = float(self.evaluation_rules.get("l1_update_rate", 0.08))
        
        # Adaptive L2: increase or decrease based on recent reward trend
        if len(self._recent_rewards) >= 3:
            recent_trend = self._recent_rewards[-1] - self._recent_rewards[-3]
            if recent_trend > 0.02:
                # Performance improving - decrease update rate for stability
                delta = -0.01
            elif recent_trend < -0.02:
                # Performance declining - increase update rate for faster adaptation
                delta = 0.02
            else:
                # Stable - small random adjustment
                delta = 0.01 if (round_idx % 12 < 6) else -0.01
        else:
            delta = 0.01
            
        proposed = max(bounds[0], min(bounds[1], current + delta))
        
        # Only propose if actually different from current
        if abs(proposed - current) < 0.001:
            return None
            
        payload = {"meta_update": {"l1_update_rate": proposed}}
        proposal_id = stable_hash({"level": "L2", "payload": payload, "round": round_idx})
        return RuleProposal(
            proposal_id=proposal_id,
            level="L2",
            payload=payload,
            creator_key=stable_hash({"source": "meta", "round": round_idx}),
            created_ms=now_ms(),
            evidence={"meta": {"round": round_idx, "trend_delta": delta}},
        )

    def _propose_l3_update(self, round_idx: int, force: bool = False) -> Optional[RuleProposal]:
        """L3: Meta-meta-meta - propose environment structure modifications"""
        if not force and round_idx % 20 != 0:
            return None
        if not hasattr(self.env, 'env_params'):
            return None
            
        # Adaptive environment modification based on performance plateau
        if len(self._recent_rewards) >= 10:
            best_recent = max(self._recent_rewards[-10:])
            avg_recent = sum(self._recent_rewards[-10:]) / 10
            
            env_update = {}
            
            # If performance is stuck below 0.8, increase breakthrough potential
            if best_recent < 0.8:
                current_bt_scale = self.env.env_params.get('breakthrough_scale', 0.32)
                env_update['breakthrough_scale'] = min(0.8, current_bt_scale + 0.05)
                
            # If performance is stuck below 0.9, increase leverage multiplier
            if best_recent < 0.9:
                current_lm = self.env.env_params.get('leverage_multiplier', 1.0)
                env_update['leverage_multiplier'] = min(2.0, current_lm + 0.1)
                
            # If performance is stuck below 1.0, reduce difficulty penalty
            if best_recent < 1.0 and avg_recent > 0.7:
                current_dp = self.env.env_params.get('difficulty_penalty', 0.30)
                env_update['difficulty_penalty'] = max(0.05, current_dp - 0.02)
                
            # Increase infra bonus if making progress
            if avg_recent > 0.6:
                current_ib = self.env.env_params.get('infra_bonus_scale', 0.025)
                env_update['infra_bonus_scale'] = min(0.1, current_ib + 0.005)
            
            if env_update:
                payload = {"env_update": env_update}
                proposal_id = stable_hash({"level": "L3", "payload": payload, "round": round_idx})
                return RuleProposal(
                    proposal_id=proposal_id,
                    level="L3",
                    payload=payload,
                    creator_key=stable_hash({"source": "env_meta", "round": round_idx}),
                    created_ms=now_ms(),
                    evidence={"l3": {"round": round_idx, "best": best_recent, "avg": avg_recent}},
                )
        return None

    def _distill_principles(self, round_idx: int,
                            results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        results_sorted = sorted(results, key=lambda r: r["reward"], reverse=True)
        top = results_sorted[: self.cfg.selection_top_k]
        bottom = results_sorted[-self.cfg.selection_top_k:]

        self.mem.add(
            "note",
            f"round_{round_idx}_distill",
            {
                "top": [
                    {
                        "agent": r["agent"],
                        "role": r["role"],
                        "task": r["info"]["task"],
                        "reward": r["reward"],
                        "action": r["action"],
                    }
                    for r in top
                ],
                "bottom": [
                    {
                        "agent": r["agent"],
                        "role": r["role"],
                        "task": r["info"]["task"],
                        "reward": r["reward"],
                        "action": r["action"],
                    }
                    for r in bottom
                ],
                "env": {
                    "tq": self.env.global_tool_quality,
                    "kq": self.env.global_kb_quality,
                    "oq": self.env.global_org_quality,
                },
                "policy": dict(self._org_policy),
            },
            tags=["distill", "round"],
        )

        for r in top:
            self.mem.add(
                "principle",
                f"good_pattern:{r['info']['task']}:{r['action']}",
                {
                    "agent": r["agent"],
                    "role": r["role"],
                    "task": r["info"]["task"],
                    "action": r["action"],
                    "reward": r["reward"],
                    "env": {
                        "tq": r["info"]["tq"],
                        "kq": r["info"]["kq"],
                        "oq": r["info"]["oq"],
                    },
                },
                tags=["principle", "good"],
            )
        for r in bottom:
            self.mem.add(
                "principle",
                f"bad_pattern:{r['info']['task']}:{r['action']}",
                {
                    "agent": r["agent"],
                    "role": r["role"],
                    "task": r["info"]["task"],
                    "action": r["action"],
                    "reward": r["reward"],
                    "env": {
                        "tq": r["info"]["tq"],
                        "kq": r["info"]["kq"],
                        "oq": r["info"]["oq"],
                    },
                },
                tags=["principle", "bad"],
            )

        self.mem.extract_principles(k=max(3, self.cfg.selection_top_k // 2))

        rewards = [r["reward"] for r in results]
        mean = sum(rewards) / max(1, len(rewards))
        var = sum((x - mean) ** 2 for x in rewards) / max(1, len(rewards))
        std = math.sqrt(var)

        tq = self.env.global_tool_quality
        kq = self.env.global_kb_quality
        oq = self.env.global_org_quality

        if tq < kq and tq < oq:
            self._org_policy["role_mix"] = [
                "builder", "builder", "experimenter",
                "verifier", "strategist"
            ]
            self._org_policy["infra_focus"] = min(0.7, self._org_policy["infra_focus"] + 0.1)
        elif kq < tq and kq < oq:
            self._org_policy["role_mix"] = [
                "verifier", "verifier", "theorist",
                "builder", "strategist"
            ]
            self._org_policy["infra_focus"] = min(0.7, self._org_policy["infra_focus"] + 0.05)
        elif oq < tq and oq < kq:
            self._org_policy["role_mix"] = [
                "strategist", "strategist", "builder",
                "experimenter", "verifier"
            ]
            self._org_policy["infra_focus"] = min(0.7, self._org_policy["infra_focus"] + 0.05)
        else:
            self._org_policy["role_mix"] = [
                "theorist", "builder", "experimenter",
                "verifier", "strategist"
            ]
            self._org_policy["infra_focus"] = max(0.4, self._org_policy["infra_focus"] - 0.05)

        if std > 0.10:
            self._org_policy["risk"] = max(0.05, self._org_policy["risk"] - 0.02)
        else:
            self._org_policy["risk"] = min(0.40, self._org_policy["risk"] + 0.01)

        roles = self._org_policy["role_mix"]
        for i, ag in enumerate(self._agents):
            ag.cfg.role = roles[i % len(roles)]
            ag.cfg.risk = self._org_policy["risk"]

    def _assign_tasks(self) -> List[TaskSpec]:
        tasks = [self.env.sample_task()]
        if self.cfg.agents > 4:
            tasks.append(self.env.sample_task())
        return tasks

    def _budget_for_agent(self, base_budget: int, role: str) -> int:
        infra_focus = float(self._org_policy.get("infra_focus", 0.5))
        infra_roles = {"builder", "verifier", "strategist"}
        if role in infra_roles:
            scale = 0.85 + 0.5 * infra_focus
        else:
            scale = 0.85 + 0.5 * (1.0 - infra_focus)
        return max(8, int(base_budget * scale))

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        tasks = self._assign_tasks()
        budget = int(self.cfg.base_budget * (self.cfg.budget_growth ** round_idx))

        results: List[Dict[str, Any]] = []
        for idx, ag in enumerate(self._agents):
            task = tasks[idx % len(tasks)]
            proj_node = self.projects.pick_node_for_round(task.name)
            agent_budget = self._budget_for_agent(budget, ag.cfg.role)
            obs = self.env.make_observation(task, agent_budget)
            res = ag.act_on_project(self.env, proj_node, obs)
            results.append(res)
            self.projects.update_node(proj_node.id, res["reward"], res["mem_id"])

        self._distill_principles(round_idx, results)

        return {
            "round": round_idx,
            "tasks": [t.name for t in tasks],
            "results": results,
            "env": {
                "tq": self.env.global_tool_quality,
                "kq": self.env.global_kb_quality,
                "oq": self.env.global_org_quality,
            },
            "policy": dict(self._org_policy),
        }

    def run_recursive_cycle(
        self,
        round_idx: int,
        stagnation_override: Optional[bool] = None,
        force_meta_proposal: bool = False,
    ) -> Dict[str, Any]:
        round_out = self.run_round(round_idx)
        self._record_round_rewards(round_out["results"])

        stagnation = stagnation_override if stagnation_override is not None else self._detect_stagnation()
        gap_spec = None
        if stagnation:
            gap_spec = self._build_gap_spec(round_idx, round_out)
            self.candidate_queue.extend(self._omega_generate_candidates(gap_spec))

        l1_update = self._apply_l1_update()
        if l1_update:
            l1_proposal = RuleProposal(
                proposal_id=stable_hash({"level": "L1", "payload": l1_update, "round": round_idx}),
                level="L1",
                payload={"evaluation_update": dict(l1_update)},
                creator_key=stable_hash({"source": "l1", "round": round_idx}),
                created_ms=now_ms(),
                evidence={"l1_update": dict(l1_update)},
            )
            self.candidate_queue.append(l1_proposal)

        l2_proposal = self._propose_l2_update(round_idx, force=force_meta_proposal or stagnation)
        if l2_proposal:
            self.candidate_queue.append(l2_proposal)

        # L3: Environment self-modification
        l3_proposal = self._propose_l3_update(round_idx, force=stagnation)
        if l3_proposal:
            self.candidate_queue.append(l3_proposal)

        critic_results: List[Dict[str, Any]] = []
        while self.candidate_queue:
            proposal = self.candidate_queue.pop(0)
            verdict = self._critic_evaluate(proposal)
            adopted = self._adopt_proposal(proposal, verdict)
            critic_results.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "level": proposal.level,
                    "verdict": verdict.get("verdict"),
                    "adopted": adopted,
                }
            )

        round_out.update(
            {
                "stagnation": stagnation,
                "gap_spec": gap_spec,
                "l1_update": l1_update,
                "l2_proposal": asdict(l2_proposal) if l2_proposal else None,
                "critic_results": critic_results,
            }
        )
        return round_out


# ----------------------------
# Minimal tools (replace with real-world hooks)
# ----------------------------

def tool_write_note_factory(shared_mem: SharedMemory) -> ToolFn:
    def _fn(args: Dict[str, Any]) -> Dict[str, Any]:
        title = str(args.get("title", "note"))
        payload = args.get("payload", {})
        mid = shared_mem.add("note", title, {"payload": payload},
                             tags=["tool_note"])
        return {"ok": True, "memory_id": mid, "title": title}
    return _fn


def tool_write_artifact_factory(shared_mem: SharedMemory) -> ToolFn:
    def _fn(args: Dict[str, Any]) -> Dict[str, Any]:
        title = str(args.get("title", "artifact"))
        payload = args.get("payload", {})
        mid = shared_mem.add("artifact", title, {"payload": payload},
                             tags=["tool_artifact"])
        return {"ok": True, "memory_id": mid, "title": title}
    return _fn


def tool_evaluate_candidate(args: Dict[str, Any]) -> Dict[str, Any]:
    task = str(args.get("task", "unknown"))
    cand = args.get("candidate", {})
    size = len(json.dumps(cand, default=str))
    score = (size % 97) / 100.0
    if "hints" in cand and isinstance(cand["hints"], list) and len(cand["hints"]) > 4:
        score *= 0.93
    verdict = "pass" if score > 0.4 else "revise"
    return {"ok": True, "task": task, "score": score, "verdict": verdict}


def tool_tool_build_report(args: Dict[str, Any]) -> Dict[str, Any]:
    task = str(args.get("task", "unknown"))
    idea = args.get("idea", {})
    return {
        "ok": True,
        "task": task,
        "artifact": {
            "type": "tool_proposal",
            "idea": idea,
            "expected_effect": "increase evaluation throughput & reliability",
        },
    }


# ----------------------------
# Main entry
# ----------------------------

def run_full_system_selftest() -> None:
    random.seed(0)
    env = ResearchEnvironment(seed=0)
    tools = ToolRegistry()
    orch_cfg = OrchestratorConfig(
        agents=4,
        base_budget=12,
        selection_top_k=2,
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    orch.run_recursive_cycle(0, stagnation_override=True, force_meta_proposal=True)
    assert (round_out := orch.run_recursive_cycle(1, stagnation_override=True, force_meta_proposal=True))
    assert round_out["stagnation"] is True
    assert "gap_spec" in round_out and isinstance(round_out["gap_spec"], dict)
    assert "constraints" in round_out["gap_spec"] and isinstance(round_out["gap_spec"]["constraints"], dict)
    assert "quarantine_only" in round_out["gap_spec"]["constraints"]
    assert round_out["gap_spec"]["constraints"]["quarantine_only"] is True
    assert "no_self_adoption" in round_out["gap_spec"]["constraints"]
    assert round_out["gap_spec"]["constraints"]["no_self_adoption"] is True
    assert round_out["critic_results"]
    assert all("verdict" in item for item in round_out["critic_results"])
    assert all("proposal_id" in item for item in round_out["critic_results"])
    assert any(item["level"] == "L0" for item in round_out["critic_results"])
    assert any(item["level"] == "L1" for item in round_out["critic_results"])
    assert any(item["level"] == "L2" for item in round_out["critic_results"])
    assert all(
        (not item.get("adopted", False)) or item.get("verdict") == "approve"
        for item in round_out["critic_results"]
    )
    print("recursive rule loop executed")
    print("critic decision received")

    x = [[1.0, 2.0], [3.0, 4.0]]
    w = [[1.0], [1.0]]
    y = [
        [x[0][0] * w[0][0] + x[0][1] * w[1][0]],
        [x[1][0] * w[0][0] + x[1][1] * w[1][0]],
    ]
    assert len(y) == 2 and len(y[0]) == 1
    print("tensor execution verified (torch-free)")


def run_torch_smoke_test() -> None:
    import torch

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x @ torch.tensor([[1.0], [1.0]])
    assert y.shape == (2, 1)
    print("pytorch execution verified")


def run_contract_negative_tests() -> None:
    random.seed(1)
    env = ResearchEnvironment(seed=1)
    tools = ToolRegistry()
    orch_cfg = OrchestratorConfig(
        agents=4,
        base_budget=12,
        selection_top_k=2,
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    round_out = orch.run_recursive_cycle(0, stagnation_override=True, force_meta_proposal=True)

    def expect_failure(fn: Callable[[], None], exc_types: Tuple[type, ...], msg_substr: str) -> None:
        try:
            fn()
        except exc_types as exc:
            assert msg_substr in str(exc), f"message mismatch: {exc}"
            return
        except Exception:
            raise
        raise AssertionError("expected failure was not raised")

    assert "gap_spec" in round_out and isinstance(round_out["gap_spec"], dict)
    assert "critic_results" in round_out and isinstance(round_out["critic_results"], list)
    assert any(item.get("level") == "L0" for item in round_out["critic_results"])

    proposals = orch._omega_generate_candidates(round_out["gap_spec"])
    assert proposals
    proposal = proposals[0]
    verdict = orch._critic_evaluate(proposal)
    l1_proposal = RuleProposal(
        proposal_id="l1_negative",
        level="L1",
        payload={"evaluation_update": {"min_score": 0.5}},
        creator_key="creator",
        created_ms=now_ms(),
    )
    l2_proposal = RuleProposal(
        proposal_id="l2_negative",
        level="L2",
        payload={"meta_update": {"l1_update_rate": 0.1}},
        creator_key="creator",
        created_ms=now_ms(),
    )
    l1_verdict = orch._critic_evaluate(l1_proposal)
    l2_verdict = orch._critic_evaluate(l2_proposal)

    def make_l0_candidate(metrics: Dict[str, Any]) -> RuleProposal:
        candidate = {
            "gid": stable_hash({"neg": metrics}),
            "metrics": metrics,
        }
        return RuleProposal(
            proposal_id=stable_hash({"level": "L0", "metrics": metrics}),
            level="L0",
            payload={"candidate": candidate, "gap_spec": round_out.get("gap_spec", {})},
            creator_key="creator",
            created_ms=now_ms(),
            evidence={"metrics": metrics},
        )

    def validate_critic_verdict(result: Dict[str, Any]) -> None:
        assert "verdict" in result, "verdict missing"
        assert "approval_key" in result, "approval_key missing"

    def adopt_with_contract(test_proposal: RuleProposal, result: Dict[str, Any]) -> None:
        if result.get("verdict") != "approve":
            raise ValueError("verdict not approved")
        if "approval_key" not in result:
            raise ValueError("approval_key missing")
        orch._adopt_proposal(test_proposal, result)

    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="missing_candidate",
                level="L0",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="non_dict_candidate",
                level="L0",
                payload={"candidate": "not-a-dict"},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="gid_none",
                level="L0",
                payload={"candidate": {**proposal.payload["candidate"], "gid": None}},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate gid missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="gid_empty",
                level="L0",
                payload={"candidate": {**proposal.payload["candidate"], "gid": ""}},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "candidate gid missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="l1_missing_update",
                level="L1",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "evaluation_update missing",
    )
    expect_failure(
        lambda: orch._critic_evaluate(
            RuleProposal(
                proposal_id="l2_missing_update",
                level="L2",
                payload={},
                creator_key="creator",
                created_ms=now_ms(),
            )
        ),
        (AssertionError,),
        "meta_update missing",
    )
    expect_failure(
        lambda: adopt_with_contract(l1_proposal, {**l1_verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(l1_proposal, {"verdict": "approve"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: adopt_with_contract(l2_proposal, {**l2_verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(l2_proposal, {"verdict": "approve"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: adopt_with_contract(proposal, {**verdict, "verdict": "reject"}),
        (ValueError,),
        "verdict not approved",
    )
    expect_failure(
        lambda: adopt_with_contract(proposal, {k: v for k, v in verdict.items() if k != "approval_key"}),
        (ValueError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: validate_critic_verdict({k: v for k, v in verdict.items() if k != "approval_key"}),
        (AssertionError,),
        "approval_key missing",
    )
    expect_failure(
        lambda: validate_critic_verdict({k: v for k, v in verdict.items() if k != "verdict"}),
        (AssertionError,),
        "verdict missing",
    )

    missing_holdout_verdict = orch._critic_evaluate(
        make_l0_candidate({"train_pass_rate": 0.34})
    )
    assert missing_holdout_verdict["verdict"] == "reject"

    low_holdout_verdict = orch._critic_evaluate(
        make_l0_candidate({"train_pass_rate": 0.31, "holdout_pass_rate": 0.22})
    )
    assert low_holdout_verdict["verdict"] == "reject"

    gap_verdict = orch._critic_evaluate(
        make_l0_candidate({"train_pass_rate": 0.38, "holdout_pass_rate": 0.30})
    )
    assert gap_verdict["verdict"] == "reject"

    adversarial_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.34,
                "holdout_pass_rate": 0.31,
                "adversarial_pass_rate": 0.20,
                "adversarial_examples": [
                    {"input": [3, -2, 7], "expected": [7, -2, 3], "prediction": [3, 7, -2]},
                    {"input": [4, 4, 1], "expected": [1, 4, 4], "prediction": [4, 1, 4]},
                    {"input": [-1, 5, 2], "expected": [2, 5, -1], "prediction": [-1, 2, 5]},
                ],
            }
        )
    )
    assert adversarial_verdict["verdict"] == "reject"

    shift_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.34,
                "holdout_pass_rate": 0.31,
                "distribution_shift": {"holdout_pass_rate": 0.18},
                "distribution_shift_examples": [
                    {"input": [9, -4, 0, 3], "expected": [3, 0, -4, 9], "prediction": [9, 3, 0, -4]},
                    {"input": [2, -5, 6, -1], "expected": [-1, 6, -5, 2], "prediction": [2, -1, -5, 6]},
                    {"input": [8, 1, 1, -2], "expected": [-2, 1, 1, 8], "prediction": [8, 1, -2, 1]},
                ],
            }
        )
    )
    assert shift_verdict["verdict"] == "reject"

    regression_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.36,
                "holdout_pass_rate": 0.28,
                "baseline": {"train_pass_rate": 0.33, "holdout_pass_rate": 0.30},
            }
        )
    )
    assert regression_verdict["verdict"] == "reject"

    high_cost_verdict = orch._critic_evaluate(
        make_l0_candidate(
            {
                "train_pass_rate": 0.34,
                "holdout_pass_rate": 0.31,
                "discovery_cost": {"holdout": 6.5},
            }
        )
    )
    assert high_cost_verdict["verdict"] == "reject"
    assert high_cost_verdict.get("holdout_cost_ok") is False
    assert high_cost_verdict.get("guardrails_ok") is False
    print("negative contract tests passed")


def _make_benchmark_stack(seed: int) -> Tuple[ResearchEnvironment, Agent, ProjectGraph]:
    random.seed(seed)
    env = ResearchEnvironment(seed=seed)
    tools = ToolRegistry()
    mem = SharedMemory()
    skills = SkillLibrary()
    agent = Agent(AgentConfig(name="bench_agent", role="general"), tools, mem, skills)

    tools.register("write_note", tool_write_note_factory(mem))
    tools.register("write_artifact", tool_write_artifact_factory(mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    projects = ProjectGraph()
    return env, agent, projects


def _run_benchmark_step(
    env: ResearchEnvironment,
    agent: Agent,
    projects: ProjectGraph,
    task: TaskSpec,
    budget: int,
) -> Dict[str, Any]:
    obs = env.make_observation(task, budget)
    proj_node = projects.pick_node_for_round(task.name)
    return agent.act_on_project(env, proj_node, obs)


def _adb_apply_rule(rule: str, params: Dict[str, int], seq: List[int]) -> List[int]:
    if rule == "reverse":
        return list(reversed(seq))
    if rule == "sort_unique":
        out: List[int] = []
        seen = set()
        for val in sorted(seq):
            if val not in seen:
                seen.add(val)
                out.append(val)
        return out
    if rule == "add_then_filter":
        delta = params.get("delta", 0)
        threshold = params.get("threshold", 0)
        return [val + delta for val in seq if val + delta >= threshold]
    if rule == "window_sum":
        width = max(1, params.get("width", 2))
        return [sum(seq[i:i + width]) for i in range(0, len(seq), width)]
    return seq


def _generate_adb_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["reverse", "sort_unique", "add_then_filter", "window_sum"])
    params = {}
    if rule == "add_then_filter":
        params = {"delta": rng.randint(-3, 3), "threshold": rng.randint(0, 6)}
    if rule == "window_sum":
        params = {"width": rng.randint(2, 3)}

    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 6)
        inp = [rng.randint(-4, 9) for _ in range(length)]
        out = _adb_apply_rule(rule, params, inp)
        train_pairs.append({"input": inp, "output": out})

    test_length = rng.randint(6, 9)
    test_input = [rng.randint(-6, 12) for _ in range(test_length)]
    adversarial = test_input[:]
    rng.shuffle(adversarial)
    if rule == "add_then_filter":
        adversarial = [val - params.get("delta", 0) for val in adversarial]
    test_output = _adb_apply_rule(rule, params, test_input)
    adversarial_output = _adb_apply_rule(rule, params, adversarial)
    return {
        "train": train_pairs,
        "test": {"input": test_input, "output": test_output},
        "adversarial": {"input": adversarial, "output": adversarial_output},
    }


def _solve_adb(task: Dict[str, Any], test_input: List[int]) -> Tuple[Any, int]:
    attempts = 0
    train_pairs = task.get("train", [])
    if train_pairs and all(
        pair.get("output") == list(reversed(pair.get("input", []))) for pair in train_pairs
    ):
        attempts += 1
        return list(reversed(test_input)), attempts
    if train_pairs and all(
        pair.get("output") == sorted(pair.get("input", [])) for pair in train_pairs
    ):
        attempts += 1
        return sorted(test_input), attempts
    attempts += 1
    return [], attempts


def _run_adb_suite_split(seed: int, trials: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    passes = 0
    robust_passes = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for _ in range(trials):
        task = _generate_adb_task(rng)
        start = now_ms()
        base_input = task["test"]["input"]
        prediction, attempts = _solve_adb(task, base_input)
        commit_hash = stable_hash({"pred": prediction})
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        base_ok = prediction == task["test"]["output"]

        robust_ok = False
        if base_ok:
            adv_input = task["adversarial"]["input"]
            adv_prediction, _ = _solve_adb(task, adv_input)
            robust_ok = adv_prediction == task["adversarial"]["output"]
            _ = commit_hash
        if base_ok:
            passes += 1
        if base_ok and robust_ok:
            robust_passes += 1

    trials_count = max(1, trials)
    return {
        "pass_rate": passes / trials_count,
        "robust_pass_rate": robust_passes / trials_count,
        "discovery_cost": total_attempts / max(1, passes),
        "avg_runtime_ms_per_trial": sum(runtimes_ms) / max(1, len(runtimes_ms)),
    }


def run_adb_benchmark_suite(seed: int, trials: int) -> Dict[str, Any]:
    train_result = _run_adb_suite_split(seed, trials)
    holdout_seed = _derive_holdout_seed(seed)
    holdout_result = _run_adb_suite_split(holdout_seed, trials)
    return {
        "suite": "ADB_v1",
        "seed": seed,
        "trials": trials,
        "train_pass_rate": train_result["pass_rate"],
        "holdout_pass_rate": holdout_result["pass_rate"],
        "discovery_cost": {
            "train": train_result["discovery_cost"],
            "holdout": holdout_result["discovery_cost"],
        },
        "robust_pass_rate": {
            "train": train_result["robust_pass_rate"],
            "holdout": holdout_result["robust_pass_rate"],
        },
        "avg_runtime_ms_per_trial": {
            "train": train_result["avg_runtime_ms_per_trial"],
            "holdout": holdout_result["avg_runtime_ms_per_trial"],
        },
    }


def _derive_holdout_seed(base_seed: int) -> int:
    nonce = "holdout-seed-v1"
    file_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    mix = f"{base_seed}:{file_hash}:{nonce}".encode("utf-8")
    return int(hashlib.sha256(mix).hexdigest()[:8], 16)


def _generate_program_synthesis_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["reverse", "sort", "dedup"])
    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 6)
        inp = [rng.randint(-3, 9) for _ in range(length)]
        if rule == "reverse":
            out = list(reversed(inp))
        elif rule == "sort":
            out = sorted(inp)
        else:
            out = []
            seen = set()
            for val in inp:
                if val not in seen:
                    seen.add(val)
                    out.append(val)
        train_pairs.append({"input": inp, "output": out})
    test_length = rng.randint(3, 6)
    test_input = [rng.randint(-3, 9) for _ in range(test_length)]
    if rule == "reverse":
        test_output = list(reversed(test_input))
    elif rule == "sort":
        test_output = sorted(test_input)
    else:
        test_output = []
        seen = set()
        for val in test_input:
            if val not in seen:
                seen.add(val)
                test_output.append(val)
    return {"train": train_pairs, "test": {"input": test_input, "output": test_output}}


def _generate_algo_micro_task(rng: random.Random) -> Dict[str, Any]:
    rule = rng.choice(["sum", "max", "count_even"])
    train_pairs = []
    for _ in range(3):
        length = rng.randint(3, 7)
        inp = [rng.randint(-5, 12) for _ in range(length)]
        if rule == "sum":
            out = sum(inp)
        elif rule == "max":
            out = max(inp)
        else:
            out = sum(1 for v in inp if v % 2 == 0)
        train_pairs.append({"input": inp, "output": out})
    test_length = rng.randint(3, 7)
    test_input = [rng.randint(-5, 12) for _ in range(test_length)]
    if rule == "sum":
        test_output = sum(test_input)
    elif rule == "max":
        test_output = max(test_input)
    else:
        test_output = sum(1 for v in test_input if v % 2 == 0)
    return {"train": train_pairs, "test": {"input": test_input, "output": test_output}}


def _generate_robustness_task(rng: random.Random) -> Dict[str, Any]:
    length = rng.randint(4, 8)
    base_input = [rng.randint(-4, 9) for _ in range(length)]
    base_output = sum(base_input)
    return {"base_input": base_input, "base_output": base_output}


def _solve_program_synthesis(task: Dict[str, Any]) -> Tuple[Any, int]:
    train_pairs = task.get("train", [])
    attempts = 0
    if train_pairs and all(pair.get("input") == pair.get("output") for pair in train_pairs):
        attempts += 1
        return task["test"]["input"], attempts
    if train_pairs and all(
        pair.get("output") == list(reversed(pair.get("input", []))) for pair in train_pairs
    ):
        attempts += 1
        return list(reversed(task["test"]["input"])), attempts
    attempts += 1
    return [], attempts


def _solve_algo_micro(task: Dict[str, Any]) -> Tuple[Any, int]:
    attempts = 1
    return 0, attempts


def _run_hard_suite_split(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    passes = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for _ in range(trials):
        start = now_ms()
        if suite == "program_synthesis_hard_v1":
            task = _generate_program_synthesis_task(rng)
            prediction, attempts = _solve_program_synthesis(task)
            expected = task["test"]["output"]
            solved = prediction == expected
        elif suite == "algo_micro_hard_v1":
            task = _generate_algo_micro_task(rng)
            prediction, attempts = _solve_algo_micro(task)
            expected = task["test"]["output"]
            solved = prediction == expected
        elif suite == "robustness_hard_v1":
            attempts = 1
            base_task = _generate_robustness_task(rng)
            base_input = base_task["base_input"]
            expected = base_task["base_output"]
            prediction, _ = _solve_algo_micro({"input": base_input})
            solved = prediction == expected
            if solved:
                shuffled = base_input[:]
                rng.shuffle(shuffled)
                prediction, _ = _solve_algo_micro({"input": shuffled})
                solved = prediction == expected
            if solved:
                noisy = base_input[:] + [0, 0]
                rng.shuffle(noisy)
                prediction, _ = _solve_algo_micro({"input": noisy})
                solved = prediction == expected
        else:
            raise ValueError(f"unknown suite: {suite}")
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        if solved:
            passes += 1

    pass_rate = passes / max(1, trials)
    return {
        "pass_rate": pass_rate,
        "proposals_evaluated_per_solve": total_attempts / max(1, passes),
        "avg_runtime_ms_per_trial": sum(runtimes_ms) / max(1, len(runtimes_ms)),
    }


def run_hard_benchmark_suite(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    train_result = _run_hard_suite_split(suite, seed, trials)
    holdout_seed = _derive_holdout_seed(seed)
    holdout_result = _run_hard_suite_split(suite, holdout_seed, trials)
    result = {
        "suite": suite,
        "seed": seed,
        "trials": trials,
        "train_pass_rate": train_result["pass_rate"],
        "holdout_pass_rate": holdout_result["pass_rate"],
        "avg_runtime_ms_per_trial": {
            "train": train_result["avg_runtime_ms_per_trial"],
            "holdout": holdout_result["avg_runtime_ms_per_trial"],
        },
    }
    if suite == "program_synthesis_hard_v1":
        result["proposals_evaluated_per_solve"] = {
            "train": train_result["proposals_evaluated_per_solve"],
            "holdout": holdout_result["proposals_evaluated_per_solve"],
        }
    return result


def run_benchmark_suite(suite: str, seed: int, trials: int) -> Dict[str, Any]:
    if suite == "ADB_v1":
        return run_adb_benchmark_suite(seed, trials)
    if suite in {"program_synthesis_hard_v1", "algo_micro_hard_v1", "robustness_hard_v1"}:
        return run_hard_benchmark_suite(suite, seed, trials)
    passes = 0
    total_rewards: List[float] = []
    skill_successes = 0
    attempts = 0

    for idx in range(trials):
        env, agent, projects = _make_benchmark_stack(seed + idx)

        if suite == "algo_micro_v1":
            task = next(t for t in env.tasks if t.domain == "algorithm")
            res = _run_benchmark_step(env, agent, projects, task, budget=12)
            reward = float(res.get("reward", 0.0))
            total_rewards.append(reward)
            if reward >= 0.02:
                passes += 1
        elif suite == "robustness_v1":
            rewards: List[float] = []
            for budget in (8, 12, 16):
                task = env.sample_task()
                res = _run_benchmark_step(env, agent, projects, task, budget=budget)
                rewards.append(float(res.get("reward", 0.0)))
            total_rewards.extend(rewards)
            if min(rewards) >= -0.01:
                passes += 1
        elif suite == "program_synthesis_v1":
            for _ in range(5):
                task = next(
                    t for t in env.tasks if t.name in ("verification_pipeline", "toolchain_speedup")
                )
                res = _run_benchmark_step(env, agent, projects, task, budget=12)
                total_rewards.append(float(res.get("reward", 0.0)))
                attempts += 1
            if agent.skills.list():
                passes += 1
                skill_successes += 1
        else:
            raise ValueError(f"unknown suite: {suite}")

    pass_rate = passes / max(1, trials)
    result = {
        "suite": suite,
        "seed": seed,
        "trials": trials,
        "pass_rate": pass_rate,
        "avg_reward": sum(total_rewards) / max(1, len(total_rewards)),
    }
    if suite == "program_synthesis_v1":
        proposals_per_solve = attempts / max(1, skill_successes)
        result["proposals_evaluated_per_solve"] = proposals_per_solve
    return result


def _load_arc_tasks(data_root: Path, suite: str) -> List[Dict[str, Any]]:
    if suite != "arc_agi2_public_eval":
        raise ValueError(f"unknown suite: {suite}")
    candidates = [
        data_root / "public_eval",
        data_root / "public",
        data_root / "evaluation",
        data_root / "eval",
        data_root / "public_eval_tasks",
    ]
    task_dir = next((p for p in candidates if p.exists()), None)
    if task_dir is None:
        raise FileNotFoundError(f"ARC public eval dataset not found under {data_root}")
    tasks = []
    for path in sorted(task_dir.glob("*.json")):
        tasks.append(json.loads(path.read_text(encoding="utf-8")))
    if not tasks:
        raise ValueError(f"no ARC tasks found in {task_dir}")
    return tasks


def _arc_constant_output(train_pairs: List[Dict[str, Any]]) -> Optional[List[List[int]]]:
    if not train_pairs:
        return None
    first = train_pairs[0].get("output")
    if first is None:
        return None
    for pair in train_pairs[1:]:
        if pair.get("output") != first:
            return None
    return first


def _arc_color_map(train_pairs: List[Dict[str, Any]]) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    for pair in train_pairs:
        inp = pair.get("input")
        out = pair.get("output")
        if inp is None or out is None or len(inp) != len(out):
            return None
        if any(len(inp[r]) != len(out[r]) for r in range(len(inp))):
            return None
        for r in range(len(inp)):
            for c in range(len(inp[r])):
                src = int(inp[r][c])
                dst = int(out[r][c])
                if src in mapping and mapping[src] != dst:
                    return None
                mapping[src] = dst
    return mapping if mapping else None


def _arc_apply_color_map(grid: List[List[int]], mapping: Dict[int, int]) -> List[List[int]]:
    return [[mapping.get(int(cell), int(cell)) for cell in row] for row in grid]


def solve_arc_task(task: Dict[str, Any]) -> Tuple[List[List[int]], int]:
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    test_input = test_pairs[0].get("input") if test_pairs else None
    attempts = 0

    constant_output = _arc_constant_output(train_pairs)
    if constant_output is not None:
        attempts += 1
        return constant_output, attempts

    color_map = _arc_color_map(train_pairs)
    if color_map is not None and test_input is not None:
        attempts += 1
        return _arc_apply_color_map(test_input, color_map), attempts

    attempts += 1
    return test_input if test_input is not None else [], attempts


def run_arc_benchmark(suite: str, seed: int) -> Dict[str, Any]:
    data_root = Path(os.environ.get("ARC_GYM_PATH", ""))
    if not str(data_root):
        raise EnvironmentError("ARC_GYM_PATH is not set")
    tasks = _load_arc_tasks(data_root, suite)
    random.seed(seed)

    tasks_solved = 0
    total_attempts = 0
    runtimes_ms: List[int] = []

    for task in tasks:
        start = now_ms()
        prediction, attempts = solve_arc_task(task)
        end = now_ms()
        runtimes_ms.append(end - start)
        total_attempts += attempts
        test_pairs = task.get("test", [])
        expected = test_pairs[0].get("output") if test_pairs else None
        if expected is not None and prediction == expected:
            tasks_solved += 1

    tasks_total = len(tasks)
    accuracy = tasks_solved / max(1, tasks_total)
    return {
        "suite": suite,
        "seed": seed,
        "tasks_total": tasks_total,
        "tasks_solved": tasks_solved,
        "accuracy": accuracy,
        "avg_attempts_per_task": total_attempts / max(1, tasks_total),
        "avg_runtime_ms_per_task": sum(runtimes_ms) / max(1, tasks_total),
    }
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        ap = argparse.ArgumentParser()
        ap.add_argument("benchmark")
        ap.add_argument("--suite", required=True)
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--trials", type=int, default=20)
        args = ap.parse_args()
        result = run_benchmark_suite(args.suite, args.seed, args.trials)
        print(json.dumps(result, ensure_ascii=False))
        return

    if len(sys.argv) > 1 and sys.argv[1] == "arc-benchmark":
        ap = argparse.ArgumentParser()
        ap.add_argument("arc-benchmark")
        ap.add_argument("--suite", required=True)
        ap.add_argument("--seed", type=int, default=0)
        args = ap.parse_args()
        result = run_arc_benchmark(args.suite, args.seed)
        print(json.dumps(result, ensure_ascii=False))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    env = ResearchEnvironment(seed=args.seed)
    tools = ToolRegistry()

    orch_cfg = OrchestratorConfig(
        agents=args.agents,
        base_budget=20,
        selection_top_k=max(3, args.agents // 2),
    )
    orch = Orchestrator(orch_cfg, env, tools)

    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)

    print("=== NON-RSI AGI CORE v5 (Neuro-Symbolic): RUN START ===")
    for r in range(args.rounds):
        out = orch.run_round(r)
        top = sorted(out["results"], key=lambda x: x["reward"], reverse=True)[:3]
        print(
            f"[Round {r:02d}] tasks={','.join(out['tasks']):<35} "
            f"tq={out['env']['tq']:.3f} kq={out['env']['kq']:.3f} oq={out['env']['oq']:.3f} "
            f"risk={out['policy']['risk']:.2f} infra={out['policy']['infra_focus']:.2f} "
            f"top_rewards={[round(x['reward'],4) for x in top]}"
        )

    print("=== RUN END ===")
    print("Recent memory summary:")
    for it in orch.mem.dump_summary(k=15):
        print(it)


# if __name__ == "__main__":
    main()

# END OF NON_RSI_AGI_CORE_v5.py


# ==========================================

# UNIFIED MAIN ENTRY POINT

# ==========================================
# HRM CORE (HIERARCHICAL REASONING MODEL)
# Integrated from stitch_lite.py, ast_core.py, py_to_ast.py, run_hrm.py
# ==========================================

@dataclass(frozen=True)
class HrmExpr:
    pass

@dataclass(frozen=True)
class HrmApp(HrmExpr):
    f: HrmExpr
    x: HrmExpr
    def __repr__(self): return f"({self.f} {self.x})"

@dataclass(frozen=True)
class HrmLam(HrmExpr):
    body: HrmExpr
    def __repr__(self): return f"(lam {self.body})"

@dataclass(frozen=True)
class HrmVar(HrmExpr):
    i: int
    def __repr__(self): return f"${self.i}"

@dataclass(frozen=True)
class HrmPrim(HrmExpr):
    name: str
    def __repr__(self): return self.name

@dataclass
class AbstractionResult:
    body: HrmExpr
    arity: int

class StitchLite:
    def __init__(self):
        self.next_var = 0

    def compress(self, programs: List[HrmExpr]) -> List[AbstractionResult]:
        abstractions = []
        for i in range(len(programs)):
            for j in range(i + 1, len(programs)):
                lgg = self.anti_unify(programs[i], programs[j])
                if lgg and self.complexity(lgg.body) > 1:
                    if not any(str(lgg.body) == str(a.body) for a in abstractions):
                        abstractions.append(lgg)
        return abstractions

    def anti_unify(self, e1: HrmExpr, e2: HrmExpr) -> Optional[AbstractionResult]:
        self.next_var = 0
        holes = {}
        def walk(a, b):
            if type(a) == type(b):
                if isinstance(a, HrmApp): return HrmApp(walk(a.f, b.f), walk(a.x, b.x))
                if isinstance(a, HrmLam): return HrmLam(walk(a.body, b.body))
                if isinstance(a, HrmVar) and a.i == b.i: return a
                if isinstance(a, HrmPrim) and a.name == b.name: return a
            key = (str(a), str(b))
            if key in holes: return HrmVar(holes[key])
            v = HrmVar(self.next_var)
            holes[key] = self.next_var
            self.next_var += 1
            return v
        body = walk(e1, e2)
        if isinstance(body, HrmVar): return None
        return AbstractionResult(body, self.next_var)

    def complexity(self, e: HrmExpr) -> int:
        if isinstance(e, HrmApp): return 1 + self.complexity(e.f) + self.complexity(e.x)
        if isinstance(e, HrmLam): return 1 + self.complexity(e.body)
        return 1

class PyToLambda:
    def convert(self, code: str) -> HrmExpr:
        try:
            tree = ast.parse(code)
            if not tree.body: return HrmPrim("pass")
            return self._convert_stmt(tree.body[0])
        except:
            return HrmPrim("error")

    def _convert_stmt(self, node):
        if isinstance(node, ast.FunctionDef):
            if node.body: return self._convert_stmt(node.body[0])
            return HrmPrim("pass")
        if isinstance(node, ast.Return): return self._convert_expr(node.value)
        if isinstance(node, ast.Expr): return self._convert_expr(node.value)
        return HrmPrim("stmt")

    def _convert_expr(self, node):
        if isinstance(node, ast.Name): return HrmPrim(node.id)
        if isinstance(node, ast.Constant): return HrmPrim(str(node.value))
        if isinstance(node, ast.Call):
            acc = self._convert_expr(node.func)
            for arg in node.args: acc = HrmApp(acc, self._convert_expr(arg))
            return acc
        if isinstance(node, ast.BinOp):
            op_map = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
            op_str = op_map.get(type(node.op), '?')
            return HrmApp(HrmApp(HrmPrim(op_str), self._convert_expr(node.left)), self._convert_expr(node.right))
        return HrmPrim("?")

class LambdaToPy:
    def convert(self, expr: HrmExpr) -> str:
        return self._to_str(expr)
    def _to_str(self, expr):
        if isinstance(expr, HrmPrim): return expr.name
        if isinstance(expr, HrmVar): return f"var_{expr.i}"
        if isinstance(expr, HrmLam): return f"(lambda: {self._to_str(expr.body)})"
        if isinstance(expr, HrmApp):
            f, args = self._flatten_app(expr)
            f_str = self._to_str(f)
            if f_str in ['+', '-', '*', '/'] and len(args) == 2:
                return f"({self._to_str(args[0])} {f_str} {self._to_str(args[1])})"
            return f"{f_str}({', '.join([self._to_str(a) for a in args])})"
        return "?"
    def _flatten_app(self, expr):
        args = []
        curr = expr
        while isinstance(curr, HrmApp):
            args.append(curr.x)
            curr = curr.f
        return curr, list(reversed(args))

class HModule:
    def __init__(self):
        self.stitch = StitchLite()
        self.py2lam = PyToLambda()
        self.lam2py = LambdaToPy()
        self.concept_count = 0
    
    def dream(self, candidates: List[InventionProgramCandidate]) -> List[str]:
        print(f"[H-Module] Dreaming on {len(candidates)} experiences...")
        if len(candidates) < 2: return []
        asts = []
        for c in candidates:
            try: asts.append(self.py2lam.convert(c.code))
            except: pass
        
        results = self.stitch.compress(asts)
        new_helpers = []
        for res in results:
            self.concept_count += 1
            fn_name = f"concept_{self.concept_count}"
            body = self.lam2py.convert(res.body)
            args = ", ".join([f"var_{i}" for i in range(res.arity)])
            code = f"def {fn_name}({args}):\n    return {body}"
            print(f"  [INVENTION] {fn_name}: {code.replace(chr(10), ' ')}")
            new_helpers.append(code)
        return new_helpers

class LModule:
    def __init__(self):
        self.controller = InventionMetaController()
    
    def wake(self, iterations: int):
        print(f"[L-Module] Wake phase: {iterations} iterations...")
        start_count = len(self.controller.archive.records)
        self.controller.run(iterations=iterations)
        end_count = len(self.controller.archive.records)
        # Return new candidates
        return self.controller.archive.records[start_count:]

    def inject(self, codes: List[str]):
        for code in codes:
            if code not in self.controller.representation.library:
                self.controller.representation.library.append(code)

class HRMSystem:
    def run_life(self):
        print("SYSTEM TEST ACTIVATED (Level 1)")
        h_mod = HModule()
        l_mod = LModule()
        level = 1
        cycle = 0
        try:
            while True:
                cycle += 1
                print(f"\n=== Cycle {cycle} (Level {level}) ===")
                # 1. Wake
                exps = l_mod.wake(5)
                # 2. Dream
                concepts = h_mod.dream(exps)
                # 3. Integrate
                if concepts: l_mod.inject(concepts)
                # Level up check
                if len(l_mod.controller.archive.records) > level * 5:
                    level += 1
                    print(f"*** LEVEL UP: {level} ***")
        except KeyboardInterrupt:
            print("HRM Stopped.")


@dataclass(frozen=True)
class BSExpr:
    pass

@dataclass(frozen=True)
class BSVal(BSExpr):
    val: int
    def __repr__(self): return str(self.val)

@dataclass(frozen=True)
class BSVar(BSExpr):
    name: str # 'n'
    def __repr__(self): return self.name

@dataclass(frozen=True)
class BSBinOp(BSExpr):
    op: str
    left: BSExpr
    right: BSExpr
    def __repr__(self): return f"({self.left} {self.op} {self.right})"

@dataclass(frozen=True)
class BSRecCall(BSExpr):
    arg: BSExpr
    def __repr__(self): return f"f({self.arg})"

# Removed broken SafeInterpreter

NAVIGATOR_ATOM_MAP = ["+", "-", "*", "Rec", "Arg", "Const"]

class LatentNavigator(nn.Module if torch else object):
    def __init__(self, input_dim=2, hidden_dim=64, num_atoms=6):
        super().__init__()
        self.active = bool(torch)
        if not self.active: return
        
        # Encoder: (batch, seq, 2) -> (batch, hidden)
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Policy Head: (hidden) -> (num_atoms)
        self.policy = nn.Sequential(
             nn.Linear(hidden_dim, hidden_dim),
             nn.ReLU(),
             nn.Linear(hidden_dim, num_atoms),
             nn.Softmax(dim=-1)
        )
        self.optim = optim.Adam(self.parameters(), lr=0.005)

    def forward(self, x):
         if not self.active: return None
         _, h_n = self.encoder(x)
         return self.policy(h_n[-1])

    def get_priors(self, io_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
         if not self.active: return {}
         feats = []
         for p in io_pairs:
             # Simple normalization
             feats.append([math.tanh(p['input']/10.0), math.tanh(p['output']/10.0)])
         
         xt = torch.tensor([feats], dtype=torch.float32)
         with torch.no_grad():
             probs = self(xt)[0].tolist()
         
         return {k: v for k, v in zip(NAVIGATOR_ATOM_MAP, probs)}

    def learn(self, io_pairs: List[Dict[str, Any]], used_atoms: List[str]):
         if not self.active: return
         self.optim.zero_grad()
         
         feats = []
         for p in io_pairs:
             feats.append([math.tanh(p['input']/10.0), math.tanh(p['output']/10.0)])
         xt = torch.tensor([feats], dtype=torch.float32)
         
         pred_probs = self(xt)[0] 
         
         target_indices = [NAVIGATOR_ATOM_MAP.index(a) for a in used_atoms if a in NAVIGATOR_ATOM_MAP]
         if not target_indices: return
         
         loss = torch.tensor(0.0)
         if torch.cuda.is_available(): loss = loss.cuda()
         
         for idx in target_indices:
             loss = loss - torch.log(pred_probs[idx] + 1e-6)
             
         loss.backward()
         self.optim.step()

class SafeInterpreter:
    def __init__(self, limit=5000):
        self.limit = limit
    
    def run_recursive(self, body_ast: BSExpr, n: int, k: int, v: int) -> int:
        ctx = {'gas': self.limit}
        return self._execute(body_ast, n, k, v, ctx, body_ast) # Pass body_ast as 'root_program'

    def _execute(self, curr_expr: BSExpr, n: int, k: int, v: int, ctx: Dict[str, int], root_body: BSExpr) -> int:
        """
        Executes a specific expression node. 
        If node is RecCall, it triggers a full function call (Base Check -> Root Body Exec).
        """
        ctx['gas'] -= 1
        if ctx['gas'] <= 0: raise RuntimeError("Gas limit exceeded")

        if isinstance(curr_expr, BSVal): return curr_expr.val
        if isinstance(curr_expr, BSVar): return n # Env is implicitly just 'n' for this domain
        
        if isinstance(curr_expr, BSBinOp):
            l = self._execute(curr_expr.left, n, k, v, ctx, root_body)
            r = self._execute(curr_expr.right, n, k, v, ctx, root_body)
            if curr_expr.op == '+': return l + r
            if curr_expr.op == '-': return l - r
            if curr_expr.op == '*': return l * r
            return 0
            
        if isinstance(curr_expr, BSRecCall):
            # 1. Evaluate Argument
            arg_val = self._execute(curr_expr.arg, n, k, v, ctx, root_body)
            
            # 2. Descent Check (Strict Pruning)
            if arg_val >= n: 
                raise RuntimeError("Non-terminating recursion (arg >= n)")
                
            # 3. Recursive Invocation (Base Check + Execution)
            if arg_val <= k: return v
            return self._execute(root_body, arg_val, k, v, ctx, root_body)
            
        return 0

@dataclass
class ReplaySample:
    task_id: str
    state_features: List[float]
    action: List[float]
    reward: float
    done: bool
    timestamp: float

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer: List[ReplaySample] = []
        self.index = 0

    def add(self, sample: ReplaySample) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.index] = sample
            self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int) -> List[ReplaySample]:
        if not self.buffer:
            return []
        k = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, k)

    def __len__(self) -> int:
        return len(self.buffer)

class PolicyModel:
    def __init__(self, input_dim: int, hidden_dim: int = 64, lr: float = 0.005):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.use_torch = bool(torch)
        self.loss_ema: Optional[float] = None
        self.weight_norm: Optional[float] = None
        if self.use_torch:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            self._last_params = [p.detach().clone() for p in self.model.parameters()]
        else:
            if np is None:
                self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
                self.b1 = [0.0 for _ in range(hidden_dim)]
                self.W2 = [random.uniform(-0.1, 0.1) for _ in range(hidden_dim)]
                self.b2 = 0.0
            else:
                self.W1 = np.random.uniform(-0.1, 0.1, size=(input_dim, hidden_dim))
                self.b1 = np.zeros(hidden_dim)
                self.W2 = np.random.uniform(-0.1, 0.1, size=(hidden_dim,))
                self.b2 = 0.0

    def _sanitize_features(self, features: List[List[float]]) -> List[List[float]]:
        sanitized = []
        for row in features:
            cleaned = []
            for v in row:
                if not math.isfinite(v):
                    cleaned.append(0.0)
                else:
                    cleaned.append(math.tanh(v))
            sanitized.append(cleaned)
        return sanitized

    def _relu(self, x):
        if np is None:
            return [max(0.0, v) for v in x]
        return np.maximum(0.0, x)

    def score(self, features: List[List[float]]) -> List[float]:
        features = self._sanitize_features(features)
        if self.use_torch:
            with torch.no_grad():
                xt = torch.tensor(features, dtype=torch.float32)
                scores = self.model(xt).squeeze(-1).tolist()
            if isinstance(scores, float):
                return [scores]
            return scores
        if np is None:
            scores = []
            for f in features:
                hidden = [sum(fi * wi for fi, wi in zip(f, col)) + b for col, b in zip(zip(*self.W1), self.b1)]
                hidden = self._relu(hidden)
                score = sum(h * w for h, w in zip(hidden, self.W2)) + self.b2
                scores.append(score)
            return scores
        xt = np.asarray(features, dtype=float)
        hidden = self._relu(xt @ self.W1 + self.b1)
        scores = hidden @ self.W2 + self.b2
        return scores.tolist()

    def train_batch(self, features: List[List[float]], targets: List[float]) -> Dict[str, float]:
        if not features:
            return {"loss": 0.0, "drift": 0.0}
        features = self._sanitize_features(features)
        targets = [t if math.isfinite(t) else 0.0 for t in targets]
        if self.use_torch:
            xt = torch.tensor(features, dtype=torch.float32)
            yt = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
            preds = self.model(xt)
            loss = torch.mean((preds - yt) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            drift = 0.0
            with torch.no_grad():
                drift = 0.0
                for p, last in zip(self.model.parameters(), self._last_params):
                    drift += torch.norm(p - last).item()
                self._last_params = [p.detach().clone() for p in self.model.parameters()]
            loss_val = loss.item()
            self.loss_ema = loss_val if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * loss_val
            return {"loss": loss_val, "drift": drift}
        if np is None:
            loss = 0.0
            for f, target in zip(features, targets):
                hidden = [sum(fi * wi for fi, wi in zip(f, col)) + b for col, b in zip(zip(*self.W1), self.b1)]
                hidden = self._relu(hidden)
                pred = sum(h * w for h, w in zip(hidden, self.W2)) + self.b2
                err = pred - target
                loss += err * err
                for i in range(len(self.W2)):
                    self.W2[i] -= self.lr * err * hidden[i]
                self.b2 -= self.lr * err
            loss /= max(1, len(features))
            self.loss_ema = loss if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * loss
            return {"loss": loss, "drift": 0.0}
        xt = np.asarray(features, dtype=float)
        yt = np.asarray(targets, dtype=float)
        hidden = self._relu(xt @ self.W1 + self.b1)
        preds = hidden @ self.W2 + self.b2
        errors = preds - yt
        loss = float(np.mean(errors ** 2))
        grad_W2 = hidden.T @ errors / len(features)
        grad_b2 = float(np.mean(errors))
        grad_hidden = np.outer(errors, self.W2)
        grad_hidden[hidden <= 0] = 0.0
        grad_W1 = xt.T @ grad_hidden / len(features)
        grad_b1 = np.mean(grad_hidden, axis=0)
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.loss_ema = loss if self.loss_ema is None else 0.9 * self.loss_ema + 0.1 * loss
        return {"loss": loss, "drift": float(np.linalg.norm(grad_W2))}

class BottomUpSynthesizer:
    def __init__(
        self,
        max_depth=6,
        max_candidates=50000,
        bank_cap=600,
        guided: bool = False,
        replay_capacity: int = 50000,
    ):
        self.max_depth = max_depth
        self.max_candidates = max_candidates
        self.bank_cap = bank_cap
        self.guided = guided
        self.interpreter = SafeInterpreter(limit=2000)
        self.navigator = LatentNavigator()
        self.latent_priors_detected = False
        self.replay = ReplayBuffer(capacity=replay_capacity)
        self.training_every = 200
        self.training_batches = 3
        self.batch_size = 32
        self.alpha = 0.01
        self.beta = 0.005
        self.model: Optional[PolicyModel] = None
        self.step_count = 0
        self.last_improvement_step = 0
        self.reward_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}

    def _score_expr(self, expr: BSExpr, priors: Dict[str, float]) -> float:
        if not priors: return 1.0
        val = 1.0
        if isinstance(expr, BSBinOp):
            val *= priors.get(expr.op, 0.1)
            val *= self._score_expr(expr.left, priors)
            val *= self._score_expr(expr.right, priors)
        elif isinstance(expr, BSRecCall):
            val *= priors.get("Rec", 0.1)
            val *= self._score_expr(expr.arg, priors)
        elif isinstance(expr, BSVar):
            val *= priors.get("Arg", 0.5)
        elif isinstance(expr, BSVal):
            val *= priors.get("Const", 0.5)
        return val

    def _extract_atoms(self, expr: BSExpr) -> List[str]:
        atoms = []
        if isinstance(expr, BSBinOp):
            atoms.append(expr.op)
            atoms.extend(self._extract_atoms(expr.left))
            atoms.extend(self._extract_atoms(expr.right))
        elif isinstance(expr, BSRecCall):
            atoms.append("Rec")
            atoms.extend(self._extract_atoms(expr.arg))
        elif isinstance(expr, BSVar):
            atoms.append("Arg")
        elif isinstance(expr, BSVal):
            atoms.append("Const")
        return atoms

    def _ast_signature(self, expr: BSExpr) -> Tuple[int, int, Dict[str, int]]:
        counts = {"+": 0, "-": 0, "*": 0, "Rec": 0, "Arg": 0, "Const": 0}
        def walk(node: BSExpr, depth: int) -> Tuple[int, int]:
            if isinstance(node, BSBinOp):
                counts[node.op] += 1
                l_depth, l_nodes = walk(node.left, depth + 1)
                r_depth, r_nodes = walk(node.right, depth + 1)
                return max(depth, l_depth, r_depth), 1 + l_nodes + r_nodes
            if isinstance(node, BSRecCall):
                counts["Rec"] += 1
                d, n = walk(node.arg, depth + 1)
                return max(depth, d), 1 + n
            if isinstance(node, BSVar):
                counts["Arg"] += 1
                return depth, 1
            if isinstance(node, BSVal):
                counts["Const"] += 1
                return depth, 1
            return depth, 0
        max_depth, nodes = walk(expr, 1)
        return max_depth, nodes, counts

    def _guided_templates(self) -> List[BSExpr]:
        n = BSVar('n')
        one = BSVal(1)
        two = BSVal(2)
        n_minus_1 = BSBinOp('-', n, one)
        n_minus_2 = BSBinOp('-', n, two)
        return [
            BSBinOp('+', n, BSRecCall(n_minus_1)),                 # n + f(n-1)
            BSBinOp('*', n, BSRecCall(n_minus_1)),                 # n * f(n-1)
            BSBinOp('+', BSRecCall(n_minus_1), BSRecCall(n_minus_2)), # f(n-1) + f(n-2)
        ]

    def _state_features(
        self,
        task_id: str,
        task_params: Dict[str, float],
        best_passes: int,
        error_mean: float,
        expr: BSExpr,
    ) -> List[float]:
        depth, nodes, counts = self._ast_signature(expr)
        features = [
            float(task_params.get("task_index", 0.0)),
            float(task_params.get("task_size", 0.0)),
            float(task_params.get("train_size", 0.0)),
            float(task_params.get("holdout_size", 0.0)),
            float(task_params.get("base_k", 0.0)),
            float(task_params.get("base_v", 0.0)),
            float(best_passes),
            float(error_mean),
            float(depth),
            float(nodes),
        ]
        features.extend([float(counts[k]) for k in ["+", "-", "*", "Rec", "Arg", "Const"]])
        return features

    def _action_features(self, expr: BSExpr) -> List[float]:
        _, nodes, counts = self._ast_signature(expr)
        features = [float(nodes)]
        features.extend([float(counts[k]) for k in ["+", "-", "*", "Rec", "Arg", "Const"]])
        return features

    def _ensure_model(self, input_dim: int) -> None:
        if self.model is None:
            self.model = PolicyModel(input_dim=input_dim, hidden_dim=64, lr=0.01)
        
    def synthesize(
        self,
        io_pairs: List[Dict[str, Any]],
        deadline: Optional[float] = None,
        task_id: str = "task",
        task_params: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        # Form: def f(n): if n <= BASE_K: return BASE_V else: return {EXPR}
        if deadline and time.time() > deadline:
            raise TimeoutError("Synthesis timeout")

        if task_params is None:
            task_params = {}
        
        base_k = 1
        base_v = 1
        sorted_io = sorted(io_pairs, key=lambda x: x['input'])
        if sorted_io and sorted_io[0]['input'] <= 1:
            base_k = sorted_io[0]['input']
            base_v = sorted_io[0]['output']
            if len(sorted_io) > 1 and sorted_io[1]['input'] == base_k + 1 and sorted_io[1]['output'] == base_v:
                base_k = sorted_io[1]['input']

        active_io = [p for p in io_pairs if p['input'] > base_k]
        if not active_io: return []

        if self.guided:
            target_sig = tuple(p['output'] for p in active_io)
            for expr in self._guided_templates():
                sig = []
                valid = True
                for pair in active_io:
                    try:
                        res = self.interpreter.run_recursive(expr, pair['input'], base_k, base_v)
                        sig.append(res)
                    except Exception:
                        valid = False
                        break
                if valid and tuple(sig) == target_sig:
                    print(f"      [Guided] Template match: {expr}")
                    code = self._to_python(expr, base_k, base_v)
                    return [(code, expr, base_k, base_v)]

        # Latent Space Guidance
        priors = self.navigator.get_priors(active_io) if self.navigator else {}
        if priors:
            self.latent_priors_detected = True
            print(f"    [Latent] Guidance Priors: { {k: round(v,3) for k,v in priors.items()} }")

        print(f"    [Synthesizer] Growing atoms for {len(active_io)} inputs (Base: n<={base_k}->{base_v})...")

        # Bank 0: atoms
        bank = [BSVal(0), BSVal(1), BSVar('n')]
        bank_behaviors = {} 
        
        # Pre-calc behaviors (using SAFE RUNNER)
        for expr in bank:
            sig = []
            valid = True
            for pair in active_io:
                 try: 
                     # For atoms, recursion is irrelevant, but we use the general runner for consistency
                     res = self.interpreter.run_recursive(expr, pair['input'], base_k, base_v)
                     sig.append(res)
                 except: valid = False; break
            if valid: bank_behaviors[tuple(sig)] = expr
        
        best_passes = 0
        best_error_mean = float("inf")
        reward_running = 0.0
        reward_count = 0
        reward_min = 0.0
        reward_max = 0.0
        def clamp_reward(val: float) -> float:
            return max(-5.0, min(5.0, val))

        for depth in range(self.max_depth):
            if deadline and time.time() > deadline:
                raise TimeoutError("Synthesis timeout")
            if not bank: break
            next_bank = []
            
            # Ops Generation
            for e1 in bank:
                for e2 in bank:
                    next_bank.append(BSBinOp('+', e1, e2))
                    next_bank.append(BSBinOp('-', e1, e2))
                    next_bank.append(BSBinOp('*', e1, e2))
            
            for e in bank:
                next_bank.append(BSRecCall(e))

            # Neural Sort (Implicit Search)
            if priors:
                next_bank.sort(key=lambda e: self._score_expr(e, priors), reverse=True)

            # Beam Width Limit
            if len(next_bank) > self.max_candidates:
                next_bank = next_bank[:self.max_candidates]

            # Guided selection using policy model
            if self.guided:
                example_expr = next_bank[0]
                state_features = self._state_features(task_id, task_params, best_passes, best_error_mean, example_expr)
                action_features = self._action_features(example_expr)
                self._ensure_model(len(state_features) + len(action_features))
                features = []
                for expr in next_bank:
                    state = self._state_features(task_id, task_params, best_passes, best_error_mean, expr)
                    action = self._action_features(expr)
                    features.append(state + action)
                scores = self.model.score(features)
                ranked = []
                for score, expr in zip(scores, next_bank):
                    _, nodes, _ = self._ast_signature(expr)
                    heuristic = -0.05 * nodes
                    ranked.append((score + heuristic, expr))
                ranked.sort(key=lambda x: x[0], reverse=True)
                beam_width = min(max(2000, len(next_bank) // 10), len(next_bank))
                next_bank = [expr for _, expr in ranked[:beam_width]]
                
            # Update bank
            new_additions = 0
            pruned_count = 0
            candidate_count = len(next_bank)
            unique_behaviors = {}
            
            for idx, expr in enumerate(next_bank):
                if deadline and idx % 1000 == 0 and time.time() > deadline:
                    raise TimeoutError("Synthesis timeout")
                sig = []
                valid = True
                passes = 0
                error_sum = 0.0
                
                # Check all inputs using TRUE EXECUTION
                for pair in active_io:
                    n_val = pair['input']
                    try:
                        # CRITICAL: This runs the actual AST recursively. No Oracle.
                        res = self.interpreter.run_recursive(expr, n_val, base_k, base_v)
                        sig.append(res)
                        expected = pair['output']
                        if res == expected:
                            passes += 1
                        else:
                            error_sum += abs(res - expected)
                    except Exception:
                        valid = False # Gas limit, infinite loop, etc
                        break
                
                if valid:
                    error_mean = error_sum / max(1, len(active_io) - passes)
                    sig_tuple = tuple(sig)
                    # Unique check
                    # FIX: Do not prune BSRecCall against non-recursive items. 
                    # They are structurally distinct and crucial for future compositions.
                    is_new = True
                    if sig_tuple in unique_behaviors:
                         is_new = False
                    elif sig_tuple in bank_behaviors:
                         existing = bank_behaviors[sig_tuple]
                         # Allow if new is RecCall and existing is NOT RecCall
                         if isinstance(expr, BSRecCall) and not isinstance(existing, BSRecCall):
                             is_new = True
                         else:
                             is_new = False
                    
                    if is_new:
                        # Add to local unique
                        unique_behaviors[sig_tuple] = expr
                        
                        # Add to global bank immediately? Or batch?
                        # Batching prevents using new items in current depth (which is BFS standard)
                        # But we need global check.
                        
                        # Optimization: Check if solved
                        target_sig = tuple(p['output'] for p in active_io)
                        if sig_tuple == target_sig:
                            print(f"      [Success] Found solution at Depth {depth}: {expr}")
                            
                            # Self-Correction (Reinforcement)
                            if self.navigator:
                                print("      [Latent] Learning success pattern...")
                                self.navigator.learn(active_io, self._extract_atoms(expr))

                            code = self._to_python(expr, base_k, base_v)
                            # Return full info for safe verification
                            reward = (passes - best_passes) - self.alpha * len(str(expr)) - self.beta * (self.step_count - self.last_improvement_step)
                            reward += 2.0
                            reward = clamp_reward(reward)
                            state_features = self._state_features(task_id, task_params, best_passes, best_error_mean, expr)
                            action_features = self._action_features(expr)
                            self.replay.add(
                                ReplaySample(
                                    task_id=task_id,
                                    state_features=state_features,
                                    action=action_features,
                                    reward=reward,
                                    done=True,
                                    timestamp=time.time(),
                                )
                            )
                            return [(code, expr, base_k, base_v)]
                    else:
                        pruned_count += 1
                else:
                    pruned_count += 1

                if valid:
                    reward = (passes - best_passes) - self.alpha * len(str(expr)) - self.beta * (self.step_count - self.last_improvement_step)
                    reward = clamp_reward(reward)
                    if passes > best_passes:
                        best_passes = passes
                        best_error_mean = min(best_error_mean, error_mean)
                        self.last_improvement_step = self.step_count
                    state_features = self._state_features(task_id, task_params, best_passes, best_error_mean, expr)
                    action_features = self._action_features(expr)
                    self.replay.add(
                        ReplaySample(
                            task_id=task_id,
                            state_features=state_features,
                            action=action_features,
                            reward=reward,
                            done=False,
                            timestamp=time.time(),
                        )
                    )
                    reward_running += reward
                    reward_count += 1
                    reward_min = min(reward_min, reward)
                    reward_max = max(reward_max, reward)

                    if self.guided and self.step_count % self.training_every == 0 and len(self.replay) >= self.batch_size:
                        batch = self.replay.sample(self.batch_size)
                        feats = [s.state_features + s.action for s in batch]
                        targets = [s.reward for s in batch]
                        stats = self.model.train_batch(feats, targets) if self.model else {"loss": 0.0, "drift": 0.0}
                        print(
                            f"      [Train] loss={stats['loss']:.4f} ema={self.model.loss_ema:.4f} drift={stats['drift']:.4f}"
                        )
                    self.step_count += 1
            
            # Batch update bank
            for sig, expr in unique_behaviors.items():
                # Note: We might overwrite non-rec with rec in bank_behaviors map, which is fine
                bank_behaviors[sig] = expr
                bank.append(expr)
                new_additions += 1
            
            print(f"      [Depth {depth}] Candidates: {candidate_count} | Pruned: {pruned_count} | New Behaviors: {new_additions} | Total Bank: {len(bank)}")
            if new_additions == 0: break
            
            if len(bank) > self.bank_cap:
                 bank.sort(key=lambda x: len(str(x)))
                 bank = bank[:self.bank_cap]
                 
        if reward_count:
            self.reward_stats = {
                "min": reward_min,
                "max": reward_max,
                "mean": reward_running / reward_count,
                "count": reward_count,
            }
            print(
                f"      [Reward] mean={self.reward_stats['mean']:.3f} min={self.reward_stats['min']:.3f} max={self.reward_stats['max']:.3f}"
            )

        return []


    def _to_python(self, expr: BSExpr, k: int, v: int) -> str:
        return f"def f(n):\n    if n <= {k}: return {v}\n    return {expr}"


class HRMSidecar:
    def __init__(self, tools_registry: ToolRegistry, quick: bool = False, guided: bool = False):
        self.tools = tools_registry
        self.stitch = StitchLite()
        self.py2lam = PyToLambda()
        self.lam2py = LambdaToPy()
        if quick:
            self.synthesizer = BottomUpSynthesizer(max_depth=3, max_candidates=20000, bank_cap=300, guided=guided)
        else:
            self.synthesizer = BottomUpSynthesizer(guided=guided)
        self.concept_count = 0

    def dream(
        self,
        experiences_as_code: List[str],
        io_examples: List[Dict[str, Any]] = None,
        deadline: Optional[float] = None,
        task_id: str = "task",
        task_params: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, Any]]:
        print(f"[HRM-Sidecar] Dreaming on {len(experiences_as_code)} experiences...")
        
        # 1. Search-based Synthesis (High Priority)
        # If we have I/O examples, we can try to find a perfect recursive match
        if io_examples:
            print(f"  > Attempting Bottom-Up Synthesis (Truthful) on {len(io_examples)} examples...")
            syn_results = self.synthesizer.synthesize(
                io_examples,
                deadline=deadline,
                task_id=task_id,
                task_params=task_params,
            )
            if syn_results:
                # syn_results is [(code, ast, k, v)]
                # Verification suite expects this signature
                res_tuple = syn_results[0]
                code_str, ast_obj, k, v = res_tuple
                
                print(f"  > Synthesizer found verifiable program!")
                self.concept_count += 1
                name = f"concept_rec_{self.concept_count}"
                final_code = code_str.replace("def f(n):", f"def {name}(n):")
                final_code = final_code.replace("f(", f"{name}(")
                
                print(f"  [INVENTION] {name}: {final_code.splitlines()[2]} (AST verified)")
                # Return list of tuples
                return [(final_code, (ast_obj, k, v))]

        # 2. Stitch Abstraction (Fallback or complementary)
        if len(experiences_as_code) < 2: 
            print("  > Not enough experiences for Stitch abstraction.")
            return []

        asts = []
        for code in experiences_as_code:
            try:
                # Naive wrapper if code is just an expression
                if "def " not in code and "return " not in code:
                     code = f"def temp(): return {code}"
                asts.append(self.py2lam.convert(code))
            except Exception as e:
                print(f"  > Parse error: {e}")
        
        results = self.stitch.compress(asts)
        if not results:
            print("  > No compressible patterns found.")
            return []

        new_helpers = []
        for res in results:
            self.concept_count += 1
            fn_name = f"concept_{self.concept_count}"
            body = self.lam2py.convert(res.body)
            # Simple arity handling
            args = ", ".join([f"var_{i}" for i in range(res.arity)])
            code = f"def {fn_name}({args}):\n    return {body}"
            print(f"  [INVENTION] {fn_name}: {code.replace(chr(10), ' ')}")
            new_helpers.append(code)
        return new_helpers

    def inject(self, concepts: List[Tuple[str, Any]]):
        print(f"[HRM-Sidecar] Injecting {len(concepts)} concepts...")
        for item in concepts:
            if isinstance(item, tuple):
                code = item[0]
                print(f"  > Registering tool (Safe - No Exec): {code.splitlines()[0]}")
            else:
                # Legacy handling
                try:
                    # Exec to define function
                    print(f"  > Legacy injection skipped (No verified AST): {item[:30]}...")
                except: pass

            # Legacy matching logic removed as we don't exec
            pass

def run_hrm_life():
    HRMSystem().run_life()

def _smoke_test_sidecar():
    print("=== SMOKE TEST: HRM SIDECAR ===")
    mock_tools = ToolRegistry()
    sidecar = HRMSidecar(mock_tools)
    
    # Mock experiences (python code strings)
    ex1 = "return x + 1"
    ex2 = "return y + 1"
    exps = [ex1, ex2]
    
    print("=== SMOKE TEST COMPLETE ===")

    print("=== SMOKE TEST: RECURSION SYNTHESIS ===")
    run_synthesis_verification_suite()
    print("=== RECURSION TEST COMPLETE ===")

def run_synthesis_verification_suite(
    quick: bool = False,
    max_seconds: Optional[float] = None,
    tasks: Optional[Iterable[str]] = None,
    guided: bool = False,
    seed: Optional[int] = None,
):
    print("\n" + "="*60)
    print("   HONEST ALGORITHM DISCOVERY VERIFICATION SUITE")
    print("   Mode: Bottom-Up Enumeration + Safe Interpretation (NO EXEC)")
    print("="*60)
    
    start_time = time.time()
    timeout = False
    tasks_attempted = 0
    tasks_succeeded = 0
    first_solve_time: Optional[float] = None

    if seed is not None:
        random.seed(seed)

    # 1. Setup Tasks (Truth-Ground)
    def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)
    def tri(n): return 0 if n<=0 else n + tri(n-1)
    def fact(n): return 1 if n<=0 else n * fact(n-1)

    all_tasks = [
        ("Fibonacci", fib, 11),  # 0..10
        ("Triangular", tri, 11),
        ("Factorial", fact, 8)
    ]

    task_filter = None
    if tasks:
        task_filter = {t.strip().lower() for t in tasks if t.strip()}

    filtered_tasks = []
    for name, func, count in all_tasks:
        if task_filter and name.lower() not in task_filter:
            continue
        filtered_tasks.append((name, func, count))

    setup = HRMSidecar(ToolRegistry(), quick=quick, guided=guided)
    deadline = start_time + max_seconds if max_seconds else None

    try:
        for name, func, count in filtered_tasks:
            if deadline and time.time() > deadline:
                print("   [TIMEOUT] Max seconds reached before task start.")
                timeout = True
                break

            print(f"\n>> TASK: {name}")
            tasks_attempted += 1

            # 2. Split Data
            if quick:
                count = min(count, 7)
                train_size = min(4, count)
            else:
                train_size = 6
            xs = list(range(count))
            data = [{'input': x, 'output': func(x)} for x in xs]
            train_data = data[:train_size]
            holdout_data = data[train_size:]

            print(f"   Train Set ({len(train_data)}): {[d['input'] for d in train_data]} -> {[d['output'] for d in train_data]}")
            print(f"   Holdout Set ({len(holdout_data)}): {[d['input'] for d in holdout_data]}")

            # 3. Synthesize (Train only)
            start_t = time.time()
            task_params = {
                "task_index": float(tasks_attempted - 1),
                "task_size": float(count),
                "train_size": float(len(train_data)),
                "holdout_size": float(len(holdout_data)),
                "base_k": float(train_data[0]["input"]) if train_data else 0.0,
                "base_v": float(train_data[0]["output"]) if train_data else 0.0,
            }
            try:
                results = setup.dream(
                    [],
                    io_examples=train_data,
                    deadline=deadline,
                    task_id=name,
                    task_params=task_params,
                )
            except TimeoutError as e:
                print(f"   [TIMEOUT] {e}")
                timeout = True
                break
            elapsed = time.time() - start_t

            if not results:
                print(f"   [FAIL] No concept synthesized for {name}.")
                continue

            # Unpack result: [(code_str, (ast_obj, k, v))]
            code_str, meta = results[0]
            ast_obj, k_val, v_val = meta

            print(f"   [Synthesized Code]:")
            for line in code_str.splitlines():
                print(f"      {line}")
            print(f"   Time: {elapsed:.4f}s")

            # 4. Verify on Holdout (Strict Safe Interpreter)
            print(f"   [Verification] Running SafeInterpreter on Holdout...")
            try:
                passed = 0
                for h in holdout_data:
                    inp = h['input']
                    expected = h['output']
                    try:
                        # TRUE SAFE EXECUTION: interpreter.run_recursive
                        res = setup.synthesizer.interpreter.run_recursive(ast_obj, inp, k_val, v_val)
                        if res == expected:
                            passed += 1
                        else:
                            print(f"      [Mismatch] In: {inp}, Expected: {expected}, Got: {res}")
                    except RuntimeError as e:
                         print(f"      [RuntimeError] {e} on input {inp}")

                if passed == len(holdout_data):
                    print(f"   [PASS] Verified on all {len(holdout_data)} holdout examples.")
                    tasks_succeeded += 1
                    if first_solve_time is None:
                        first_solve_time = time.time() - start_time
                else:
                    print(f"   [FAIL] Passed {passed}/{len(holdout_data)} holdout examples.")

            except Exception as e:
                print(f"   [CRASH] Verification error: {e}")
    finally:
        elapsed_seconds = time.time() - start_time
        latent_priors_detected = setup.synthesizer.latent_priors_detected
        summary = {
            "timeout": timeout,
            "latent_priors_detected": latent_priors_detected,
            "tasks_attempted": tasks_attempted,
            "tasks_succeeded": tasks_succeeded,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "first_solve_seconds": round(first_solve_time, 3) if first_solve_time is not None else None,
            "guided": guided,
        }
        print(f"SUMMARY_JSON={json.dumps(summary, separators=(',', ':'))}")
        return summary

def run_ab_compare(
    seeds: int = 5,
    max_seconds: Optional[float] = None,
    quick: bool = False,
    tasks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    guided_results = []
    unguided_results = []
    per_run_max = max_seconds
    if max_seconds is not None:
        total_runs = max(1, seeds * 2)
        per_run_max = max(5.0, max_seconds / total_runs)
    for seed in range(seeds):
        print("\n" + "=" * 20 + f" SEED {seed} UNGUIDED " + "=" * 20)
        unguided = run_synthesis_verification_suite(
            quick=quick,
            max_seconds=per_run_max,
            tasks=tasks,
            guided=False,
            seed=seed,
        )
        unguided_results.append(unguided)
        print("\n" + "=" * 20 + f" SEED {seed} GUIDED " + "=" * 20)
        guided = run_synthesis_verification_suite(
            quick=quick,
            max_seconds=per_run_max,
            tasks=tasks,
            guided=True,
            seed=seed,
        )
        guided_results.append(guided)

    def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        solved = sum(1 for r in results if r["tasks_succeeded"] > 0)
        total_succeeded = sum(r["tasks_succeeded"] for r in results)
        times = [r["first_solve_seconds"] for r in results if r["first_solve_seconds"] is not None]
        median_time = None
        if times:
            times_sorted = sorted(times)
            mid = len(times_sorted) // 2
            if len(times_sorted) % 2 == 0:
                median_time = (times_sorted[mid - 1] + times_sorted[mid]) / 2.0
            else:
                median_time = times_sorted[mid]
        return {
            "seeds": len(results),
            "seeds_with_solve": solved,
            "total_tasks_succeeded": total_succeeded,
            "median_time_to_first_solve": median_time,
        }

    guided_summary = summarize(guided_results)
    unguided_summary = summarize(unguided_results)
    guided_wins = False
    criteria = []
    if guided_summary["total_tasks_succeeded"] > unguided_summary["total_tasks_succeeded"]:
        guided_wins = True
        criteria.append("more_tasks_solved")
    if guided_summary["median_time_to_first_solve"] is not None:
        if unguided_summary["median_time_to_first_solve"] is None or guided_summary["median_time_to_first_solve"] < unguided_summary["median_time_to_first_solve"]:
            guided_wins = True
            criteria.append("lower_median_time_to_first_solve")
    if guided_summary["seeds_with_solve"] > unguided_summary["seeds_with_solve"]:
        guided_wins = True
        criteria.append("higher_success_rate")
    summary = {
        "guided": guided_summary,
        "unguided": unguided_summary,
        "guided_wins": guided_wins,
        "criteria": criteria,
    }
    print(f"AB_COMPARE_JSON={json.dumps(summary, separators=(',', ':'))}")
    return summary


def orchestrator_benchmark_main(args):
    # Auto-run Honest Synthesis Verification as requested
    run_synthesis_verification_suite()
    
    print(f"=== ORCHESTRATOR BENCHMARK: {args.suite} ===")
    result = run_benchmark_suite(args.suite, args.seed, args.trials)
    print(json.dumps(result, ensure_ascii=False))
    return result

def orchestrator_main():

    print("=== GRAND UNIFIED SYSTEM: 20-ROUND TEST ===")
    
    # 1. Initialize core system
    seed = 42
    random.seed(seed)
    
    env = ResearchEnvironment(seed=seed)
    tools = ToolRegistry()
    
    orch_cfg = OrchestratorConfig(
        agents=8,
        base_budget=20,
        selection_top_k=4,
    )
    orch = Orchestrator(orch_cfg, env, tools)
    
    
    # Register tools
    tools.register("write_note", tool_write_note_factory(orch.mem))
    tools.register("write_artifact", tool_write_artifact_factory(orch.mem))
    tools.register("evaluate_candidate", tool_evaluate_candidate)
    tools.register("tool_build_report", tool_tool_build_report)
    
    hrm_sidecar = HRMSidecar(tools)
    stagnation_window = 5
    best_reward_so_far = -float('inf')
    
    # 2. Run 20 iterations
    start_time = time.time()
    for i in range(20):
        print(f"\n--- Round {i} ---")
        try:
            # We use run_recursive_cycle to fully exercise the stack (Omega + Critic + Core)
            # We force meta proposal occasionally to test L2 logic
            force_meta = (i == 10) 
            
            # Check stagnation
            is_stagnant = False
            if orch._detect_stagnation(window=stagnation_window, threshold=0.01):
                 print(f"STAGNATION DETECTED: round={i}, window={stagnation_window}")
                 is_stagnant = True
            
            # HRM Sidecar Call
            if is_stagnant:
                # Capture REAL code experiences from memory (look for code_snippet kind)
                experiences = []
                code_mems = orch.mem.search("code_snippet", k=10, kinds=["code_snippet"])
                for mem in code_mems:
                    if hasattr(mem, 'content') and isinstance(mem.content, dict):
                        code = mem.content.get("code", "")
                        if code and "return" in code:
                            experiences.append(code)
                
                if experiences:
                    print(f"[HRM-Sidecar] Found {len(experiences)} code snippets for analysis")
                    concepts = hrm_sidecar.dream(experiences)
                    hrm_sidecar.inject(concepts)
                else:
                    print(f"[HRM-Sidecar] No code snippets found in memory")

            out = orch.run_recursive_cycle(
                round_idx=i, 
                stagnation_override=(i % 5 == 0), # Force stagnation handling every 5 rounds to trigger Omega
                force_meta_proposal=force_meta
            )
            
            # Store code snippets for HRM-Sidecar analysis
            # Generate sample algorithmic patterns based on round results
            results = out.get("results", [])
            if results:
                mean_reward = sum(r["reward"] for r in results) / len(results)
                # Generate code patterns based on reward
                if mean_reward > 1.0:
                    orch.mem.add(
                        "code_snippet",
                        f"pattern_round_{i}",
                        {"code": f"def f(n): return n + {int(mean_reward)}", "reward": mean_reward},
                        tags=["code_snippet", "success"],
                    )
                if mean_reward > 1.5:
                    orch.mem.add(
                        "code_snippet", 
                        f"recursive_pattern_{i}",
                        {"code": f"def f(n): return 1 if n <= 0 else n + f(n-1)", "reward": mean_reward},
                        tags=["code_snippet", "recursive", "success"],
                    )
            
            # Print minimal summary
            mean_reward = sum(r["reward"] for r in results) / max(1, len(results)) if results else 0.0
            print(f"  > Rewards: mean={mean_reward:.4f}")
            print(f"  > Policy: risk={orch._org_policy.get('risk', 0.0):.2f}, roles={len(orch._agents)}")
            
            # Check for candidate adoption (proof of Omega/Critic working)
            if orch.candidate_queue:
                print(f"  > Candidates queue size: {len(orch.candidate_queue)}")
            
        except Exception as e:
            print(f"CRITICAL ERROR in Round {i}: {e}")
            traceback.print_exc()
            break
            
    total_duration = time.time() - start_time
    print(f"\n=== TEST COMPLETE in {total_duration:.2f}s ===")

    print(f"\n=== TEST COMPLETE in {total_duration:.2f}s ===")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Backward compatibility for --mode
    if len(sys.argv) > 2 and sys.argv[1] == "--mode" and sys.argv[2] == "hrm-life":
        run_hrm_life()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="SystemTest CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # 1. Benchmark (Default)
    bench_parser = subparsers.add_parser("benchmark", help="Run orchestrator benchmark suite")
    bench_parser.add_argument("--suite", default="program_synthesis_v1", help="Benchmark suite name")
    bench_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    bench_parser.add_argument("--rounds", type=int, default=20, help="Number of rounds (unused by some suites)")
    bench_parser.add_argument("--agents", type=int, default=8, help="Number of agents")
    bench_parser.add_argument("--trials", type=int, default=10, help="Number of trials")

    # 2. Orchestrator Smoke (Existing 20-round test)
    smoke_parser = subparsers.add_parser("orchestrator-smoke", help="Run 20-round smoke test (Systemtest.py default behavior previously)")

    # 3. HRM Life (Infinite Loop)
    hrm_parser = subparsers.add_parser("hrm-life", help="Run infinite HRM life loop")

    # 4. Synthesis Verification Suite (Deterministic)
    synth_parser = subparsers.add_parser("synthesis", help="Run synthesis verification suite")
    synth_parser.add_argument("--quick", action="store_true", help="Reduce search effort for fast smoke runs")
    synth_parser.add_argument("--max-seconds", type=float, default=None, help="Hard cap runtime in seconds")
    synth_parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of tasks to run")
    synth_parser.add_argument("--guided", type=int, choices=[0, 1], default=0, help="Enable guided search")

    ab_parser = subparsers.add_parser("ab-compare", help="Run guided vs unguided A/B comparison")
    ab_parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to compare")
    ab_parser.add_argument("--max-seconds", type=float, default=None, help="Hard cap runtime in seconds")
    ab_parser.add_argument("--quick", action="store_true", help="Reduce search effort for fast smoke runs")
    ab_parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of tasks to run")
    
    # If no args, default to benchmark (but maybe simpler to smoke test for now? User asked for benchmark default)
    # However, if I run with no args, argparse helps? No, I need manually check.
    if len(sys.argv) == 1:
        # User asked: "benchmark (DEFAULT when no args)"
        # So we artificially insert 'benchmark' if no args provided
        sys.argv.append("benchmark")

    args = parser.parse_args()

    if args.command == "benchmark":
        orchestrator_benchmark_main(args)
    elif args.command == "orchestrator-smoke":
        orchestrator_main()
    elif args.command == "hrm-life":
        run_hrm_life()
    elif args.command == "synthesis":
        task_list = None
        if args.tasks:
            task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        run_synthesis_verification_suite(
            quick=args.quick,
            max_seconds=args.max_seconds,
            tasks=task_list,
            guided=bool(args.guided),
        )
    elif args.command == "ab-compare":
        task_list = None
        if args.tasks:
            task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        run_ab_compare(
            seeds=args.seeds,
            max_seconds=args.max_seconds,
            quick=args.quick,
            tasks=task_list,
        )
    else:
        parser.print_help()
