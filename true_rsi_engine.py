"""
True RSI Engine - The Ultimate Self-Improvement System
======================================================
This is the main RSI engine that actually improves itself.

Key differences from previous attempts:
1. POSITIVE SCORING - Fitness is 0-100, not negative
2. UNCONDITIONAL SELF-MOD - Attempts improvement every cycle
3. REAL CODE EVOLUTION - Uses LLM + AST mutations
4. MEASURABLE TASKS - Clear success/failure criteria

This engine:
1. Generates code to solve tasks
2. Evaluates against real test cases
3. Uses LLM to generate improved versions
4. Tracks and applies improvements
5. Actually modifies its own strategies
"""

import os
import sys
import json
import time
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# Import our LLM synthesizer
from llm_code_synthesizer import (
    CodeSynthesizer, 
    LLMConfig, 
    CodeEvaluator, 
    EvalResult,
    TASKS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RSIConfig:
    """Configuration for the RSI engine."""
    max_cycles: int = 100
    improvement_threshold: float = 1.01  # 1% improvement
    save_every: int = 5
    output_dir: str = "rsi_runs"
    
    # Self-modification
    enable_self_mod: bool = True
    self_mod_every: int = 10  # Cycles between self-mod attempts
    
    # Task rotation
    rotate_tasks: bool = True
    tasks: List[str] = field(default_factory=lambda: ["fibonacci", "sort", "is_prime"])


# =============================================================================
# RSI STATE
# =============================================================================

@dataclass
class RSIState:
    """Current state of the RSI system."""
    cycle: int = 0
    total_improvements: int = 0
    
    # Per-task tracking
    task_fitness: Dict[str, float] = field(default_factory=dict)
    task_best_code: Dict[str, str] = field(default_factory=dict)
    task_improvement_count: Dict[str, int] = field(default_factory=dict)
    
    # Overall metrics
    avg_fitness: float = 0.0
    improvement_rate: float = 0.0
    
    # History
    fitness_history: List[float] = field(default_factory=list)
    
    def update_average(self):
        """Update average fitness."""
        if self.task_fitness:
            self.avg_fitness = sum(self.task_fitness.values()) / len(self.task_fitness)
        if self.cycle > 0:
            self.improvement_rate = self.total_improvements / self.cycle
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "total_improvements": self.total_improvements,
            "avg_fitness": self.avg_fitness,
            "improvement_rate": self.improvement_rate,
            "task_fitness": self.task_fitness,
            "fitness_history": self.fitness_history[-50:],
        }
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "RSIState":
        if not path.exists():
            return cls()
        with open(path, 'r') as f:
            data = json.load(f)
        state = cls()
        state.cycle = data.get("cycle", 0)
        state.total_improvements = data.get("total_improvements", 0)
        state.avg_fitness = data.get("avg_fitness", 0.0)
        state.improvement_rate = data.get("improvement_rate", 0.0)
        state.task_fitness = data.get("task_fitness", {})
        state.fitness_history = data.get("fitness_history", [])
        return state


# =============================================================================
# RSI ENGINE
# =============================================================================

class TrueRSIEngine:
    """
    The true self-improving engine.
    
    Unlike previous attempts, this engine:
    1. Uses POSITIVE fitness scores (0-100)
    2. Always attempts improvement (no threshold blocking)
    3. Actually generates and tests code
    4. Tracks real, measurable progress
    """
    
    def __init__(self, config: Optional[RSIConfig] = None):
        self.config = config or RSIConfig()
        self.state = RSIState()
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Code synthesizer
        llm_config = LLMConfig()
        self.synthesizer = CodeSynthesizer(llm_config)
        self.evaluator = CodeEvaluator()
        
        # Initialize task states
        for task_name in self.config.tasks:
            if task_name in TASKS:
                self.state.task_fitness[task_name] = 0.0
                self.state.task_improvement_count[task_name] = 0
                
                # Load initial code
                task = TASKS[task_name]
                self.state.task_best_code[task_name] = task["initial_code"]
    
    def run(self, resume: bool = False):
        """Run the RSI loop."""
        run_name = f"run_{int(time.time())}"
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Maybe resume
        state_path = run_dir / "state.json"
        if resume and state_path.exists():
            self.state = RSIState.load(state_path)
            print(f"[RSI] Resumed from cycle {self.state.cycle}")
        
        print("=" * 70)
        print("TRUE RSI ENGINE - Starting")
        print("=" * 70)
        print(f"Tasks: {self.config.tasks}")
        print(f"Max Cycles: {self.config.max_cycles}")
        print(f"LLM Available: {self.synthesizer.llm.initialized}")
        print("=" * 70)
        
        try:
            while self.state.cycle < self.config.max_cycles:
                self._run_cycle(run_dir)
                
                # Save periodically
                if self.state.cycle % self.config.save_every == 0:
                    self.state.save(run_dir / "state.json")
                    self._save_best_code(run_dir)
                
                # Self-modification attempt
                if (self.config.enable_self_mod and 
                    self.state.cycle % self.config.self_mod_every == 0 and
                    self.state.cycle > 0):
                    self._attempt_self_modification()
        
        except KeyboardInterrupt:
            print("\n[RSI] Interrupted by user")
        
        # Final save
        self.state.save(run_dir / "state.json")
        self._save_best_code(run_dir)
        self._print_final_report(run_dir)
    
    def _run_cycle(self, run_dir: Path):
        """Run a single improvement cycle."""
        self.state.cycle += 1
        cycle = self.state.cycle
        
        # Select task (rotate or random)
        if self.config.rotate_tasks:
            task_idx = (cycle - 1) % len(self.config.tasks)
            task_name = self.config.tasks[task_idx]
        else:
            task_name = random.choice(self.config.tasks)
        
        task = TASKS[task_name]
        current_code = self.state.task_best_code.get(task_name, task["initial_code"])
        
        print(f"\n[Cycle {cycle}] Task: {task_name}")
        
        # Evaluate current
        current_result = self.evaluator.evaluate(current_code, task["test_cases"])
        print(f"  Current fitness: {current_result.fitness:.2f}")
        
        # Attempt improvement
        new_result, improved = self.synthesizer.improve(
            current_code,
            task["description"],
            task["test_cases"],
        )
        
        if improved:
            self.state.total_improvements += 1
            self.state.task_improvement_count[task_name] = \
                self.state.task_improvement_count.get(task_name, 0) + 1
            self.state.task_best_code[task_name] = new_result.code
            self.state.task_fitness[task_name] = new_result.fitness
            print(f"  ✓ IMPROVED: {current_result.fitness:.2f} -> {new_result.fitness:.2f}")
        else:
            # Even if not improved, update fitness tracking
            self.state.task_fitness[task_name] = current_result.fitness
        
        # Update overall state
        self.state.update_average()
        self.state.fitness_history.append(self.state.avg_fitness)
        
        # Progress report every 10 cycles
        if cycle % 10 == 0:
            print(f"\n[Progress] Cycle {cycle}")
            print(f"  Total improvements: {self.state.total_improvements}")
            print(f"  Improvement rate: {self.state.improvement_rate:.2%}")
            print(f"  Avg fitness: {self.state.avg_fitness:.2f}")
    
    def _attempt_self_modification(self):
        """
        Attempt to modify the RSI engine itself.
        
        This is the true "recursive" part - the engine improves
        the way it improves.
        """
        print("\n" + "=" * 50)
        print("[SELF-MOD] Attempting self-modification...")
        print("=" * 50)
        
        # Strategy 1: Adjust LLM temperature based on improvement rate
        if self.state.improvement_rate < 0.1:
            # Not improving much - increase exploration
            new_temp = min(1.0, self.synthesizer.config.temperature + 0.1)
            print(f"[SELF-MOD] Low improvement rate - increasing temperature to {new_temp:.2f}")
            self.synthesizer.config.temperature = new_temp
            self.synthesizer.llm.config.temperature = new_temp
        elif self.state.improvement_rate > 0.3:
            # Improving well - reduce exploration
            new_temp = max(0.3, self.synthesizer.config.temperature - 0.1)
            print(f"[SELF-MOD] High improvement rate - decreasing temperature to {new_temp:.2f}")
            self.synthesizer.config.temperature = new_temp
            self.synthesizer.llm.config.temperature = new_temp
        
        # Strategy 2: Adjust improvement threshold
        if self.state.cycle > 50 and self.state.total_improvements < 5:
            # Very few improvements - lower the bar
            new_threshold = max(1.001, self.config.improvement_threshold - 0.005)
            print(f"[SELF-MOD] Lowering improvement threshold to {new_threshold:.4f}")
            self.config.improvement_threshold = new_threshold
            self.synthesizer.config.min_improvement_ratio = new_threshold
        
        # Strategy 3: Add more test cases if doing well
        if self.state.avg_fitness > 90:
            print("[SELF-MOD] High fitness - would add harder test cases (not implemented)")
        
        print("[SELF-MOD] Self-modification complete")
    
    def _save_best_code(self, run_dir: Path):
        """Save best code for each task."""
        code_dir = run_dir / "best_code"
        code_dir.mkdir(exist_ok=True)
        
        for task_name, code in self.state.task_best_code.items():
            with open(code_dir / f"{task_name}.py", 'w') as f:
                f.write(f'"""\nBest code for: {task_name}\n')
                f.write(f'Fitness: {self.state.task_fitness.get(task_name, 0.0):.2f}\n')
                f.write(f'Improvements: {self.state.task_improvement_count.get(task_name, 0)}\n')
                f.write(f'"""\n\n')
                f.write(code)
    
    def _print_final_report(self, run_dir: Path):
        """Print final report."""
        print("\n" + "=" * 70)
        print("RSI ENGINE - FINAL REPORT")
        print("=" * 70)
        print(f"Total cycles: {self.state.cycle}")
        print(f"Total improvements: {self.state.total_improvements}")
        print(f"Improvement rate: {self.state.improvement_rate:.2%}")
        print(f"Final avg fitness: {self.state.avg_fitness:.2f}")
        print()
        print("Per-task results:")
        for task_name in self.config.tasks:
            fitness = self.state.task_fitness.get(task_name, 0.0)
            improvements = self.state.task_improvement_count.get(task_name, 0)
            print(f"  {task_name}: fitness={fitness:.2f}, improvements={improvements}")
        print()
        print(f"Results saved to: {run_dir}")
        print("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def test():
    """Quick test to verify everything works."""
    print("=" * 60)
    print("True RSI Engine - Quick Test")
    print("=" * 60)
    
    config = RSIConfig(
        max_cycles=5,
        tasks=["fibonacci"],
        enable_self_mod=False,
    )
    
    engine = TrueRSIEngine(config)
    engine.run()
    
    print("\n[TEST] Results:")
    print(f"  Cycles completed: {engine.state.cycle}")
    print(f"  Improvements: {engine.state.total_improvements}")
    print(f"  Avg fitness: {engine.state.avg_fitness:.2f}")
    
    assert engine.state.cycle == 5, "Should complete 5 cycles"
    assert engine.state.avg_fitness >= 0, "Fitness should be positive"
    
    print("\n[TEST] PASSED!")
    print(f"SELF_TEST_FITNESS:{engine.state.avg_fitness}")


def run(max_cycles: int = 50, tasks: str = "all"):
    """Run the full RSI engine."""
    task_list = list(TASKS.keys()) if tasks == "all" else tasks.split(",")
    
    config = RSIConfig(
        max_cycles=max_cycles,
        tasks=task_list,
        enable_self_mod=True,
        self_mod_every=10,
    )
    
    engine = TrueRSIEngine(config)
    engine.run()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python true_rsi_engine.py test")
        print("  python true_rsi_engine.py run [max_cycles] [tasks]")
        print("Example:")
        print("  python true_rsi_engine.py run 100 fibonacci,sort")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "test":
        test()
    elif cmd == "run":
        max_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        tasks = sys.argv[3] if len(sys.argv) > 3 else "all"
        run(max_cycles, tasks)
    else:
        print(f"Unknown command: {cmd}")
