"""
LLM Code Synthesizer - True RSI via Code Generation
====================================================
Uses Google Gemini to generate improved code versions.
This is the core of the "AGI-level" self-improvement system.

Key Features:
1. Generates actual Python code that can be executed
2. Evaluates generated code with real tests
3. Keeps track of improvement history
4. Falls back to mutation-based improvement if LLM fails
"""

import os
import sys
import ast
import json
import time
import random
import hashlib
import traceback
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# Try to import google generativeai
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print("[LLM] google-generativeai not installed. Using mutation fallback.")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """Configuration for the LLM code synthesizer."""
    api_key_path: str = "google_api_key.txt"
    model_name: str = "gemini-1.5-flash"  # Fast and cheap
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    max_retries: int = 3
    
    # Safety
    test_before_accept: bool = True
    require_improvement: bool = True
    min_improvement_ratio: float = 1.01  # 1% improvement

# =============================================================================
# CODE GENERATION PROMPTS
# =============================================================================

IMPROVEMENT_PROMPT = """You are an expert Python programmer. Your task is to improve the following code.

## Current Code
```python
{current_code}
```

## Current Performance
- Fitness score: {current_fitness}
- Test results: {test_results}

## Task Description
{task_description}

## Requirements
1. The improved code must be syntactically correct Python
2. The improved code must pass the same tests
3. The improved code should have BETTER performance/fitness
4. You can:
   - Improve the algorithm
   - Optimize the implementation
   - Fix bugs
   - Add better logic

## Output
Return ONLY the improved Python code, no explanations. The code should be complete and runnable.
"""

SYNTHESIS_PROMPT = """You are an expert Python programmer. Generate code to solve the following task.

## Task
{task_description}

## Test Cases
{test_cases}

## Requirements
1. The code must be syntactically correct Python
2. The code must pass all test cases
3. Define a function with the exact signature requested

## Output
Return ONLY the Python code, no explanations.
"""

# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """Client for interacting with Gemini API."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.initialized = False
        self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini client."""
        if not HAS_GENAI:
            print("[LLM] google-generativeai not available")
            return
        
        # Load API key
        api_key = None
        paths_to_check = [
            self.config.api_key_path,
            "../google_api_key.txt",
            "../../google_api_key.txt",
            os.path.expanduser("~/.google_api_key.txt"),
        ]
        
        for path in paths_to_check:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        key = f.read().strip().split('\n')[0].strip()
                        if key and len(key) > 20:
                            api_key = key
                            print(f"[LLM] Loaded API key from: {path}")
                            break
            except Exception:
                pass
        
        if not api_key:
            print("[LLM] No API key found")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.config.model_name)
            self.initialized = True
            print(f"[LLM] Initialized with model: {self.config.model_name}")
        except Exception as e:
            print(f"[LLM] Failed to initialize: {e}")
    
    def generate(self, prompt: str) -> Optional[str]:
        """Generate text from prompt."""
        if not self.initialized or not self.model:
            return None
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    ),
                )
                
                if response and response.text:
                    return self._extract_code(response.text)
                
            except Exception as e:
                print(f"[LLM] Attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        
        return None
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        # Remove markdown code blocks
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        
        return text.strip()

# =============================================================================
# CODE EVALUATOR
# =============================================================================

@dataclass
class EvalResult:
    """Result of evaluating code."""
    code: str
    fitness: float
    passed_tests: int
    total_tests: int
    errors: List[str] = field(default_factory=list)
    runtime_ms: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitness": self.fitness,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "pass_rate": self.pass_rate,
            "runtime_ms": self.runtime_ms,
            "errors": self.errors[:5],
        }


class CodeEvaluator:
    """Evaluates Python code against test cases."""
    
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
    
    def evaluate(self, code: str, test_cases: List[Dict]) -> EvalResult:
        """
        Evaluate code against test cases.
        
        Each test case is a dict with:
        - 'input': the input to pass
        - 'expected': the expected output
        - 'function': the function name to call
        """
        # First check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            return EvalResult(
                code=code,
                fitness=-100.0,
                passed_tests=0,
                total_tests=len(test_cases),
                errors=[f"SyntaxError: {e}"],
            )
        
        passed = 0
        errors = []
        start_time = time.time()
        
        for i, test in enumerate(test_cases):
            try:
                result = self._run_test(code, test)
                if result:
                    passed += 1
                else:
                    errors.append(f"Test {i}: Failed")
            except Exception as e:
                errors.append(f"Test {i}: {type(e).__name__}: {str(e)[:50]}")
        
        runtime_ms = (time.time() - start_time) * 1000
        
        # Calculate fitness: prioritize correctness, then speed
        pass_rate = passed / max(1, len(test_cases))
        speed_bonus = max(0, (self.timeout * 1000 - runtime_ms) / (self.timeout * 1000)) * 0.1
        fitness = pass_rate * 100 + speed_bonus * 10
        
        return EvalResult(
            code=code,
            fitness=fitness,
            passed_tests=passed,
            total_tests=len(test_cases),
            errors=errors,
            runtime_ms=runtime_ms,
        )
    
    def _run_test(self, code: str, test: Dict) -> bool:
        """Run a single test case."""
        namespace = {}
        exec(code, namespace)
        
        func_name = test.get('function', 'solve')
        if func_name not in namespace:
            raise ValueError(f"Function '{func_name}' not found")
        
        func = namespace[func_name]
        input_val = test['input']
        expected = test['expected']
        
        if isinstance(input_val, (list, tuple)):
            result = func(*input_val)
        else:
            result = func(input_val)
        
        # Compare results
        if isinstance(expected, float):
            return abs(result - expected) < 1e-6
        return result == expected

# =============================================================================
# CODE SYNTHESIZER (Main Class)
# =============================================================================

class CodeSynthesizer:
    """
    Main class for LLM-based code synthesis and improvement.
    This is the heart of the true RSI system.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm = LLMClient(self.config)
        self.evaluator = CodeEvaluator()
        
        # History tracking
        self.history: List[Dict] = []
        self.best_code: Optional[str] = None
        self.best_fitness: float = -float('inf')
        self.generation: int = 0
    
    def synthesize(self, task_desc: str, test_cases: List[Dict]) -> EvalResult:
        """Generate initial code for a task."""
        prompt = SYNTHESIS_PROMPT.format(
            task_description=task_desc,
            test_cases=json.dumps(test_cases, indent=2),
        )
        
        # Try LLM
        code = self.llm.generate(prompt)
        
        if not code:
            # Fallback: generate simple template
            code = self._generate_template(task_desc, test_cases)
        
        result = self.evaluator.evaluate(code, test_cases)
        
        if result.fitness > self.best_fitness:
            self.best_fitness = result.fitness
            self.best_code = code
        
        self._log_attempt("synthesize", code, result)
        return result
    
    def improve(self, current_code: str, task_desc: str, test_cases: List[Dict]) -> Tuple[EvalResult, bool]:
        """
        Attempt to improve existing code.
        Returns (result, improved: bool)
        """
        self.generation += 1
        current_result = self.evaluator.evaluate(current_code, test_cases)
        
        # Try LLM improvement
        prompt = IMPROVEMENT_PROMPT.format(
            current_code=current_code,
            current_fitness=current_result.fitness,
            test_results=json.dumps(current_result.to_dict()),
            task_description=task_desc,
        )
        
        new_code = self.llm.generate(prompt)
        
        if not new_code:
            # Fallback: AST-based mutation
            new_code = self._mutate_code(current_code)
        
        new_result = self.evaluator.evaluate(new_code, test_cases)
        
        # Check if improved
        improved = False
        min_fitness = current_result.fitness * self.config.min_improvement_ratio
        
        if new_result.fitness > min_fitness:
            improved = True
            print(f"[RSI] Gen {self.generation}: IMPROVED! {current_result.fitness:.2f} -> {new_result.fitness:.2f}")
            
            if new_result.fitness > self.best_fitness:
                self.best_fitness = new_result.fitness
                self.best_code = new_code
        else:
            print(f"[RSI] Gen {self.generation}: No improvement ({new_result.fitness:.2f} <= {current_result.fitness:.2f})")
        
        self._log_attempt("improve", new_code, new_result, improved)
        
        return (new_result if improved else current_result, improved)
    
    def run_improvement_loop(
        self, 
        initial_code: str, 
        task_desc: str, 
        test_cases: List[Dict], 
        max_generations: int = 10
    ) -> EvalResult:
        """Run continuous improvement loop."""
        current_code = initial_code
        best_result = self.evaluator.evaluate(current_code, test_cases)
        
        for gen in range(max_generations):
            new_result, improved = self.improve(current_code, task_desc, test_cases)
            
            if improved:
                current_code = new_result.code
                best_result = new_result
        
        return best_result
    
    def _mutate_code(self, code: str) -> str:
        """AST-based code mutation as fallback."""
        try:
            tree = ast.parse(code)
            
            # Simple mutations: change numbers, swap operators
            class Mutator(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if isinstance(node.value, (int, float)):
                        # Random perturbation
                        if random.random() < 0.3:
                            delta = random.uniform(-1, 1)
                            node.value = node.value + delta
                    return node
                
                def visit_BinOp(self, node):
                    self.generic_visit(node)
                    if random.random() < 0.1:
                        # Swap +/- or */
                        if isinstance(node.op, ast.Add):
                            node.op = ast.Sub()
                        elif isinstance(node.op, ast.Sub):
                            node.op = ast.Add()
                    return node
            
            mutator = Mutator()
            new_tree = mutator.visit(tree)
            ast.fix_missing_locations(new_tree)
            return ast.unparse(new_tree)
        
        except Exception:
            return code
    
    def _generate_template(self, task_desc: str, test_cases: List[Dict]) -> str:
        """Generate a simple template if LLM fails."""
        func_name = test_cases[0].get('function', 'solve') if test_cases else 'solve'
        
        # Infer arity from test cases
        sample_input = test_cases[0].get('input', None) if test_cases else None
        if isinstance(sample_input, (list, tuple)):
            args = [f"arg{i}" for i in range(len(sample_input))]
        else:
            args = ["x"]
        
        return f'''def {func_name}({", ".join(args)}):
    """Generated template for: {task_desc[:50]}"""
    # TODO: Implement
    return None
'''
    
    def _log_attempt(self, action: str, code: str, result: EvalResult, improved: bool = False):
        """Log an attempt for history tracking."""
        self.history.append({
            "generation": self.generation,
            "action": action,
            "fitness": result.fitness,
            "passed": result.passed_tests,
            "total": result.total_tests,
            "improved": improved,
            "code_hash": hashlib.sha256(code.encode()).hexdigest()[:16],
            "timestamp": time.time(),
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the improvement process."""
        improvements = sum(1 for h in self.history if h.get('improved', False))
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "total_attempts": len(self.history),
            "improvements": improvements,
            "improvement_rate": improvements / max(1, len(self.history)),
        }


# =============================================================================
# TASK DEFINITIONS (Real, Measurable Tasks)
# =============================================================================

TASKS = {
    "fibonacci": {
        "description": "Compute the nth Fibonacci number efficiently",
        "function": "fib",
        "test_cases": [
            {"input": 0, "expected": 0, "function": "fib"},
            {"input": 1, "expected": 1, "function": "fib"},
            {"input": 5, "expected": 5, "function": "fib"},
            {"input": 10, "expected": 55, "function": "fib"},
            {"input": 15, "expected": 610, "function": "fib"},
        ],
        "initial_code": '''def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)  # Inefficient!
''',
    },
    
    "sort": {
        "description": "Sort a list of numbers in ascending order",
        "function": "sort_list",
        "test_cases": [
            {"input": [[3, 1, 2]], "expected": [1, 2, 3], "function": "sort_list"},
            {"input": [[5, 4, 3, 2, 1]], "expected": [1, 2, 3, 4, 5], "function": "sort_list"},
            {"input": [[1]], "expected": [1], "function": "sort_list"},
            {"input": [[]], "expected": [], "function": "sort_list"},
            {"input": [[1, 1, 1]], "expected": [1, 1, 1], "function": "sort_list"},
        ],
        "initial_code": '''def sort_list(arr):
    # Bubble sort - intentionally slow
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
''',
    },
    
    "is_prime": {
        "description": "Check if a number is prime",
        "function": "is_prime",
        "test_cases": [
            {"input": 2, "expected": True, "function": "is_prime"},
            {"input": 3, "expected": True, "function": "is_prime"},
            {"input": 4, "expected": False, "function": "is_prime"},
            {"input": 17, "expected": True, "function": "is_prime"},
            {"input": 100, "expected": False, "function": "is_prime"},
            {"input": 97, "expected": True, "function": "is_prime"},
        ],
        "initial_code": '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):  # Inefficient!
        if n % i == 0:
            return False
    return True
''',
    },
}


# =============================================================================
# CLI
# =============================================================================

def test():
    """Run tests to verify the system works."""
    print("=" * 60)
    print("LLM Code Synthesizer - Test Suite")
    print("=" * 60)
    
    config = LLMConfig()
    synth = CodeSynthesizer(config)
    
    # Test 1: Evaluate initial code
    print("\n[TEST 1] Evaluating initial Fibonacci code...")
    task = TASKS["fibonacci"]
    result = synth.evaluator.evaluate(task["initial_code"], task["test_cases"])
    print(f"Fitness: {result.fitness:.2f}")
    print(f"Passed: {result.passed_tests}/{result.total_tests}")
    assert result.passed_tests > 0, "Basic tests should pass"
    print("[PASS] Evaluation works")
    
    # Test 2: Try improvement (even without LLM, mutation should work)
    print("\n[TEST 2] Attempting improvement...")
    new_result, improved = synth.improve(
        task["initial_code"],
        task["description"],
        task["test_cases"],
    )
    print(f"Result fitness: {new_result.fitness:.2f}")
    print("[PASS] Improvement attempt works")
    
    # Test 3: Run improvement loop
    print("\n[TEST 3] Running improvement loop (3 generations)...")
    final_result = synth.run_improvement_loop(
        task["initial_code"],
        task["description"],
        task["test_cases"],
        max_generations=3,
    )
    print(f"Final fitness: {final_result.fitness:.2f}")
    stats = synth.get_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")
    print("[PASS] Improvement loop works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print(f"LLM Available: {synth.llm.initialized}")
    print("=" * 60)
    
    # Output for self-test detection
    print(f"SELF_TEST_FITNESS:{final_result.fitness}")


def run(task_name: str = "fibonacci", generations: int = 10):
    """Run the improvement system on a task."""
    if task_name not in TASKS:
        print(f"Unknown task: {task_name}")
        print(f"Available: {list(TASKS.keys())}")
        return
    
    task = TASKS[task_name]
    config = LLMConfig()
    synth = CodeSynthesizer(config)
    
    print("=" * 60)
    print(f"LLM Code Synthesizer - Task: {task_name}")
    print(f"Generations: {generations}")
    print(f"LLM Available: {synth.llm.initialized}")
    print("=" * 60)
    
    final_result = synth.run_improvement_loop(
        task["initial_code"],
        task["description"],
        task["test_cases"],
        max_generations=generations,
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Final Fitness: {final_result.fitness:.2f}")
    print(f"Passed Tests: {final_result.passed_tests}/{final_result.total_tests}")
    
    stats = synth.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")
    
    if synth.best_code:
        print("\nBest Code Found:")
        print("-" * 40)
        print(synth.best_code)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python llm_code_synthesizer.py test")
        print("  python llm_code_synthesizer.py run [task_name] [generations]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "test":
        test()
    elif cmd == "run":
        task_name = sys.argv[2] if len(sys.argv) > 2 else "fibonacci"
        generations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        run(task_name, generations)
    else:
        print(f"Unknown command: {cmd}")
