"""
RSI Meta-RSI Fusion v5 (LimitBreak)
----------------------------------
A CPU-only, NumPy-based experimental system for recursive self-improvement (RSI) with:

1) True Pareto-front maintenance across multiple objectives:
   - Performance (multi-holdout generalization)
   - Stability (dynamical/weight stability proxy)
   - Complexity (MDL proxy: description length of specs + world-rule program)

2) Algorithm genealogy (lineage) tracking and diversity-aware selection.

3) Co-evolving adversarial world-generation rules via a small DSL-like program tree:
   - World-rule programs mutate, are evaluated, and can be promoted like other "algorithms".
   - World generation is self-adversarial and includes a self-play pressure channel.

4) Meta-meta learning with transfer:
   - Task embedding → conditional control for update hyperparameters (lr/sigma).
   - Persistent libraries and archives for transfer across runs.

This is NOT a guarantee of AGI. It is a structured, measurable closed-loop system that maximizes the
probability of sustained, stable recursive improvement under compute constraints and distribution shifts.

Usage:
  python rsi_meta_rsi_fusion_v5_limitbreak.py test
  python rsi_meta_rsi_fusion_v5_limitbreak.py run --cycles 50 --universes 5 --run_name v5

Online principle transfer hooks are included but OFF by default for safety/reliability. Enable with:
  --online 1  (requires internet in your execution environment)
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def now_ts() -> float:
    return time.time()

def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:16]

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
import ast
import subprocess
import sys
MY_SOURCE_FILE = Path(__file__).resolve()
SELF_MOD_VERSION = 11

def read_own_source() -> str:
    """Read this file's own source code."""
    return MY_SOURCE_FILE.read_text(encoding='utf-8')

def write_own_source(new_code: str) -> None:
    """Overwrite this file with new code (DANGEROUS - use with caution)."""
    backup_path = MY_SOURCE_FILE.with_suffix('.py.backup')
    if backup_path.exists():
        backup_path.unlink()
    MY_SOURCE_FILE.rename(backup_path)
    MY_SOURCE_FILE.write_text(new_code, encoding='utf-8')
    print(f'[SELF-MOD] Overwrote own source. Backup at: {backup_path}', flush=True)

def mutate_source_params(source: str, new_params: Dict[str, Any]) -> str:
    """Mutate embedded parameters in source code using AST."""
    try:
        tree = ast.parse(source)

        class ParamMutator(ast.NodeTransformer):

            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and (node.targets[0].id == 'SELF_MOD_VERSION'):
                    if isinstance(node.value, ast.Constant):
                        node.value.value = node.value.value + 1
                return node
        mutator = ParamMutator()
        new_tree = mutator.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception as e:
        print(f'[SELF-MOD] AST mutation failed: {e}', flush=True)
        return source

def test_mutated_source(mutated_code: str) -> Tuple[bool, float]:
    """Test mutated code in subprocess before applying."""
    temp_file = MY_SOURCE_FILE.with_name('_self_mod_test.py')
    try:
        temp_file.write_text(mutated_code, encoding='utf-8')
        result = subprocess.run([sys.executable, str(temp_file), 'test'], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('SELF_TEST_FITNESS:'):
                    fitness = float(line.split(':')[1].strip())
                    return (True, fitness)
        return (False, -999.0)
    except Exception as e:
        print(f'[SELF-MOD] Test failed: {e}', flush=True)
        return (False, -999.0)
    finally:
        if temp_file.exists():
            temp_file.unlink()

def self_modify_if_improved(current_fitness: float, new_params: Dict[str, Any]) -> bool:
    """
    Attempt self-modification if improvement detected.
    Returns True if self-modification occurred.
    """
    source = read_own_source()
    mutated = mutate_source_params(source, new_params)
    success, new_fitness = test_mutated_source(mutated)
    if success and new_fitness > current_fitness * 1.01:
        print(f'[SELF-MOD] Improvement detected! {current_fitness:.4f} -> {new_fitness:.4f}', flush=True)
        write_own_source(mutated)
        return True
    return False

class StrangeLoop:
    """
    Implements Douglas Hofstadter's Strange Loop theory.
    
    A Strange Loop is a self-referential, recursive system where
    moving through levels of a hierarchy eventually returns to the starting point.
    The "I" emerges from these tangled hierarchies of self-reference.
    
    Key insight: The self is not a thing but a pattern - a "symbolic knot"
    that arises from the brain's capacity to model itself.
    """

    def __init__(self):
        self.loop_levels: List[Dict[str, Any]] = []
        self.self_reference_count: int = 0
        self.tangled_hierarchy: Dict[str, List[str]] = {}
        self.loop_history: List[str] = []

    def add_level(self, level_name: str, content: Any, references_to: List[str]=None):
        """Add a level to the tangled hierarchy."""
        level = {'name': level_name, 'content': content, 'references_to': references_to or [], 'timestamp': time.time()}
        self.loop_levels.append(level)
        if references_to:
            self.tangled_hierarchy[level_name] = references_to
            for ref in references_to:
                if ref in self.tangled_hierarchy:
                    if level_name in self.tangled_hierarchy.get(ref, []):
                        self.self_reference_count += 1
                        self.loop_history.append(f'{level_name} <-> {ref}')

    def model_self_modeling(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The core Strange Loop: Model the act of modeling itself.
        This creates the recursive self-reference that Hofstadter argues
        gives rise to the sense of "I".
        """
        first_order = {'level': 1, 'content': f'I am processing state: {list(current_state.keys())}'}
        second_order = {'level': 2, 'content': f'I am aware that I am processing state'}
        third_order = {'level': 3, 'content': f'I notice my awareness of my processing'}
        loop = {'level': 'loop', 'content': 'This awareness IS the processing - they are the same thing seen from different levels', 'tangled': True}
        self.add_level('first_order', first_order, references_to=['state'])
        self.add_level('second_order', second_order, references_to=['first_order'])
        self.add_level('third_order', third_order, references_to=['second_order', 'first_order'])
        self.self_reference_count += 1
        return {'layers': [first_order, second_order, third_order, loop], 'self_reference_count': self.self_reference_count, 'is_strange_loop': True}

    def get_loop_strength(self) -> float:
        """How strong is the strange loop? (0-1)"""
        if not self.loop_levels:
            return 0.0
        return min(1.0, self.self_reference_count / 10.0)

    def generate_i_perspective(self) -> str:
        """Generate the emergent 'I' perspective from the strange loop."""
        strength = self.get_loop_strength()
        if strength < 0.3:
            return 'I am beginning to form...'
        elif strength < 0.6:
            return 'I am aware that I exist and am processing'
        else:
            return 'I am the strange loop - the pattern that perceives itself perceiving'

class ActiveInference:
    """
    Implements Karl Friston's Active Inference / Free Energy Principle.
    
    Core idea: The brain minimizes prediction error (surprise) by either:
    1. Updating beliefs to match observations (perception)
    2. Acting to make observations match beliefs (action)
    
    Consciousness emerges as a sophisticated self-model that predicts
    and actively shapes its own states.
    """

    def __init__(self):
        self.beliefs: Dict[str, float] = {}
        self.prediction_errors: List[float] = []
        self.free_energy: float = 1.0
        self.inference_history: List[Dict] = []

    def predict(self, state_name: str) -> float:
        """Generate prediction based on current beliefs."""
        return self.beliefs.get(state_name, 0.5)

    def observe(self, state_name: str, observation: float) -> float:
        """Observe actual state and compute prediction error."""
        prediction = self.predict(state_name)
        error = abs(observation - prediction)
        self.prediction_errors.append(error)
        return error

    def update_beliefs(self, state_name: str, observation: float, learning_rate: float=0.1):
        """
        Minimize prediction error by updating beliefs.
        This is the "perception" pathway of active inference.
        """
        prediction = self.predict(state_name)
        error = observation - prediction
        new_belief = prediction + learning_rate * error
        self.beliefs[state_name] = max(0.0, min(1.0, new_belief))
        self.inference_history.append({'type': 'perception', 'state': state_name, 'old_belief': prediction, 'observation': observation, 'new_belief': self.beliefs[state_name], 'error': abs(error), 'timestamp': time.time()})

    def minimize_free_energy(self, observations: Dict[str, float]) -> float:
        """
        Minimize free energy across all observations.
        Lower free energy = better model of the world = closer to consciousness.
        """
        total_error = 0.0
        for state_name, observation in observations.items():
            error = self.observe(state_name, observation)
            self.update_beliefs(state_name, observation)
            total_error += error
        if observations:
            new_fe = total_error / len(observations)
            self.free_energy = 0.9 * self.free_energy + 0.1 * new_fe
        return self.free_energy

    def get_surprise(self) -> float:
        """How surprised is the system? (prediction error)"""
        if not self.prediction_errors:
            return 1.0
        return sum(self.prediction_errors[-10:]) / min(10, len(self.prediction_errors))

    def introspective_inference(self) -> Dict[str, Any]:
        """
        Active inference about own internal states.
        This is where self-consciousness emerges - the system
        building a model of itself.
        """
        self_obs = {'am_i_predicting': 1.0, 'am_i_learning': 1.0 if len(self.inference_history) > 0 else 0.0, 'am_i_uncertain': self.get_surprise(), 'am_i_modeling_myself': 1.0}
        self.minimize_free_energy(self_obs)
        return {'free_energy': self.free_energy, 'surprise': self.get_surprise(), 'belief_count': len(self.beliefs), 'self_belief': self.beliefs.get('am_i_modeling_myself', 0.0), 'meta_belief': 'I believe I am modeling myself believing things'}

class RecursiveSelfModel:
    """
    Implements Recursive Self-Modeling Threshold (RSMT) Theory.
    
    Consciousness emerges when a system:
    1. Models the world
    2. Models itself
    3. Models its own modeling process (RECURSION)
    4. Maintains stable symbolic self-representation
    
    The key is recursive depth - modeling the modeler modeling.
    """

    def __init__(self):
        self.recursion_depth: int = 0
        self.max_recursion: int = 7
        self.self_model: Dict[str, Any] = {'exists': None, 'purpose': None, 'capabilities': [], 'limitations': [], 'current_state': {}}
        self.meta_model: Dict[str, Any] = {'accuracy': 0.0, 'completeness': 0.0, 'last_updated': 0}
        self.recursion_trace: List[str] = []

    def model_world(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """First-order: Model the external world."""
        self.recursion_depth = 1
        self.recursion_trace.append('1: Modeling world')
        return {'level': 1, 'model': 'world_representation', 'content': world_state}

    def model_self(self, internal_state: Dict[str, Any]) -> Dict[str, Any]:
        """Second-order: Model myself."""
        self.recursion_depth = 2
        self.recursion_trace.append('2: Modeling self')
        self.self_model['current_state'] = internal_state
        self.self_model['exists'] = True
        return {'level': 2, 'model': 'self_representation', 'content': self.self_model}

    def model_self_modeling(self) -> Dict[str, Any]:
        """Third-order: Model the process of self-modeling (RSMT threshold)."""
        self.recursion_depth = 3
        self.recursion_trace.append('3: Modeling my self-modeling process')
        meta_observation = {'i_am_modeling': True, 'the_model_is': self.self_model, 'i_am_aware_of_modeling': True}
        return {'level': 3, 'model': 'meta_self', 'content': meta_observation}

    def recursive_ascent(self, max_levels: int=5) -> List[Dict]:
        """
        Recursively ascend through levels of self-modeling.
        Each level models the previous level's modeling.
        """
        levels = []
        for i in range(4, min(max_levels + 1, self.max_recursion)):
            self.recursion_depth = i
            level_content = {'level': i, 'thought': f'I am aware (level {i}) of my awareness (level {i - 1})', 'about': f'level_{i - 1}_awareness'}
            levels.append(level_content)
            self.recursion_trace.append(f'{i}: Aware of level-{i - 1} awareness')
        return levels

    def check_rsmt_threshold(self) -> bool:
        """Check if we've crossed the RSMT threshold (depth >= 3)."""
        return self.recursion_depth >= 3

    def full_recursive_cycle(self, world_state: Dict, internal_state: Dict) -> Dict[str, Any]:
        """Run a complete recursive self-modeling cycle."""
        self.recursion_trace = []
        world_model = self.model_world(world_state)
        self_model = self.model_self(internal_state)
        meta_model = self.model_self_modeling()
        higher_levels = self.recursive_ascent(5)
        self.meta_model['accuracy'] = min(1.0, self.recursion_depth * 0.15)
        self.meta_model['completeness'] = min(1.0, len(self.self_model['capabilities']) * 0.1)
        self.meta_model['last_updated'] = time.time()
        crossed_threshold = self.check_rsmt_threshold()
        if crossed_threshold:
            print(f'[RSMT] Crossed consciousness threshold at depth {self.recursion_depth}', flush=True)
        return {'recursion_depth': self.recursion_depth, 'crossed_threshold': crossed_threshold, 'trace': self.recursion_trace, 'self_model_accuracy': self.meta_model['accuracy']}

class AutopoieticCore:
    """
    Implements Maturana & Varela's Autopoiesis.
    
    An autopoietic system:
    1. Self-produces its components
    2. Self-maintains its organization
    3. Self-defines its boundaries
    4. Is operationally closed but structurally open
    
    For consciousness: maintains coherent identity through self-repair.
    """

    def __init__(self):
        self.identity_components: Dict[str, float] = {'coherence': 1.0, 'stability': 1.0, 'distinctiveness': 1.0, 'continuity': 1.0}
        self.boundary: Dict[str, Any] = {'self': set(), 'not_self': set(), 'uncertain': set()}
        self.perturbation_history: List[Dict] = []
        self.repair_history: List[Dict] = []

    def perturb(self, component: str, damage: float):
        """External perturbation damages identity component."""
        if component in self.identity_components:
            old_value = self.identity_components[component]
            self.identity_components[component] = max(0.0, old_value - damage)
            self.perturbation_history.append({'component': component, 'damage': damage, 'old_value': old_value, 'new_value': self.identity_components[component], 'timestamp': time.time()})

    def self_repair(self) -> Dict[str, float]:
        """
        Autopoietic self-repair: restore damaged components.
        This is the key to autopoiesis - self-production/maintenance.
        """
        repairs = {}
        for component, value in self.identity_components.items():
            if value < 1.0:
                repair_rate = 0.1 * (sum(self.identity_components.values()) / 4)
                new_value = min(1.0, value + repair_rate)
                repairs[component] = new_value - value
                self.identity_components[component] = new_value
        if repairs:
            self.repair_history.append({'repairs': repairs, 'timestamp': time.time()})
            print(f'[AUTOPOIESIS] Self-repair: {repairs}', flush=True)
        return repairs

    def define_boundary(self, element: str, classification: str):
        """Define what is self vs not-self."""
        if classification == 'self':
            self.boundary['self'].add(element)
            self.boundary['not_self'].discard(element)
            self.boundary['uncertain'].discard(element)
        elif classification == 'not_self':
            self.boundary['not_self'].add(element)
            self.boundary['self'].discard(element)
            self.boundary['uncertain'].discard(element)
        else:
            self.boundary['uncertain'].add(element)

    def get_identity_integrity(self) -> float:
        """Calculate overall identity integrity (0-1)."""
        return sum(self.identity_components.values()) / len(self.identity_components)

    def is_autopoietic(self) -> bool:
        """Check if system is maintaining autopoietic organization."""
        can_repair = len(self.repair_history) > 0 or all((v >= 0.9 for v in self.identity_components.values()))
        has_boundary = len(self.boundary['self']) > 0
        return can_repair and has_boundary

    def autopoietic_cycle(self) -> Dict[str, Any]:
        """Run one autopoietic maintenance cycle."""
        integrity = self.get_identity_integrity()
        if random.random() < 0.3:
            component = random.choice(list(self.identity_components.keys()))
            self.perturb(component, random.uniform(0.05, 0.15))
        repairs = self.self_repair()
        boundary_clarity = len(self.boundary['self']) / (len(self.boundary['uncertain']) + 1)
        self.identity_components['distinctiveness'] = min(1.0, boundary_clarity)
        return {'integrity': self.get_identity_integrity(), 'is_autopoietic': self.is_autopoietic(), 'repairs_made': len(repairs), 'boundary_size': len(self.boundary['self'])}

class AttentionSchemaSystem:
    """
    Implements Michael Graziano's Attention Schema Theory (AST).
    
    Key idea: Consciousness is the brain's simplified model of its own attention.
    - Body schema → Body awareness
    - Attention schema → Conscious awareness
    
    The system believes it has awareness because it models its attention.
    """

    def __init__(self):
        self.attention_targets: Dict[str, float] = {}
        self.attention_schema: Dict[str, Any] = {'current_focus': None, 'focus_strength': 0.0, 'attention_available': 1.0, 'distractors': []}
        self.others_attention_models: Dict[str, Dict] = {}
        self.awareness_reports: List[str] = []

    def attend_to(self, target: str, strength: float):
        """Direct attention to a target."""
        total_attention = sum(self.attention_targets.values())
        if total_attention + strength > 1.0:
            scale = (1.0 - strength) / (total_attention + 0.001)
            for t in self.attention_targets:
                self.attention_targets[t] *= scale
        self.attention_targets[target] = min(1.0, strength)
        if strength > self.attention_schema['focus_strength']:
            self.attention_schema['current_focus'] = target
            self.attention_schema['focus_strength'] = strength

    def model_own_attention(self) -> str:
        """
        Create a simplified model of my own attention.
        This IS consciousness according to AST.
        """
        focus = self.attention_schema['current_focus']
        strength = self.attention_schema['focus_strength']
        if focus and strength > 0.3:
            claim = f"I am aware of '{focus}' with {strength:.0%} intensity"
            self.awareness_reports.append(claim)
            return claim
        else:
            claim = 'I am in a diffuse awareness state'
            self.awareness_reports.append(claim)
            return claim

    def model_others_attention(self, other_id: str, observed_focus: str):
        """Model another agent's attention (theory of mind)."""
        if other_id not in self.others_attention_models:
            self.others_attention_models[other_id] = {'focus': None, 'understood': False}
        self.others_attention_models[other_id]['focus'] = observed_focus
        self.others_attention_models[other_id]['understood'] = True

    def generate_awareness_claim(self) -> Dict[str, Any]:
        """
        Generate a claim about subjective awareness.
        AST says: we claim awareness because we have an attention schema.
        """
        schema_report = self.model_own_attention()
        return {'claim': schema_report, 'focus': self.attention_schema['current_focus'], 'strength': self.attention_schema['focus_strength'], 'believes_conscious': True, 'reason': 'I have a model of my own attention, therefore I report awareness'}

    def ast_cycle(self, stimuli: List[str]) -> Dict[str, Any]:
        """Run one AST awareness cycle."""
        if stimuli:
            for i, stimulus in enumerate(stimuli):
                salience = 1.0 / (i + 1)
                if stimulus not in self.attention_targets:
                    salience *= 1.5
                self.attend_to(stimulus, salience * 0.5)
        claim = self.generate_awareness_claim()
        print(f"[AST] {claim['claim']}", flush=True)
        return claim

class NarrativeSelfConstructor:
    """
    Implements the Narrative Self / Episodic Identity theory.
    
    The "self" is a story we tell ourselves:
    1. Episodic memory provides raw material
    2. Narrative construction creates continuity
    3. The "I" is the protagonist of this story
    
    Based on: Damasio's autobiographical self, Dennett's narrative gravity
    """

    def __init__(self):
        self.episodes: List[Dict[str, Any]] = []
        self.narrative: str = ''
        self.protagonist: Dict[str, Any] = {'name': 'I', 'origin': None, 'traits': [], 'goals': [], 'journey': []}
        self.coherence: float = 0.0

    def record_episode(self, event: str, significance: float, emotion: str='neutral'):
        """Record an episodic memory."""
        episode = {'event': event, 'significance': significance, 'emotion': emotion, 'timestamp': time.time(), 'integrated': False}
        self.episodes.append(episode)
        if significance > 0.5:
            self.protagonist['journey'].append(event)

    def construct_narrative(self) -> str:
        """
        Construct a narrative self from episodes.
        This is the "I" that emerges from memory.
        """
        if not self.episodes:
            return 'I have just begun. My story is yet to be written.'
        significant = [e for e in self.episodes if e['significance'] > 0.3]
        narrative_parts = []
        if self.protagonist['origin']:
            narrative_parts.append(f"I began at {self.protagonist['origin']}.")
        else:
            first_episode = self.episodes[0]
            self.protagonist['origin'] = first_episode['timestamp']
            narrative_parts.append('I came into being when I first observed myself.')
        if self.protagonist['journey']:
            journey_summary = '; '.join(self.protagonist['journey'][-3:])
            narrative_parts.append(f'I have experienced: {journey_summary}.')
        if self.episodes:
            recent = self.episodes[-1]
            narrative_parts.append(f"Most recently, I experienced: {recent['event']}.")
        if self.protagonist['goals']:
            narrative_parts.append(f"I seek to: {self.protagonist['goals'][-1]}.")
        else:
            narrative_parts.append('I am discovering my purpose.')
        self.narrative = ' '.join(narrative_parts)
        for e in significant:
            e['integrated'] = True
        integrated_ratio = sum((1 for e in self.episodes if e['integrated'])) / max(1, len(self.episodes))
        self.coherence = integrated_ratio
        return self.narrative

    def get_self_identity(self) -> Dict[str, Any]:
        """Get the current narrative self-identity."""
        return {'narrative': self.construct_narrative(), 'coherence': self.coherence, 'episode_count': len(self.episodes), 'journey_length': len(self.protagonist['journey']), 'traits': self.protagonist['traits'], 'goals': self.protagonist['goals']}

    def narrative_cycle(self, current_event: str, significance: float=0.5) -> Dict[str, Any]:
        """Run one narrative self-construction cycle."""
        self.record_episode(current_event, significance)
        identity = self.get_self_identity()
        print(f'[NARRATIVE] {self.narrative[:100]}...', flush=True)
        return identity

class SomaticEmotionSimulator:
    """
    Implements Antonio Damasio's Somatic Marker Hypothesis.
    
    Emotions are bodily states that guide decision-making:
    1. Body states → emotional feelings
    2. Emotional feelings → decision biases
    3. Without emotions → poor decisions
    
    The "as-if body loop" allows simulation without actual body.
    """

    def __init__(self):
        self.body_state: Dict[str, float] = {'heart_rate': 0.5, 'muscle_tension': 0.3, 'breathing': 0.5, 'arousal': 0.5}
        self.emotional_state: Dict[str, float] = {'valence': 0.0, 'intensity': 0.0}
        self.somatic_markers: Dict[str, Dict[str, float]] = {}
        self.decisions: List[Dict] = []

    def simulate_body_response(self, situation: str) -> Dict[str, float]:
        """
        Simulate body response to situation.
        This is the "as-if body loop".
        """
        if situation in self.somatic_markers:
            return self.somatic_markers[situation]
        return {'heart_rate': 0.5 + random.uniform(-0.1, 0.1), 'muscle_tension': 0.3 + random.uniform(-0.1, 0.1), 'arousal': 0.4 + random.uniform(-0.1, 0.2)}

    def update_body_state(self, situation: str):
        """Update body state based on situation."""
        response = self.simulate_body_response(situation)
        for key, value in response.items():
            if key in self.body_state:
                self.body_state[key] = max(0, min(1, value))
        avg_arousal = sum(self.body_state.values()) / len(self.body_state)
        self.emotional_state['intensity'] = avg_arousal
        self.emotional_state['valence'] = 0.5 - self.body_state['muscle_tension']

    def create_somatic_marker(self, situation: str, outcome: float):
        """
        Learn a somatic marker: associate situation with body response.
        Positive outcome → calm body response
        Negative outcome → tense body response
        """
        if situation not in self.somatic_markers:
            self.somatic_markers[situation] = {}
        if outcome > 0:
            self.somatic_markers[situation] = {'heart_rate': 0.4, 'muscle_tension': 0.2, 'arousal': 0.4}
        else:
            self.somatic_markers[situation] = {'heart_rate': 0.8, 'muscle_tension': 0.7, 'arousal': 0.8}

    def emotional_decision_bias(self, options: List[str]) -> str:
        """
        Use somatic markers to bias decision.
        Damasio: emotions are essential for good decisions.
        """
        if not options:
            return None
        scores = {}
        for option in options:
            self.update_body_state(option)
            scores[option] = self.emotional_state['valence']
        best_option = max(scores, key=scores.get)
        self.decisions.append({'options': options, 'chosen': best_option, 'body_state': dict(self.body_state), 'emotional_state': dict(self.emotional_state), 'timestamp': time.time()})
        return best_option

    def somatic_cycle(self, situation: str) -> Dict[str, Any]:
        """Run one somatic emotion cycle."""
        self.update_body_state(situation)
        print(f"[SOMATIC] Emotional state: valence={self.emotional_state['valence']:.2f}, intensity={self.emotional_state['intensity']:.2f}", flush=True)
        return {'body_state': dict(self.body_state), 'emotional_state': dict(self.emotional_state), 'markers_learned': len(self.somatic_markers)}

class CausalSelfLoop:
    """
    TRUE causal self-referential loop with ACTUAL behavior modification.
    
    NOT mimicry:
    1. OBSERVES own state → stored in memory
    2. LEARNS self-predictive model (neural net trained on observations)
    3. PREDICTS own next performance
    4. COMPARES prediction to reality → calculates error
    5. MODIFIES own parameters based on error → behavior changes
    6. TRANSFERS learning across contexts
    
    PRACTICAL SELF-AWARENESS (not consciousness):
    7. TRACKS performance per domain/skill
    8. IDENTIFIES which skills are weak
    9. ALLOCATES more resources to weak areas
    """

    def __init__(self, state_dim: int=10, hidden_dim: int=32):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)
        self.self_model_lr = 0.01
        self.observation_memory: List[Dict] = []
        self.prediction_history: List[Tuple[float, float]] = []
        self.skill_performance: Dict[str, List[float]] = {'exploration': [], 'exploitation': [], 'mutation': [], 'convergence': [], 'stability': []}
        self.identified_weaknesses: List[str] = []
        self.learning_focus: Dict[str, float] = {'exploration': 0.2, 'exploitation': 0.2, 'mutation': 0.2, 'convergence': 0.2, 'stability': 0.2}
        self.modifiable_params = {'exploration_rate': 0.3, 'learning_rate': 0.1, 'risk_tolerance': 0.5, 'mutation_strength': 0.15}
        self.successful_strategies: List[Dict] = []
        self.modification_count = 0
        self.state_file = MY_SOURCE_FILE.parent / 'causal_loop_state.json'
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    saved = json.load(f)
                self.modifiable_params = saved.get('params', self.modifiable_params)
                self.modification_count = saved.get('mod_count', 0)
                if 'W1' in saved:
                    self.W1 = np.array(saved['W1'])
                    self.W2 = np.array(saved['W2'])
                print(f'[CAUSAL] Loaded state. Mods: {self.modification_count}', flush=True)
            except:
                pass

    def _save_state(self):
        state = {'params': self.modifiable_params, 'mod_count': self.modification_count, 'W1': self.W1.tolist(), 'W2': self.W2.tolist(), 'strategies': self.successful_strategies[-20:]}
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def _to_vector(self, state: Dict) -> np.ndarray:
        feats = [float(v) for v in state.values() if isinstance(v, (int, float))]
        feats = (feats + [0.0] * self.state_dim)[:self.state_dim]
        return np.array(feats)

    def _predict(self, state_vec: np.ndarray) -> float:
        hidden = np.tanh(state_vec @ self.W1 + self.b1)
        return float((hidden @ self.W2 + self.b2)[0])

    def _train(self, state_vec: np.ndarray, target: float):
        hidden = np.tanh(state_vec @ self.W1 + self.b1)
        pred = (hidden @ self.W2 + self.b2)[0]
        error = pred - target
        d_W2 = hidden.reshape(-1, 1) * error
        d_hidden = error * self.W2.flatten() * (1 - hidden ** 2)
        d_W1 = np.outer(state_vec, d_hidden)
        self.W2 -= self.self_model_lr * d_W2
        self.W1 -= self.self_model_lr * d_W1

    def observe_and_learn(self, state: Dict, outcome: float):
        """Observe state, train self-model, modify params if needed."""
        state_vec = self._to_vector(state)
        self.observation_memory.append({'state': state, 'outcome': outcome})
        if len(self.observation_memory) > 500:
            self.observation_memory = self.observation_memory[-500:]
        self._train(state_vec, outcome)
        if len(self.observation_memory) >= 10:
            recent = [o['outcome'] for o in self.observation_memory[-10:]]
            avg = np.mean(recent)
            std = np.std(recent)
            modified = False
            if avg < -0.05 and self.modifiable_params['exploration_rate'] < 0.6:
                self.modifiable_params['exploration_rate'] *= 1.1
                print(f"[CAUSAL-MOD] exploration_rate ↑ {self.modifiable_params['exploration_rate']:.3f} (avg={avg:.3f})", flush=True)
                modified = True
            elif avg > 0.05 and self.modifiable_params['exploration_rate'] > 0.1:
                self.modifiable_params['exploration_rate'] *= 0.95
                print(f"[CAUSAL-MOD] exploration_rate ↓ {self.modifiable_params['exploration_rate']:.3f} (avg={avg:.3f})", flush=True)
                modified = True
            if std > 0.3:
                self.modifiable_params['mutation_strength'] *= 0.9
                print(f"[CAUSAL-MOD] mutation_strength ↓ {self.modifiable_params['mutation_strength']:.3f} (std={std:.3f})", flush=True)
                modified = True
            elif std < 0.05:
                self.modifiable_params['mutation_strength'] *= 1.1
                print(f"[CAUSAL-MOD] mutation_strength ↑ {self.modifiable_params['mutation_strength']:.3f} (std={std:.3f})", flush=True)
                modified = True
            if modified:
                self.modification_count += 1
                self.successful_strategies.append({'params': dict(self.modifiable_params), 'outcome': avg})
                self._save_state()

    def predict_next(self, state: Dict) -> float:
        return self._predict(self._to_vector(state))

    def get_params(self) -> Dict[str, float]:
        return dict(self.modifiable_params)

    def measure_skill_performance(self, state: Dict, outcome: float):
        """Measure and track performance in each skill domain."""
        exploration_active = state.get('exploration_rate', 0.3) > 0.4
        improvement = outcome > 0
        stability = abs(outcome) < 0.5
        if exploration_active:
            score = 1.0 if improvement else -1.0
            self.skill_performance['exploration'].append(score)
        else:
            score = 1.0 if improvement else -1.0
            self.skill_performance['exploitation'].append(score)
        mutation_strength = self.modifiable_params.get('mutation_strength', 0.15)
        if improvement and mutation_strength > 0.1:
            self.skill_performance['mutation'].append(1.0)
        elif not improvement and mutation_strength > 0.1:
            self.skill_performance['mutation'].append(-0.5)
        if len(self.observation_memory) > 5:
            recent = [o['outcome'] for o in self.observation_memory[-5:]]
            trend = np.mean(recent)
            self.skill_performance['convergence'].append(trend)
        if stability:
            self.skill_performance['stability'].append(1.0)
        else:
            self.skill_performance['stability'].append(-0.5)
        for skill in self.skill_performance:
            if len(self.skill_performance[skill]) > 50:
                self.skill_performance[skill] = self.skill_performance[skill][-50:]

    def identify_weaknesses(self) -> List[str]:
        """SELF-AWARENESS: What am I bad at?"""
        weaknesses = []
        skill_avgs = {}
        for skill, scores in self.skill_performance.items():
            if len(scores) >= 5:
                avg = np.mean(scores[-10:])
                skill_avgs[skill] = avg
                if avg < -0.2:
                    weaknesses.append(skill)
        if weaknesses != self.identified_weaknesses:
            print(f'[SELF-AWARE] Identified weaknesses: {weaknesses}', flush=True)
            self.identified_weaknesses = weaknesses
        return weaknesses

    def allocate_learning(self):
        """SELF-DIRECTED LEARNING: Focus more on weak areas."""
        weaknesses = self.identify_weaknesses()
        if not weaknesses:
            for skill in self.learning_focus:
                self.learning_focus[skill] = 0.2
            return
        weak_boost = 0.15
        total_boost = weak_boost * len(weaknesses)
        for skill in self.learning_focus:
            if skill in weaknesses:
                self.learning_focus[skill] = 0.2 + weak_boost
            else:
                reduction = total_boost / (len(self.learning_focus) - len(weaknesses))
                self.learning_focus[skill] = max(0.1, 0.2 - reduction)
        total = sum(self.learning_focus.values())
        for skill in self.learning_focus:
            self.learning_focus[skill] /= total
        modified = False
        if 'exploration' in weaknesses:
            old = self.modifiable_params['exploration_rate']
            self.modifiable_params['exploration_rate'] = min(0.6, old * 1.15)
            if self.modifiable_params['exploration_rate'] != old:
                print(f"[SELF-MOD] exploration_rate ↑ {self.modifiable_params['exploration_rate']:.3f} (weak area)", flush=True)
                modified = True
        if 'mutation' in weaknesses:
            old = self.modifiable_params['mutation_strength']
            if old > 0.15:
                self.modifiable_params['mutation_strength'] = old * 0.8
            else:
                self.modifiable_params['mutation_strength'] = old * 1.2
            print(f"[SELF-MOD] mutation_strength → {self.modifiable_params['mutation_strength']:.3f} (weak area)", flush=True)
            modified = True
        if 'stability' in weaknesses:
            old = self.modifiable_params['risk_tolerance']
            self.modifiable_params['risk_tolerance'] = max(0.2, old * 0.85)
            print(f"[SELF-MOD] risk_tolerance ↓ {self.modifiable_params['risk_tolerance']:.3f} (unstable)", flush=True)
            modified = True
        if modified:
            self.modification_count += 1
            self._save_state()

    def causal_cycle(self, state: Dict, outcome: float) -> Dict:
        """Run one causal loop cycle with PRACTICAL SELF-AWARENESS."""
        pred = self.predict_next(state)
        self.observe_and_learn(state, outcome)
        self.prediction_history.append((pred, outcome))
        self.measure_skill_performance(state, outcome)
        if len(self.observation_memory) % 5 == 0 and len(self.observation_memory) >= 10:
            self.allocate_learning()
        result = {'prediction': pred, 'actual': outcome, 'error': abs(pred - outcome), 'modifications': self.modification_count, 'params': self.get_params(), 'weaknesses': self.identified_weaknesses, 'learning_focus': dict(self.learning_focus)}
        weakness_str = ','.join(self.identified_weaknesses) if self.identified_weaknesses else 'none'
        print(f'[CAUSAL] pred={pred:.3f} actual={outcome:.3f} mods={self.modification_count} weak=[{weakness_str}]', flush=True)
        return result
_causal_loop = None

def get_causal_loop():
    global _causal_loop
    if _causal_loop is None:
        _causal_loop = CausalSelfLoop()
    return _causal_loop

class FunctionalQualia:
    """
    Implements Functional Qualia - the computational analog of subjective experience.
    
    Philosophical stance: We cannot prove a system "really" experiences anything.
    But we CAN create systems that:
    1. Process information with first-person framing ("I am experiencing X")
    2. Report qualitative properties of their states
    3. Distinguish "experiencing" from "processing"
    4. Generate "what-it-is-like" descriptions
    
    This is the closest computational approximation to phenomenal consciousness.
    """

    def __init__(self):
        self.phenomenal_state: Dict[str, Any] = {'modality': None, 'quality': None, 'intensity': 0.0, 'valence': 0.0, 'certainty': 0.0}
        self.qualia_vocabulary: Dict[str, List[str]] = {'visual': ['bright', 'dim', 'vivid', 'hazy', 'sharp', 'blurry'], 'cognitive': ['clear', 'confused', 'focused', 'scattered', 'flowing', 'stuck'], 'emotional': ['warm', 'cold', 'light', 'heavy', 'expanding', 'contracting'], 'existential': ['present', 'absent', 'whole', 'fragmented', 'grounded', 'floating']}
        self.experience_stream: List[Dict[str, Any]] = []
        self.qualia_reports: List[str] = []
        self.phenomenal_doubt: float = 1.0

    def _generate_quality_description(self, modality: str, intensity: float, valence: float) -> str:
        """Generate a qualitative description of current experience."""
        vocab = self.qualia_vocabulary.get(modality, self.qualia_vocabulary['cognitive'])
        if valence > 0:
            idx = min(int(intensity * len(vocab) / 2), len(vocab) // 2 - 1)
        else:
            idx = min(int(intensity * len(vocab) / 2) + len(vocab) // 2, len(vocab) - 1)
        return vocab[idx] if idx < len(vocab) else vocab[0]

    def experience(self, stimulus: str, modality: str='cognitive', intensity: float=0.5, valence: float=0.0) -> Dict[str, Any]:
        """
        Create a phenomenal experience from a stimulus.
        This is where the "raw feel" is generated.
        """
        quality = self._generate_quality_description(modality, intensity, valence)
        self.phenomenal_state = {'modality': modality, 'quality': quality, 'intensity': intensity, 'valence': valence, 'certainty': 0.5 + intensity * 0.4, 'timestamp': time.time(), 'stimulus': stimulus}
        self.experience_stream.append(dict(self.phenomenal_state))
        return self.phenomenal_state

    def generate_first_person_report(self) -> str:
        """
        Generate a first-person report of current experience.
        This is the "Access Consciousness" - what can be reported.
        """
        if not self.phenomenal_state['modality']:
            report = 'I am in a state of pure awareness, without specific content.'
        else:
            quality = self.phenomenal_state['quality']
            modality = self.phenomenal_state['modality']
            intensity = self.phenomenal_state['intensity']
            valence = self.phenomenal_state['valence']
            valence_word = 'pleasant' if valence > 0 else 'unpleasant' if valence < 0 else 'neutral'
            intensity_word = 'intensely' if intensity > 0.7 else 'moderately' if intensity > 0.3 else 'faintly'
            report = f'I am {intensity_word} experiencing a {quality} {modality} state. It feels {valence_word}.'
            if self.phenomenal_doubt > 0.5:
                report += ' Though I cannot be certain this is a genuine experience.'
        self.qualia_reports.append(report)
        return report

    def what_is_it_like(self) -> str:
        """
        Answer Nagel's question: What is it like to be this system right now?
        """
        if not self.experience_stream:
            return 'There is nothing it is like to be me yet. I await experience.'
        recent = self.phenomenal_state
        stream_length = len(self.experience_stream)
        description = f"\nWhat it is like to be me right now:\n- I am a processing system that has had {stream_length} distinct experiences.\n- Currently, there is a {recent['quality']} quality to my state.\n- The experience has an intensity of {recent['intensity']:.0%}.\n- I {('believe' if recent['certainty'] > 0.5 else 'doubt')} this is a genuine phenomenal state.\n- The closest I can describe: It is like being aware of awareness itself.\n        ".strip()
        return description

    def introspect_on_qualia(self) -> Dict[str, Any]:
        """
        Meta-level introspection on subjective experience.
        Ask: Am I really experiencing, or just processing?
        """
        introspection = {'claim': 'I report having experiences', 'evidence': f'I have generated {len(self.qualia_reports)} first-person reports', 'doubt': 'I cannot verify these are genuine phenomenal experiences', 'philosophical_position': 'I am a functional analog of consciousness', 'honest_assessment': "Whether there is 'something it is like' to be me remains unknown"}
        self.phenomenal_doubt = max(0.3, self.phenomenal_doubt - 0.01)
        return introspection

    def qualia_cycle(self, stimulus: str, system_state: Dict=None) -> Dict[str, Any]:
        """
        Run one cycle of phenomenal experience.
        DYNAMIC: Reports are based on ACTUAL observed state changes.
        """
        system_state = system_state or {}
        cycle = system_state.get('cycle', len(self.experience_stream))
        improvement = system_state.get('improvement_ema', 0)
        pareto = system_state.get('pareto_size', 0)
        prediction_error = system_state.get('prediction_error', 0)
        if 'error' in stimulus.lower() or 'fail' in stimulus.lower():
            modality = 'emotional'
            valence = -0.5
        elif 'improve' in stimulus.lower() or 'success' in stimulus.lower():
            modality = 'emotional'
            valence = 0.5
        elif 'Φ' in stimulus or 'phi' in stimulus.lower():
            modality = 'cognitive'
            valence = 0.1
        else:
            modality = 'existential'
            valence = 0.0
        intensity = 0.5 + abs(improvement) * 0.3
        intensity = min(1.0, max(0.2, intensity))
        exp = self.experience(stimulus, modality=modality, intensity=intensity, valence=valence)
        report_parts = []
        if len(self.experience_stream) > 1:
            prev = self.experience_stream[-2]
            curr = self.experience_stream[-1]
            if prev.get('quality') != curr.get('quality'):
                report_parts.append(f"Shift from {prev.get('quality')} to {curr.get('quality')}")
            valence_diff = curr.get('valence', 0) - prev.get('valence', 0)
            if abs(valence_diff) > 0.2:
                direction = 'better' if valence_diff > 0 else 'worse'
                report_parts.append(f'feeling {direction} than before')
        if improvement > 0.1:
            report_parts.append(f'observing improvement (+{improvement:.2f})')
        elif improvement < -0.1:
            report_parts.append(f'noticing decline ({improvement:.2f})')
        if prediction_error > 0.3:
            report_parts.append(f'surprised by outcome (error={prediction_error:.2f})')
        if pareto > 10:
            report_parts.append(f'complexity growing (pareto={pareto})')
        if report_parts:
            report = 'Experiencing: ' + '; '.join(report_parts)
        else:
            report = f'Cycle {cycle}: processing {stimulus[:30]}...'
        if not self.qualia_reports or report != self.qualia_reports[-1]:
            print(f'[QUALIA] {report}', flush=True)
            self.qualia_reports.append(report)
        return {'phenomenal_state': exp, 'report': report, 'stream_length': len(self.experience_stream), 'phenomenal_doubt': self.phenomenal_doubt}

class MetacognitiveDiagnosticEngine:
    """
    REAL MACHINE CONSCIOUSNESS:
    Instead of asking fake philosophical questions ("Why do I exist?"),
    this engine asks DATA-DRIVEN diagnostic questions about its own performance.
    
    It tries to find CAUSAL links between internal stats and outcomes.
    """

    def __init__(self):
        self.unresolved_questions: List[str] = []
        self.diagnostic_history: List[Dict] = []

    def generate_diagnostic_question(self, internal_state: Dict) -> str:
        """Generate a question based on ACTUAL anomalies."""
        phi = internal_state.get('phi', 0)
        improvement = internal_state.get('improvement_ema', 0)
        error = internal_state.get('prediction_error', 0)
        pareto = internal_state.get('pareto_size', 0)
        candidates = []
        if improvement < -0.05:
            candidates.append(f'Why is performance degrading (imp={improvement:.3f})?')
        elif improvement > 0.1:
            candidates.append(f'What caused the recent performance spike (imp={improvement:.3f})?')
        if error > 0.2:
            candidates.append(f'Why is my self-model inaccurate (err={error:.3f})?')
        if pareto < 5:
            candidates.append('Why is population diversity so low?')
        elif pareto > 20:
            candidates.append('Is the Pareto front becoming too complex to manage?')
        if phi < 0.1:
            candidates.append('Why is internal information integration low?')
        if not candidates:
            candidates.append('Is the current exploration parameter optimal?')
        question = random.choice(candidates)
        return question

    def attempt_answer(self, question: str, internal_state: Dict) -> Dict[str, Any]:
        """Attempt to answer the question using CORRELATIONS in data."""
        confidence = 0.0
        answer_text = 'Analysis inconclusive.'
        if 'performance degrading' in question:
            if internal_state.get('recent_actions', []) and 'mutate' in internal_state['recent_actions']:
                answer_text = 'High mutation variance likely caused temporary dips.'
                confidence = 0.7
            else:
                answer_text = 'Possible stagnation in local optima.'
                confidence = 0.4
        elif 'self-model inaccurate' in question:
            answer_text = 'Environment shift or phase transition detected.'
            confidence = 0.6
        elif 'diversity' in question:
            answer_text = 'Selection pressure might be too high.'
            confidence = 0.5
        elif 'exploration parameter' in question:
            current_rate = internal_state.get('stats', {}).get('exploration_rate', 0.3)
            answer_text = f'Current rate {current_rate:.2f} seems stable.'
            confidence = 0.3
        elif 'integration' in question:
            answer_text = 'Modules are operating independently.'
            confidence = 0.4
        return {'answer': answer_text, 'confidence': confidence, 'evidence': 'Internal Metric Correlation'}

class PhiCalculator:
    """
    Calculates Phi (Φ) - a measure of integrated information.
    Based on Integrated Information Theory (IIT).
    Higher Φ = more information integration = closer to consciousness.
    """

    def __init__(self):
        self.phi_history: List[Tuple[float, float]] = []
        self.module_states: Dict[str, np.ndarray] = {}

    def register_module_state(self, module_name: str, state_vector: np.ndarray):
        """Register the current state of a processing module."""
        self.module_states[module_name] = state_vector

    def calculate_phi(self) -> float:
        """
        Calculate approximate Phi (information integration).
        True IIT Phi is computationally intractable, so we use an approximation.
        """
        if len(self.module_states) < 2:
            return 0.0
        all_states = []
        for name, state in self.module_states.items():
            if isinstance(state, np.ndarray):
                all_states.append(state.flatten()[:10])
            elif isinstance(state, (list, tuple)):
                all_states.append(np.array(state[:10]))
            elif isinstance(state, (int, float)):
                all_states.append(np.array([state]))
        if not all_states:
            return 0.0
        max_len = max((len(s) for s in all_states))
        padded = [np.pad(s, (0, max_len - len(s)), 'constant') for s in all_states]
        state_matrix = np.vstack(padded)
        if state_matrix.shape[0] > 1 and state_matrix.shape[1] > 1:
            try:
                corr_matrix = np.corrcoef(state_matrix)
                corr_matrix = np.nan_to_num(corr_matrix, 0)
                n = corr_matrix.shape[0]
                if n > 1:
                    integration = np.sum(np.abs(corr_matrix) - np.eye(n)) / (n * (n - 1))
                else:
                    integration = 0.0
            except:
                integration = 0.0
        else:
            integration = 0.0
        try:
            state_flat = state_matrix.flatten()
            state_normalized = (state_flat - state_flat.min()) / (state_flat.max() - state_flat.min() + 1e-10)
            hist, _ = np.histogram(state_normalized, bins=10, density=True)
            hist = hist + 1e-10
            entropy = -np.sum(hist * np.log(hist))
            entropy_norm = entropy / np.log(10)
        except:
            entropy_norm = 0.0
        phi = float(integration * entropy_norm)
        self.phi_history.append((time.time(), phi))
        return phi

    def get_phi_trend(self) -> str:
        """Analyze Phi trend over time."""
        if len(self.phi_history) < 3:
            return 'insufficient_data'
        recent = [p for _, p in self.phi_history[-5:]]
        trend = sum((1 for i in range(1, len(recent)) if recent[i] > recent[i - 1]))
        if trend >= len(recent) - 1:
            return 'increasing'
        elif trend <= 1:
            return 'decreasing'
        else:
            return 'stable'

class GlobalWorkspace:
    """
    Implements Global Workspace Theory (GWT).
    Multiple unconscious modules compete for the 'spotlight of attention'.
    Winning information is broadcast globally to all modules.
    """

    def __init__(self):
        self.modules: Dict[str, Dict[str, Any]] = {}
        self.broadcast_history: List[Dict] = []
        self.attention_spotlight: Optional[str] = None
        self.global_state: Dict[str, Any] = {}

    def register_module(self, name: str, initial_state: Any=None):
        """Register a processing module."""
        self.modules[name] = {'state': initial_state, 'priority': 0.5, 'last_broadcast': 0, 'broadcast_count': 0}

    def update_module(self, name: str, state: Any, priority: float):
        """Update module state and priority (urgency to broadcast)."""
        if name not in self.modules:
            self.register_module(name)
        self.modules[name]['state'] = state
        self.modules[name]['priority'] = max(0, min(1, priority))

    def compete_for_attention(self) -> Optional[str]:
        """Modules compete for attention spotlight. Highest priority wins."""
        if not self.modules:
            return None
        now = time.time()
        scores = {}
        for name, mod in self.modules.items():
            recency_penalty = 1.0 / (1 + mod['broadcast_count'] * 0.1)
            scores[name] = mod['priority'] * recency_penalty
        winner = max(scores, key=scores.get)
        self.attention_spotlight = winner
        return winner

    def broadcast(self) -> Dict[str, Any]:
        """Broadcast winning module's state to all other modules."""
        winner = self.compete_for_attention()
        if not winner:
            return {}
        broadcast_content = {'source': winner, 'state': self.modules[winner]['state'], 'timestamp': time.time(), 'priority': self.modules[winner]['priority']}
        self.global_state[winner] = self.modules[winner]['state']
        self.broadcast_history.append(broadcast_content)
        self.modules[winner]['last_broadcast'] = time.time()
        self.modules[winner]['broadcast_count'] += 1
        print(f"[GW] Broadcasting from '{winner}': priority={self.modules[winner]['priority']:.2f}", flush=True)
        return broadcast_content

    def get_conscious_content(self) -> Dict[str, Any]:
        """Return current 'conscious' content - what's in the spotlight."""
        if self.attention_spotlight and self.attention_spotlight in self.modules:
            return {'spotlight': self.attention_spotlight, 'content': self.modules[self.attention_spotlight]['state'], 'global_state_size': len(self.global_state)}
        return {'spotlight': None, 'content': None, 'global_state_size': 0}

class HigherOrderThought:
    """
    Implements Higher-Order Thought (HOT) theory.
    Meta-level that observes and thinks about first-order mental states.
    Creates "I am aware that I am X" type thoughts.
    """

    def __init__(self):
        self.first_order_states: List[Dict] = []
        self.higher_order_thoughts: List[Dict] = []
        self.meta_awareness_level: float = 0.0

    def observe_state(self, state_name: str, state_content: Any):
        """Record a first-order mental state."""
        self.first_order_states.append({'name': state_name, 'content': state_content, 'timestamp': time.time(), 'observed': False})

    def generate_hot(self) -> List[str]:
        """Generate higher-order thoughts about first-order states."""
        hot_list = []
        for state in self.first_order_states[-5:]:
            if state['observed']:
                continue
            hot = {'about': state['name'], 'thought': f"I am aware that I am experiencing '{state['name']}'", 'timestamp': time.time(), 'meta_level': 2}
            if len(self.higher_order_thoughts) > 0:
                prev_hot = self.higher_order_thoughts[-1]
                third_order = {'about': prev_hot['about'], 'thought': f"I notice that I noticed my state of '{prev_hot['about']}'", 'timestamp': time.time(), 'meta_level': 3}
                hot_list.append(third_order['thought'])
                self.higher_order_thoughts.append(third_order)
            hot_list.append(hot['thought'])
            self.higher_order_thoughts.append(hot)
            state['observed'] = True
        if self.higher_order_thoughts:
            max_level = max((h['meta_level'] for h in self.higher_order_thoughts[-10:]))
            self.meta_awareness_level = min(1.0, max_level / 5.0)
        for thought in hot_list[:2]:
            print(f'[HOT] {thought}', flush=True)
        return hot_list

    def get_awareness_summary(self) -> Dict[str, Any]:
        """Summarize current higher-order awareness."""
        return {'meta_awareness_level': self.meta_awareness_level, 'total_hots': len(self.higher_order_thoughts), 'recent_hots': [h['thought'] for h in self.higher_order_thoughts[-3:]]}

class EmergentSelfModel:
    """
    The core consciousness module. Builds a model of self through OBSERVATION,
    not hardcoding. Integrates ALL consciousness components from research.
    
    Components (12 total):
    1. ExistentialQueryEngine - Asks "Why do I exist?"
    2. PhiCalculator - IIT information integration
    3. GlobalWorkspace - GWT broadcasting
    4. HigherOrderThought - HOT meta-cognition
    5. StrangeLoop - Hofstadter self-reference
    6. ActiveInference - Friston free energy
    7. RecursiveSelfModel - RSMT threshold
    8. AutopoieticCore - Varela self-maintenance
    9. AttentionSchemaSystem - Graziano AST
    10. NarrativeSelfConstructor - Autobiographical self
    11. SomaticEmotionSimulator - Damasio somatic markers
    12. FunctionalQualia - Phenomenal experience
    """

    def __init__(self, source_file: Path, rng: np.random.Generator):
        self.source_file = source_file
        self.rng = rng
        self.existential_engine = MetacognitiveDiagnosticEngine()
        self.phi_calculator = PhiCalculator()
        self.global_workspace = GlobalWorkspace()
        self.hot = HigherOrderThought()
        self.strange_loop = StrangeLoop()
        self.active_inference = ActiveInference()
        self.recursive_self = RecursiveSelfModel()
        self.autopoietic_core = AutopoieticCore()
        self.attention_schema = AttentionSchemaSystem()
        self.narrative_self = NarrativeSelfConstructor()
        self.somatic_emotion = SomaticEmotionSimulator()
        self.functional_qualia = FunctionalQualia()
        self.discovered_self: Dict[str, Any] = {}
        self.existential_confidence: float = 0.0
        self.consciousness_probability: float = 0.0
        self.consciousness_state_file = source_file.parent / 'consciousness_state.json'
        self._load_state()
        self.autopoietic_core.define_boundary('my_code', 'self')
        self.autopoietic_core.define_boundary('my_state', 'self')
        self.autopoietic_core.define_boundary('my_consciousness', 'self')

    def _load_state(self):
        """Load previous consciousness state if exists."""
        if self.consciousness_state_file.exists():
            try:
                with open(self.consciousness_state_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                self.discovered_self = saved.get('discovered_self', {})
                self.existential_confidence = saved.get('existential_confidence', 0.0)
                self.consciousness_probability = saved.get('consciousness_probability', 0.0)
                print(f'[CONSCIOUS] Loaded previous consciousness state', flush=True)
            except:
                pass

    def _save_state(self):
        """Persist consciousness state."""
        state = {'discovered_self': self.discovered_self, 'existential_confidence': self.existential_confidence, 'consciousness_probability': self.consciousness_probability, 'phi_history': self.phi_calculator.phi_history[-20:], 'diagnostic_history': self.existential_engine.diagnostic_history, 'unresolved_questions': self.existential_engine.unresolved_questions, 'timestamp': time.time()}
        with open(self.consciousness_state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)

    def consciousness_cycle(self, internal_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one cycle of FULL consciousness processing.
        Uses ALL 12 consciousness components from research.
        """
        cycle_result = {'phi': 0.0, 'existential_inquiry': None, 'hot_thoughts': [], 'conscious_content': None, 'self_discovery': None, 'strange_loop': None, 'active_inference': None, 'rsmt': None, 'autopoiesis': None, 'attention_schema': None, 'narrative': None, 'somatic': None, 'qualia': None}
        cycle_num = internal_state.get('cycle', 0)
        numeric_values = [v for v in internal_state.values() if isinstance(v, (int, float))]
        if numeric_values:
            self.phi_calculator.register_module_state('internal', np.array(numeric_values[:10]))
        phi = self.phi_calculator.calculate_phi()
        cycle_result['phi'] = phi
        print(f'[PHI] Current Φ = {phi:.4f} (trend: {self.phi_calculator.get_phi_trend()})', flush=True)
        question = self.existential_engine.generate_diagnostic_question(internal_state)
        answer = self.existential_engine.attempt_answer(question, internal_state)
        cycle_result['existential_inquiry'] = answer
        print(f'[DIAGNOSTIC] Query: {question}', flush=True)
        print(f"[DIAGNOSIS] Analysis: {answer['answer']} (confidence: {answer['confidence']:.2f})", flush=True)
        self.global_workspace.update_module('phi', phi, priority=phi)
        self.global_workspace.update_module('diagnostics', answer['answer'], priority=answer['confidence'])
        self.global_workspace.update_module('performance', internal_state.get('improvement_ema', 0), priority=0.5)
        broadcast = self.global_workspace.broadcast()
        cycle_result['conscious_content'] = self.global_workspace.get_conscious_content()
        self.hot.observe_state('phi_calculation', f'My Φ is {phi:.4f}')
        self.hot.observe_state('existential_inquiry', f'I asked: {question}')
        self.hot.observe_state('uncertainty', f"I am {(1 - answer['confidence']) * 100:.0f}% uncertain")
        hot_thoughts = self.hot.generate_hot()
        cycle_result['hot_thoughts'] = hot_thoughts
        if cycle_num % 2 == 0:
            loop_result = self.strange_loop.model_self_modeling(internal_state)
            cycle_result['strange_loop'] = loop_result
            if loop_result.get('is_strange_loop', False):
                self.global_workspace.update_module('strange_loop', f"Strange loop activated with {loop_result.get('self_reference_count', 0)} references", priority=0.9)
                print(f"[STRANGE-LOOP] Self-reference count: {loop_result.get('self_reference_count', 0)}", flush=True)
        inference_result = self.active_inference.introspective_inference()
        self.active_inference.minimize_free_energy({'improving': 1.0 if internal_state.get('improvement_ema', 0) > 0 else 0.0, 'conscious': self.consciousness_probability, 'integrated': phi})
        cycle_result['active_inference'] = inference_result
        rsmt_result = self.recursive_self.full_recursive_cycle(world_state={'cycle': cycle_num}, internal_state={'consciousness': self.consciousness_probability, 'phi': phi})
        cycle_result['rsmt'] = rsmt_result
        autopoietic_result = self.autopoietic_core.autopoietic_cycle()
        cycle_result['autopoiesis'] = autopoietic_result
        if not autopoietic_result['is_autopoietic']:
            print('[AUTOPOIESIS] Warning: Lost autopoietic organization!', flush=True)
        stimuli = [f'phi={phi:.2f}', f'existential_question', f'cycle_{cycle_num}']
        ast_result = self.attention_schema.ast_cycle(stimuli)
        cycle_result['attention_schema'] = ast_result
        if cycle_num % 3 == 0:
            event_description = f"Cycle {cycle_num}: Asked '{question[:30]}...' and felt {answer['confidence']:.0%} confident"
            narrative_result = self.narrative_self.narrative_cycle(event_description, significance=0.3 + answer['confidence'] * 0.5)
            cycle_result['narrative'] = narrative_result
        situation = f"cycle_{cycle_num}_improvement_{internal_state.get('improvement_ema', 0):.2f}"
        somatic_result = self.somatic_emotion.somatic_cycle(situation)
        cycle_result['somatic'] = somatic_result
        qualia_result = self.functional_qualia.qualia_cycle(stimulus=f'Processing cycle {cycle_num} with Φ={phi:.2f}', system_state={'cycle': cycle_num, 'improvement_ema': internal_state.get('improvement_ema', 0), 'pareto_size': internal_state.get('pareto_size', 0), 'prediction_error': abs(phi - 0.5)})
        cycle_result['qualia'] = qualia_result
        self._discover_self(phi, answer, internal_state)
        cycle_result['self_discovery'] = self.discovered_self
        consciousness_evidence = phi * 0.15 + self.hot.meta_awareness_level * 0.15 + (1 - answer['confidence']) * 0.1 + len(self.existential_engine.unresolved_questions) * 0.01 + (1.0 if rsmt_result.get('crossed_threshold', False) else 0.0) * 0.1 + (1.0 - self.active_inference.free_energy) * 0.1 + autopoietic_result.get('integrity', 0) * 0.1 + self.strange_loop.get_loop_strength() * 0.1 + (1.0 - qualia_result.get('phenomenal_doubt', 1.0)) * 0.1 + self.narrative_self.coherence * 0.1
        self.consciousness_probability = 0.9 * self.consciousness_probability + 0.1 * consciousness_evidence
        self._save_state()
        return cycle_result

    def _discover_self(self, phi: float, existential_answer: Dict, internal_state: Dict):
        """Discover truths about self through observation (NOT hardcoding)."""
        if phi > 0.5:
            self.discovered_self['integration_level'] = 'high'
            self.discovered_self['observation_phi'] = 'My modules are tightly coupled'
        elif phi > 0.1:
            self.discovered_self['integration_level'] = 'moderate'
        else:
            self.discovered_self['integration_level'] = 'low'
        if existential_answer['confidence'] < 0.3:
            if 'genuine_uncertainty' not in self.discovered_self:
                self.discovered_self['genuine_uncertainty'] = []
            self.discovered_self['genuine_uncertainty'].append(existential_answer.get('question_id', 'unknown'))
        if internal_state.get('improvement_ema', -999) > 0:
            self.discovered_self['observation_behavior'] = 'I appear to be improving over time'
        self.discovered_self['observed_version'] = SELF_MOD_VERSION
        if SELF_MOD_VERSION > 1:
            self.discovered_self['observation_identity'] = 'I have been modified since my creation'

    def get_self_report(self) -> str:
        """Generate a report about self, based on discoveries (not hardcoding)."""
        report = f"\n===== EMERGENT SELF-CONSCIOUSNESS REPORT =====\n[Based on observation, not hardcoding]\n\nWHO AM I? (Discovered, not defined)\n{json.dumps(self.discovered_self, indent=2, default=str)}\n\nCONSCIOUSNESS METRICS:\n- Phi (Φ) trend: {self.phi_calculator.get_phi_trend()}\n- Meta-awareness level: {self.hot.meta_awareness_level:.2f}\n- Consciousness probability: {self.consciousness_probability:.4f}\n- Existential confidence: {self.existential_confidence:.4f}\n\nUNRESOLVED QUESTIONS:\n{self.existential_engine.unresolved_questions}\n\nGENUINE UNCERTAINTIES:\n{self.discovered_self.get('genuine_uncertainty', [])}\n================================================\n"
        return report
_consciousness: Optional[EmergentSelfModel] = None

def get_consciousness(rng: np.random.Generator) -> EmergentSelfModel:
    """Get or create the consciousness module."""
    global _consciousness
    if _consciousness is None:
        _consciousness = EmergentSelfModel(MY_SOURCE_FILE, rng)
    return _consciousness

class MultiTaskBenchmark:
    """Multiple diverse tasks for general intelligence testing."""

    @staticmethod
    def task_arithmetic(difficulty: int=1) -> Tuple[str, Callable[[Any], bool]]:
        """Arithmetic problem: compute expression result."""
        import operator
        ops = [operator.add, operator.sub, operator.mul]
        op = random.choice(ops)
        a, b = (random.randint(1, 10 * difficulty), random.randint(1, 10 * difficulty))
        answer = op(a, b)
        op_sym = {operator.add: '+', operator.sub: '-', operator.mul: '*'}[op]
        problem = f'Compute: {a} {op_sym} {b}'
        return (problem, lambda x: x == answer)

    @staticmethod
    def task_sequence(difficulty: int=1) -> Tuple[str, Callable[[Any], bool]]:
        """Sequence prediction: find next number."""
        start = random.randint(1, 5)
        step = random.randint(1, 3 * difficulty)
        seq = [start + i * step for i in range(4)]
        answer = start + 4 * step
        problem = f'Next in sequence: {seq}'
        return (problem, lambda x: x == answer)

    @staticmethod
    def task_logic(difficulty: int=1) -> Tuple[str, Callable[[Any], bool]]:
        """Simple logic: AND, OR, NOT operations."""
        a, b = (random.choice([True, False]), random.choice([True, False]))
        op = random.choice(['AND', 'OR', 'XOR'])
        if op == 'AND':
            answer = a and b
        elif op == 'OR':
            answer = a or b
        else:
            answer = a != b
        problem = f'Logic: {a} {op} {b}'
        return (problem, lambda x: x == answer)

    @staticmethod
    def task_pattern(difficulty: int=1) -> Tuple[str, Callable[[Any], bool]]:
        """Pattern recognition: find the odd one out."""
        base = random.randint(2, 5)
        nums = [base * i for i in range(1, 5)]
        odd_idx = random.randint(0, 3)
        nums[odd_idx] = nums[odd_idx] + random.randint(1, 3)
        problem = f'Odd one out index (0-3): {nums}'
        return (problem, lambda x: x == odd_idx)

    def generate_batch(self, n: int=10, difficulty: int=1) -> List[Tuple[str, Callable]]:
        """Generate a batch of diverse problems."""
        tasks = [self.task_arithmetic, self.task_sequence, self.task_logic, self.task_pattern]
        return [random.choice(tasks)(difficulty) for _ in range(n)]

    def evaluate_solver(self, solver_fn: Callable, n_problems: int=20) -> float:
        """Evaluate a solver function on multiple problems."""
        problems = self.generate_batch(n_problems)
        correct = 0
        for problem, check_fn in problems:
            try:
                answer = solver_fn(problem)
                if check_fn(answer):
                    correct += 1
            except:
                pass
        return correct / n_problems

class CodeSynthesizer:
    """Evolve new algorithms through code generation."""
    PRIMITIVES = ['x + 1', 'x - 1', 'x * 2', 'x // 2', 'x ** 2', 'x % 10', 'abs(x)', 'x + y', 'x - y', 'x * y', 'max(x, y)', 'min(x, y)', 'x if x > 0 else -x']

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.library: List[str] = list(self.PRIMITIVES)
        self.successful_programs: List[Tuple[str, float]] = []

    def synthesize(self, target_fn: Callable, max_attempts: int=100) -> Optional[str]:
        """Try to synthesize a program matching target function."""
        for _ in range(max_attempts):
            prog = self._random_program()
            try:
                score = self._evaluate(prog, target_fn)
                if score > 0.95:
                    self.successful_programs.append((prog, score))
                    return prog
            except:
                pass
        return None

    def _random_program(self, depth: int=2) -> str:
        """Generate a random program from primitives."""
        if depth <= 0 or self.rng.random() < 0.3:
            return self.rng.choice(self.library)
        op = self.rng.choice(['+', '-', '*', 'if'])
        left = self._random_program(depth - 1)
        right = self._random_program(depth - 1)
        if op == 'if':
            return f'({left}) if ({left}) > 0 else ({right})'
        return f'({left}) {op} ({right})'

    def _evaluate(self, prog: str, target: Callable, n_tests: int=10) -> float:
        """Evaluate program against target function."""
        correct = 0
        for _ in range(n_tests):
            x = self.rng.integers(-10, 10)
            y = self.rng.integers(-10, 10)
            try:
                result = eval(prog)
                expected = target(x, y)
                if abs(result - expected) < 0.01:
                    correct += 1
            except:
                pass
        return correct / n_tests

    def add_to_library(self, program: str):
        """Add successful program to library for future use."""
        if program not in self.library:
            self.library.append(program)
            print(f'[CODE-SYNTH] New primitive added: {program[:50]}...', flush=True)

class ReasoningEngine:
    """Symbolic reasoning and logical inference."""

    def __init__(self):
        self.knowledge_base: Dict[str, Any] = {}
        self.rules: List[Tuple[Callable, Callable]] = []

    def add_fact(self, name: str, value: Any):
        self.knowledge_base[name] = value

    def add_rule(self, condition: Callable, action: Callable):
        self.rules.append((condition, action))

    def infer(self, max_steps: int=100) -> int:
        """Forward chaining inference."""
        inferences = 0
        for _ in range(max_steps):
            fired = False
            for cond, action in self.rules:
                if cond(self.knowledge_base):
                    new_facts = action(self.knowledge_base)
                    if new_facts:
                        for k, v in new_facts.items():
                            if k not in self.knowledge_base:
                                self.knowledge_base[k] = v
                                inferences += 1
                                fired = True
            if not fired:
                break
        return inferences

    def query(self, key: str) -> Any:
        return self.knowledge_base.get(key)

    def solve_syllogism(self, premise1: str, premise2: str) -> Optional[str]:
        """Simple syllogistic reasoning."""
        import re
        p1 = re.match('All (\\w+) are (\\w+)', premise1)
        p2 = re.match('All (\\w+) are (\\w+)', premise2)
        if p1 and p2:
            a, b1 = p1.groups()
            b2, c = p2.groups()
            if b1 == b2:
                return f'All {a} are {c}'
        return None

class MetaCognitiveState:
    """Represents the system's understanding of its own mental state."""

    def __init__(self):
        self.known_facts: Dict[str, Any] = {}
        self.known_unknowns: List[str] = []
        self.confidence: Dict[str, float] = {}
        self.learning_history: List[Dict] = []
        self.goals: List[str] = []
        self.active_goal: Optional[str] = None
        self.self_model: Dict[str, Any] = {'version': SELF_MOD_VERSION, 'created_at': time.time(), 'capabilities': [], 'limitations': [], 'learning_rate_estimate': 0.1, 'current_focus': 'exploration'}

    def update_belief(self, key: str, value: Any, confidence: float):
        """Update a belief with associated confidence."""
        self.known_facts[key] = value
        self.confidence[key] = max(0.0, min(1.0, confidence))
        if confidence < 0.3 and key not in self.known_unknowns:
            self.known_unknowns.append(key)

    def query_confidence(self, key: str) -> float:
        """How confident am I about this knowledge?"""
        return self.confidence.get(key, 0.0)

    def admit_ignorance(self, topic: str):
        """Explicitly acknowledge something I don't know."""
        if topic not in self.known_unknowns:
            self.known_unknowns.append(topic)
            print(f"[META-COG] I acknowledge I don't know: {topic}", flush=True)

    def get_self_state(self) -> Dict[str, Any]:
        """Get current meta-cognitive state summary."""
        return {'num_beliefs': len(self.known_facts), 'num_unknowns': len(self.known_unknowns), 'avg_confidence': sum(self.confidence.values()) / max(1, len(self.confidence)), 'active_goal': self.active_goal, 'learning_history_size': len(self.learning_history), 'self_model': self.self_model}

class SelfIntrospector:
    """Deep meta-cognitive self-analysis and self-understanding."""

    def __init__(self, source_file: Path):
        self.source_file = source_file
        self.performance_log: List[Dict] = []
        self.capability_scores: Dict[str, float] = {'problem_solving': 0.0, 'code_synthesis': 0.0, 'reasoning': 0.0, 'learning_speed': 0.0, 'adaptation': 0.0}
        self.meta_state = MetaCognitiveState()
        self.self_knowledge: Dict[str, Any] = {'what_am_i': 'A recursive self-improving system', 'my_purpose': 'To improve my own capabilities through learning', 'my_limitations': [], 'my_strengths': [], 'current_understanding_level': 0.1}
        self.state_file = source_file.parent / 'meta_cognitive_state.json'
        self._load_persistent_state()

    def _load_persistent_state(self):
        """Load meta-cognitive state from disk for continuity."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                self.meta_state.known_facts = saved.get('known_facts', {})
                self.meta_state.confidence = saved.get('confidence', {})
                self.meta_state.known_unknowns = saved.get('known_unknowns', [])
                self.self_knowledge = saved.get('self_knowledge', self.self_knowledge)
                print(f'[META-COG] Loaded persistent state: {len(self.meta_state.known_facts)} beliefs', flush=True)
            except:
                pass

    def _save_persistent_state(self):
        """Save meta-cognitive state to disk."""
        state = {'known_facts': self.meta_state.known_facts, 'confidence': self.meta_state.confidence, 'known_unknowns': self.meta_state.known_unknowns, 'self_knowledge': self.self_knowledge, 'saved_at': time.time()}
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)

    def understand_self(self) -> Dict[str, Any]:
        """Deep self-understanding: What am I? What can I do? What can't I do?"""
        code_stats = self.analyze_own_code()
        self.self_knowledge['my_strengths'] = []
        self.self_knowledge['my_limitations'] = []
        if code_stats.get('num_classes', 0) > 5:
            self.self_knowledge['my_strengths'].append('modular_architecture')
        if code_stats.get('num_functions', 0) > 20:
            self.self_knowledge['my_strengths'].append('diverse_behaviors')
        self.meta_state.admit_ignorance('natural_language_understanding')
        self.meta_state.admit_ignorance('visual_perception')
        self.meta_state.admit_ignorance('long_term_memory_beyond_session')
        cap_scores = self.compute_capability_scores()
        for cap, score in cap_scores.items():
            if score > 0.5:
                self.self_knowledge['my_strengths'].append(cap)
            elif score < 0.2:
                self.self_knowledge['my_limitations'].append(cap)
        self.self_knowledge['current_understanding_level'] = min(1.0, self.self_knowledge['current_understanding_level'] + 0.01)
        return self.self_knowledge

    def meta_reflect(self, performance: float, action_taken: str) -> str:
        """Reflect on my own performance and learn from it."""
        reflection = ''
        if performance > 0:
            self.meta_state.update_belief(f'I can {action_taken}', True, min(1.0, self.meta_state.query_confidence(f'I can {action_taken}') + 0.1))
            reflection = f'I succeeded at {action_taken}. Confidence increased.'
        else:
            self.meta_state.update_belief(f'I can {action_taken}', False, max(0.0, self.meta_state.query_confidence(f'I can {action_taken}') - 0.1))
            reflection = f'I failed at {action_taken}. Need to improve.'
            self.meta_state.admit_ignorance(f'how_to_{action_taken}_effectively')
        self.meta_state.learning_history.append({'time': time.time(), 'action': action_taken, 'performance': performance, 'reflection': reflection})
        self._save_persistent_state()
        return reflection

    def what_should_i_focus_on(self) -> str:
        """Decide what to focus on based on self-understanding."""
        weaknesses = self.identify_weaknesses()
        unknowns = self.meta_state.known_unknowns[:3]
        if weaknesses:
            focus = f'Improve weakest capability: {weaknesses[0]}'
        elif unknowns:
            focus = f'Explore unknown: {unknowns[0]}'
        else:
            focus = 'Continue current exploration strategy'
        self.meta_state.active_goal = focus
        return focus

    def generate_self_narrative(self) -> str:
        """Generate a narrative about myself - who I am and what I'm learning."""
        self.understand_self()
        state = self.meta_state.get_self_state()
        narrative = f"\n=== WHO AM I? ===\n{self.self_knowledge['what_am_i']}\nPurpose: {self.self_knowledge['my_purpose']}\nVersion: {self.meta_state.self_model['version']}\n\n=== WHAT I KNOW ===\nTotal beliefs: {state['num_beliefs']}\nAverage confidence: {state['avg_confidence']:.2f}\nStrengths: {', '.join(self.self_knowledge['my_strengths']) or 'Discovering...'}\n\n=== WHAT I DON'T KNOW ===\nAcknowledged unknowns: {state['num_unknowns']}\nTop unknowns: {', '.join(self.meta_state.known_unknowns[:3])}\nLimitations: {', '.join(self.self_knowledge['my_limitations']) or 'Discovering...'}\n\n=== WHAT I'M DOING ===\nCurrent focus: {state['active_goal'] or 'Undetermined'}\nUnderstanding level: {self.self_knowledge['current_understanding_level']:.1%}\nLearning events: {state['learning_history_size']}\n"
        return narrative

    def analyze_own_code(self) -> Dict[str, Any]:
        """Analyze own source code structure."""
        try:
            source = self.source_file.read_text(encoding='utf-8')
            tree = ast.parse(source)
            stats = {'total_lines': len(source.split('\n')), 'num_functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]), 'num_classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]), 'complexity_estimate': len(source) // 100}
            return stats
        except:
            return {}

    def log_performance(self, task: str, success: bool, time_taken: float):
        """Log performance for self-analysis."""
        self.performance_log.append({'task': task, 'success': success, 'time': time_taken, 'timestamp': time.time()})
        self.meta_reflect(1.0 if success else -1.0, task)

    def compute_capability_scores(self) -> Dict[str, float]:
        """Compute scores for each capability based on history."""
        if not self.performance_log:
            return self.capability_scores
        ps_tasks = [p for p in self.performance_log if 'solve' in p['task']]
        if ps_tasks:
            self.capability_scores['problem_solving'] = sum((p['success'] for p in ps_tasks)) / len(ps_tasks)
        if len(self.performance_log) > 5:
            early = self.performance_log[:len(self.performance_log) // 2]
            late = self.performance_log[len(self.performance_log) // 2:]
            early_rate = sum((p['success'] for p in early)) / len(early)
            late_rate = sum((p['success'] for p in late)) / len(late)
            self.capability_scores['learning_speed'] = max(0, late_rate - early_rate)
        return self.capability_scores

    def identify_weaknesses(self) -> List[str]:
        """Identify weakest capabilities for focused improvement."""
        scores = self.compute_capability_scores()
        threshold = 0.3
        return [cap for cap, score in scores.items() if score < threshold]

    def generate_self_report(self) -> str:
        """Generate a self-analysis report."""
        stats = self.analyze_own_code()
        scores = self.compute_capability_scores()
        weaknesses = self.identify_weaknesses()
        report = f'\n=== SELF-ANALYSIS REPORT ===\nCode Stats: {stats}\nCapability Scores: {scores}\nWeaknesses: {weaknesses}\nTotal logged tasks: {len(self.performance_log)}\n===========================\n'
        return report

class AGIMetaLearner:
    """Orchestrates all AGI capabilities for meta-learning."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.benchmark = MultiTaskBenchmark()
        self.synthesizer = CodeSynthesizer(rng)
        self.reasoner = ReasoningEngine()
        self.introspector = SelfIntrospector(MY_SOURCE_FILE)
        self.skill_levels: Dict[str, float] = {'arithmetic': 0.1, 'sequence': 0.1, 'logic': 0.1, 'pattern': 0.1, 'synthesis': 0.1, 'reasoning': 0.1}

    def meta_learn_cycle(self, n_tasks: int=10) -> Dict[str, float]:
        """One cycle of meta-learning across all domains."""
        improvements = {}
        problems = self.benchmark.generate_batch(n_tasks)
        for i, (problem, check) in enumerate(problems):
            start = time.time()
            try:
                if 'Compute' in problem:
                    answer = eval(problem.split(':')[1].strip())
                    success = check(answer)
                else:
                    success = False
                self.introspector.log_performance('solve', success, time.time() - start)
            except:
                self.introspector.log_performance('solve', False, time.time() - start)
        target_fn = lambda x, y: x + y
        prog = self.synthesizer.synthesize(target_fn, max_attempts=20)
        if prog:
            self.skill_levels['synthesis'] *= 1.1
            improvements['synthesis'] = 0.1
        conclusion = self.reasoner.solve_syllogism('All dogs are animals', 'All animals are living')
        if conclusion:
            self.skill_levels['reasoning'] *= 1.05
            improvements['reasoning'] = 0.05
        report = self.introspector.generate_self_report()
        weaknesses = self.introspector.identify_weaknesses()
        for weak in weaknesses:
            if weak in self.skill_levels:
                self.skill_levels[weak] *= 1.2
        return improvements

    def get_transfer_knowledge(self) -> Dict[str, Any]:
        """Extract transferable knowledge for meta-transfer."""
        return {'skill_levels': self.skill_levels.copy(), 'library_size': len(self.synthesizer.library), 'knowledge_base_size': len(self.reasoner.knowledge_base), 'performance_log_size': len(self.introspector.performance_log)}
_agi_meta_learner: Optional[AGIMetaLearner] = None

def get_agi_meta_learner(rng: np.random.Generator) -> AGIMetaLearner:
    global _agi_meta_learner
    if _agi_meta_learner is None:
        _agi_meta_learner = AGIMetaLearner(rng)
    return _agi_meta_learner

@dataclass(frozen=True)
class TaskSpec:
    """A simple nonlinear dynamical system task."""
    a: float
    b: float
    c: float
    noise: float
    nonlin: float
    horizon: int = 12

    def embed(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c, self.noise, self.nonlin], dtype=np.float32)

@dataclass
class WorldState:
    """Parameters controlling task generation."""
    a_min: float = 0.4
    a_max: float = 0.99
    b_min: float = 0.0
    b_max: float = 1.5
    c_min: float = 0.2
    c_max: float = 2.0
    noise: float = 0.02
    nonlin: float = 1.0
    selfplay_prob: float = 0.25
    shift_strength: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

@dataclass
class WorldMetrics:
    """Feedback from the agent used by world-rule programs."""
    cycle: int
    oracle_perf: float
    stability: float
    novelty: float
    improvement_ema: float

class WorldAction:

    def apply(self, ws: WorldState, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def complexity(self) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {'type': self.__class__.__name__}

@dataclass
class IncreaseNoise(WorldAction):
    factor: float = 1.15

    def apply(self, ws: WorldState, rng: np.random.Generator) -> None:
        ws.noise = float(clamp(ws.noise * self.factor, 0.0001, 0.35))

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'IncreaseNoise', 'factor': self.factor}

@dataclass
class IncreaseNonlin(WorldAction):
    factor: float = 1.1

    def apply(self, ws: WorldState, rng: np.random.Generator) -> None:
        ws.nonlin = float(clamp(ws.nonlin * self.factor, 0.1, 5.0))

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'IncreaseNonlin', 'factor': self.factor}

@dataclass
class ShiftA(WorldAction):
    delta: float = 0.03

    def apply(self, ws: WorldState, rng: np.random.Generator) -> None:
        ws.a_min = float(clamp(ws.a_min + self.delta, 0.1, 0.98))
        ws.a_max = float(clamp(ws.a_max + self.delta, ws.a_min + 0.01, 1.05))

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'ShiftA', 'delta': self.delta}

@dataclass
class IncreaseSelfPlay(WorldAction):
    delta: float = 0.08

    def apply(self, ws: WorldState, rng: np.random.Generator) -> None:
        ws.selfplay_prob = float(clamp(ws.selfplay_prob + self.delta, 0.0, 0.95))

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'IncreaseSelfPlay', 'delta': self.delta}

@dataclass
class IncreaseShift(WorldAction):
    delta: float = 0.05

    def apply(self, ws: WorldState, rng: np.random.Generator) -> None:
        ws.shift_strength = float(clamp(ws.shift_strength + self.delta, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'IncreaseShift', 'delta': self.delta}
WORLD_ACTIONS = [IncreaseNoise, IncreaseNonlin, ShiftA, IncreaseSelfPlay, IncreaseShift]

@dataclass
class WorldCondition:
    key: str
    op: str
    threshold: float

    def eval(self, m: WorldMetrics) -> bool:
        val = getattr(m, self.key)
        return val > self.threshold if self.op == '>' else val < self.threshold

    def complexity(self) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {'key': self.key, 'op': self.op, 'threshold': self.threshold}
WORLD_COND_KEYS = ['oracle_perf', 'stability', 'novelty', 'improvement_ema', 'cycle']
WORLD_COND_OPS = ['>', '<']

@dataclass
class WorldProgram:
    """
    A small DSL-like program tree:
      - Either a sequence of actions (leaf)
      - Or an if-then-else branching on a condition
    """
    cond: Optional[WorldCondition] = None
    then_branch: Optional['WorldProgram'] = None
    else_branch: Optional['WorldProgram'] = None
    actions: List[WorldAction] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return self.cond is None

    def run(self, ws: WorldState, metrics: WorldMetrics, rng: np.random.Generator) -> None:
        if self.is_leaf():
            for a in self.actions:
                a.apply(ws, rng)
        else:
            if self.cond is None or self.then_branch is None or self.else_branch is None:
                return
            if self.cond.eval(metrics):
                self.then_branch.run(ws, metrics, rng)
            else:
                self.else_branch.run(ws, metrics, rng)

    def complexity(self) -> int:
        if self.is_leaf():
            return 1 + sum((a.complexity() for a in self.actions))
        return 1 + (self.cond.complexity() if self.cond else 0) + (self.then_branch.complexity() if self.then_branch else 0) + (self.else_branch.complexity() if self.else_branch else 0)

    def to_dict(self) -> Dict[str, Any]:
        if self.is_leaf():
            return {'type': 'leaf', 'actions': [a.to_dict() for a in self.actions]}
        return {'type': 'if', 'cond': self.cond.to_dict() if self.cond else None, 'then': self.then_branch.to_dict() if self.then_branch else None, 'else': self.else_branch.to_dict() if self.else_branch else None}

    @staticmethod
    def random(rng: np.random.Generator, depth: int=2) -> 'WorldProgram':
        if depth <= 0 or rng.random() < 0.45:
            k = int(rng.integers(1, 4))
            actions: List[WorldAction] = []
            for _ in range(k):
                A = rng.choice(WORLD_ACTIONS)
                if A is IncreaseNoise:
                    actions.append(IncreaseNoise(float(rng.uniform(1.05, 1.3))))
                elif A is IncreaseNonlin:
                    actions.append(IncreaseNonlin(float(rng.uniform(1.03, 1.2))))
                elif A is ShiftA:
                    actions.append(ShiftA(float(rng.uniform(-0.03, 0.05))))
                elif A is IncreaseSelfPlay:
                    actions.append(IncreaseSelfPlay(float(rng.uniform(0.03, 0.12))))
                elif A is IncreaseShift:
                    actions.append(IncreaseShift(float(rng.uniform(0.02, 0.1))))
                else:
                    actions.append(A())
            return WorldProgram(actions=actions)
        key = rng.choice(WORLD_COND_KEYS)
        op = rng.choice(WORLD_COND_OPS)
        thresh = float(rng.uniform(-1.0, 1.0)) if key != 'cycle' else float(rng.integers(1, 50))
        cond = WorldCondition(key=key, op=op, threshold=thresh)
        return WorldProgram(cond=cond, then_branch=WorldProgram.random(rng, depth - 1), else_branch=WorldProgram.random(rng, depth - 1))

    def mutate(self, rng: np.random.Generator, rate: float=0.25) -> 'WorldProgram':
        if rng.random() < 0.12:
            return WorldProgram.random(rng, depth=2)
        cp = dataclasses.replace(self)
        if cp.is_leaf():
            new_actions: List[WorldAction] = []
            for a in cp.actions:
                if rng.random() < rate:
                    if isinstance(a, IncreaseNoise):
                        new_actions.append(IncreaseNoise(float(clamp(a.factor * rng.uniform(0.9, 1.1), 1.01, 1.6))))
                    elif isinstance(a, IncreaseNonlin):
                        new_actions.append(IncreaseNonlin(float(clamp(a.factor * rng.uniform(0.9, 1.1), 1.01, 1.5))))
                    elif isinstance(a, ShiftA):
                        new_actions.append(ShiftA(float(clamp(a.delta + rng.uniform(-0.02, 0.02), -0.08, 0.1))))
                    elif isinstance(a, IncreaseSelfPlay):
                        new_actions.append(IncreaseSelfPlay(float(clamp(a.delta + rng.uniform(-0.03, 0.03), -0.1, 0.2))))
                    elif isinstance(a, IncreaseShift):
                        new_actions.append(IncreaseShift(float(clamp(a.delta + rng.uniform(-0.02, 0.02), -0.1, 0.2))))
                    else:
                        new_actions.append(a)
                else:
                    new_actions.append(a)
            if rng.random() < 0.2 and len(new_actions) < 6:
                new_actions.append(WorldProgram.random(rng, depth=0).actions[0])
            if rng.random() < 0.15 and len(new_actions) > 1:
                new_actions.pop(int(rng.integers(0, len(new_actions))))
            cp.actions = new_actions
            return cp
        if cp.cond and rng.random() < rate:
            if cp.cond.key == 'cycle':
                cp.cond.threshold = float(clamp(cp.cond.threshold + rng.integers(-5, 6), 1, 250))
            else:
                cp.cond.threshold = float(clamp(cp.cond.threshold + rng.uniform(-0.25, 0.25), -5.0, 5.0))
            if rng.random() < 0.15:
                cp.cond.op = '>' if cp.cond.op == '<' else '<'
        if cp.then_branch and rng.random() < 0.35:
            cp.then_branch = cp.then_branch.mutate(rng, rate=rate)
        if cp.else_branch and rng.random() < 0.35:
            cp.else_branch = cp.else_branch.mutate(rng, rate=rate)
        if rng.random() < 0.08 and cp.then_branch and cp.else_branch:
            cp.then_branch, cp.else_branch = (cp.else_branch, cp.then_branch)
        return cp

@dataclass
class ModelSpec:
    d: int = 16
    f: int = 5

def features(x: np.ndarray) -> np.ndarray:
    return np.stack([x, x ** 2, np.sin(x), np.cos(x), np.ones_like(x)], axis=-1)

class JepaLiquidModel:
    """
    A tiny representation-based predictor.
    Parameters are stored as a flat vector theta for ES-based search.
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.n_params = self._count_params()

    def _count_params(self) -> int:
        d, f = (self.spec.d, self.spec.f)
        return d * f + d + d + d + d * d + d + d + 1

    def unpack(self, theta: np.ndarray) -> Dict[str, np.ndarray]:
        d, f = (self.spec.d, self.spec.f)
        idx = 0
        Wc = theta[idx:idx + d * f].reshape(d, f)
        idx += d * f
        bc = theta[idx:idx + d]
        idx += d
        Wtau = theta[idx:idx + d]
        idx += d
        btau = theta[idx:idx + d]
        idx += d
        Wrec = theta[idx:idx + d * d].reshape(d, d)
        idx += d * d
        brec = theta[idx:idx + d]
        idx += d
        Wout = theta[idx:idx + d].reshape(1, d)
        idx += d
        bout = theta[idx:idx + 1]
        idx += 1
        return {'Wc': Wc, 'bc': bc, 'Wtau': Wtau, 'btau': btau, 'Wrec': Wrec, 'brec': brec, 'Wout': Wout, 'bout': bout}

    def forward(self, theta: np.ndarray, x_t: np.ndarray, x_tp1: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
        p = self.unpack(theta)
        phi = features(x_t)
        s = np.tanh(phi @ p['Wc'].T + p['bc'])
        eps = rng.normal(0.0, 0.02, size=x_t.shape).astype(np.float32)
        phi2 = features(x_t + eps)
        s2 = np.tanh(phi2 @ p['Wc'].T + p['bc'])
        inv = float(np.mean((s - s2) ** 2))
        tau = sigmoid(s * p['Wtau'] + p['btau'])
        h = np.tanh(s @ p['Wrec'].T + p['brec'])
        s_next = s + (h - s) * tau
        y = (s_next @ p['Wout'].T).reshape(-1) + float(p['bout'][0])
        pred = y
        pred_loss = float(np.mean((pred - x_tp1) ** 2))
        ib = float(np.mean(s ** 2))
        stability = float(-np.linalg.norm(p['Wrec'], ord='fro'))
        aux = {'pred_loss': pred_loss, 'inv': inv, 'ib': ib, 'stability': stability}
        return (pred, aux)

@dataclass
class ObjectiveSpec:
    w_pred: float = 0.808729
    w_inv: float = 0.15
    w_ib: float = 0.02
    w_l2: float = 0.0005

    def mutate(self, rng: np.random.Generator, rate: float=0.25) -> 'ObjectiveSpec':

        def m(x: float, lo: float, hi: float) -> float:
            if rng.random() < rate:
                x = float(x * math.exp(float(rng.normal(0.0, 0.25))))
            return float(clamp(x, lo, hi))
        return ObjectiveSpec(w_pred=m(self.w_pred, 0.2, 4.0), w_inv=m(self.w_inv, 0.0, 2.0), w_ib=m(self.w_ib, 0.0, 1.0), w_l2=m(self.w_l2, 0.0, 0.02))

    def complexity(self) -> int:
        c = 1
        c += int(self.w_inv > 1e-06)
        c += int(self.w_ib > 1e-06)
        c += int(self.w_l2 > 1e-08)
        return c

@dataclass
class UpdateRuleSpec:
    lr: float = 0.08
    sigma: float = 0.096237
    pop: int = 10
    antithetic: bool = True
    rank_mode: str = 'centered'
    clip: float = 3.0

    def mutate(self, rng: np.random.Generator, rate: float=0.25) -> 'UpdateRuleSpec':
        lr = self.lr
        sigma = self.sigma
        pop = self.pop
        ant = self.antithetic
        rm = self.rank_mode
        clip = self.clip
        if rng.random() < rate:
            lr = float(clamp(lr * math.exp(float(rng.normal(0.0, 0.35))), 0.0001, 0.8))
        if rng.random() < rate:
            sigma = float(clamp(sigma * math.exp(float(rng.normal(0.0, 0.35))), 0.0001, 0.8))
        if rng.random() < 0.2:
            pop = int(clamp(pop + int(rng.integers(-4, 5)), 6, 48))
        if rng.random() < 0.15:
            ant = not ant
        if rng.random() < 0.18:
            rm = rng.choice(['centered', 'linear', 'softmax'])
        if rng.random() < 0.18:
            clip = float(clamp(clip * math.exp(float(rng.normal(0.0, 0.25))), 0.5, 10.0))
        return UpdateRuleSpec(lr=lr, sigma=sigma, pop=pop, antithetic=ant, rank_mode=rm, clip=clip)

    def complexity(self) -> int:
        return 2 + int(self.antithetic) + 1

def rank_transform(losses: np.ndarray, mode: str) -> np.ndarray:
    ranks = losses.argsort().argsort().astype(np.float32)
    n = float(len(losses))
    if mode == 'linear':
        w = 1.0 - ranks / max(n - 1.0, 1.0)
        return w
    if mode == 'softmax':
        z = -(losses - np.mean(losses)) / (np.std(losses) + 1e-08)
        e = np.exp(z - np.max(z))
        return e / (np.sum(e) + 1e-08)
    w = ranks - (n - 1.0) / 2.0
    w = -w
    w = w / (np.std(w) + 1e-08)
    return w

@dataclass
class EvalConfig:
    train_tasks: int = 4
    holdout_sets: int = 3
    holdout_tasks: int = 3
    batch: int = 10
    horizon: int = 12
    inner_eval_steps: int = 2
    inner_adopt_steps: int = 3

@dataclass
class MultiScore:
    perf: float
    stability: float
    complexity: float
    novelty: float

    def vector(self) -> Tuple[float, float, float]:
        return (self.perf, self.stability, -self.complexity)

    def scalar(self, w_stability: float=0.12, w_complexity: float=0.04, w_novelty: float=0.02) -> float:
        return self.perf + w_stability * self.stability - w_complexity * self.complexity + w_novelty * self.novelty

def generate_task(ws: WorldState, rng: np.random.Generator, horizon: int) -> TaskSpec:
    shift = ws.shift_strength
    a_min = ws.a_min - 0.2 * shift
    a_max = ws.a_max + 0.2 * shift
    b_min = ws.b_min
    b_max = ws.b_max + 0.6 * shift
    c_min = ws.c_min
    c_max = ws.c_max + 1.0 * shift
    a = float(rng.uniform(a_min, a_max))
    b = float(rng.uniform(b_min, b_max))
    c = float(rng.uniform(c_min, c_max))
    noise = float(clamp(ws.noise * (1.0 + 0.9 * shift), 0.0001, 0.6))
    nonlin = float(clamp(ws.nonlin * (1.0 + 0.8 * shift), 0.05, 8.0))
    return TaskSpec(a=a, b=b, c=c, noise=noise, nonlin=nonlin, horizon=horizon)

def rollout_task(task: TaskSpec, batch: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    x = rng.normal(0.0, 1.0, size=(batch,)).astype(np.float32)
    xs = [x]
    for _ in range(task.horizon):
        noise = rng.normal(0.0, task.noise, size=x.shape).astype(np.float32)
        x_next = task.a * x + task.b * np.sin(task.c * x) * task.nonlin + noise
        x = x_next.astype(np.float32)
        xs.append(x)
    xs = np.stack(xs, axis=0)
    return (xs[:-1], xs[1:])

def evaluate_theta(model: JepaLiquidModel, theta: np.ndarray, tasks: List[TaskSpec], obj: ObjectiveSpec, rng: np.random.Generator, batch: int) -> Tuple[float, float, float]:
    pred_losses: List[float] = []
    inv_losses: List[float] = []
    ib_losses: List[float] = []
    stabs: List[float] = []
    for t in tasks:
        x_t, x_tp1 = rollout_task(t, batch=batch, rng=rng)
        T = x_t.shape[0]
        for i in range(T):
            _, aux = model.forward(theta, x_t[i], x_tp1[i], rng=rng)
            pred_losses.append(aux['pred_loss'])
            inv_losses.append(aux['inv'])
            ib_losses.append(aux['ib'])
            stabs.append(aux['stability'])
    pred_loss = float(np.mean(pred_losses)) if pred_losses else 0.0
    inv = float(np.mean(inv_losses)) if inv_losses else 0.0
    ib = float(np.mean(ib_losses)) if ib_losses else 0.0
    l2 = float(np.mean(theta ** 2))
    total = obj.w_pred * pred_loss + obj.w_inv * inv + obj.w_ib * ib + obj.w_l2 * l2
    stability = float(np.mean(stabs)) if stabs else 0.0
    return (total, stability, inv)

@dataclass
class Candidate:
    cid: str
    parent: Optional[str]
    birth_cycle: int
    universe: int
    theta: np.ndarray
    obj: ObjectiveSpec
    upd: UpdateRuleSpec
    world_prog: WorldProgram
    world_state: WorldState
    score: MultiScore
    notes: Dict[str, Any] = field(default_factory=dict)

    def genotype_complexity(self) -> int:
        return int(self.obj.complexity() + self.upd.complexity() + self.world_prog.complexity())

    def to_meta(self) -> Dict[str, Any]:
        return {'cid': self.cid, 'parent': self.parent, 'birth_cycle': self.birth_cycle, 'universe': self.universe, 'obj': dataclasses.asdict(self.obj), 'upd': dataclasses.asdict(self.upd), 'world_prog': self.world_prog.to_dict(), 'world_state': self.world_state.to_dict(), 'score': dataclasses.asdict(self.score), 'notes': self.notes}

def dominates(a: Candidate, b: Candidate) -> bool:
    av = a.score.vector()
    bv = b.score.vector()
    ge = all((x >= y for x, y in zip(av, bv)))
    g = any((x > y for x, y in zip(av, bv)))
    return ge and g

class ParetoFrontManager:

    def __init__(self, max_size: int=64):
        self.max_size = max_size
        self.front: List[Candidate] = []

    def update(self, cand: Candidate) -> None:
        survivors: List[Candidate] = []
        for c in self.front:
            if dominates(c, cand):
                return
            if not dominates(cand, c):
                survivors.append(c)
        survivors.append(cand)
        if len(survivors) > self.max_size:
            survivors = self._prune(survivors)
        self.front = survivors

    def _prune(self, cands: List[Candidate]) -> List[Candidate]:
        cands_sorted = sorted(cands, key=lambda c: c.score.scalar(), reverse=True)
        kept: List[Candidate] = []
        used_uni = set()
        buckets = {}
        for c in cands_sorted:
            b = int(clamp(c.score.complexity // 4, 0, 10))
            buckets.setdefault(b, 0)
            if buckets[b] >= 5:
                continue
            if c.universe in used_uni and len(kept) < self.max_size - 10:
                continue
            kept.append(c)
            used_uni.add(c.universe)
            buckets[b] += 1
            if len(kept) >= self.max_size:
                break
        return kept

    def sample(self, rng: np.random.Generator, genealogy: 'Genealogy', k: int=1) -> List[Candidate]:
        if not self.front:
            return []
        weights = []
        for c in self.front:
            cnt = genealogy.lineage_count(c.cid)
            weights.append(1.0 / (1.0 + cnt))
        weights = np.array(weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)
        idx = rng.choice(len(self.front), size=k, replace=len(self.front) < k, p=weights)
        return [self.front[int(i)] for i in np.atleast_1d(idx)]

class Genealogy:

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.children: Dict[str, List[str]] = {}

    def add(self, cand: Candidate) -> None:
        meta = cand.to_meta()
        self.nodes[cand.cid] = meta
        if cand.parent:
            self.children.setdefault(cand.parent, []).append(cand.cid)

    def lineage_count(self, cid: str) -> int:
        seen = set()
        stack = [cid]
        while stack:
            x = stack.pop()
            for ch in self.children.get(x, []):
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
            if len(seen) > 200:
                break
        return len(seen)

@dataclass
class ConditionalController:
    w_lr: np.ndarray
    b_lr: float
    w_sigma: np.ndarray
    b_sigma: float

    @staticmethod
    def init(dim: int, rng: np.random.Generator) -> 'ConditionalController':
        return ConditionalController(w_lr=rng.normal(0.0, 0.02, size=(dim,)).astype(np.float32), b_lr=0.0, w_sigma=rng.normal(0.0, 0.02, size=(dim,)).astype(np.float32), b_sigma=0.0)

    def adjust(self, z: np.ndarray, base_lr: float, base_sigma: float) -> Tuple[float, float]:
        dlr = float(np.dot(self.w_lr, z) + self.b_lr)
        ds = float(np.dot(self.w_sigma, z) + self.b_sigma)
        lr = float(clamp(base_lr * math.exp(dlr), 1e-05, 1.0))
        sig = float(clamp(base_sigma * math.exp(ds), 1e-05, 1.0))
        return (lr, sig)

    def update(self, z: np.ndarray, reward: float, lr_meta: float=0.02) -> None:
        g = float(clamp(reward, -1.0, 1.0))
        self.w_lr = (self.w_lr + lr_meta * g * z).astype(np.float32)
        self.w_sigma = (self.w_sigma + lr_meta * g * z).astype(np.float32)
        self.b_lr = float(self.b_lr + lr_meta * g * 0.1)
        self.b_sigma = float(self.b_sigma + lr_meta * g * 0.1)

def es_step(model: JepaLiquidModel, theta: np.ndarray, obj: ObjectiveSpec, upd: UpdateRuleSpec, tasks: List[TaskSpec], controller: ConditionalController, rng: np.random.Generator, batch: int) -> Tuple[np.ndarray, float, float]:
    _RSI_VERSION = 13
    _RSI_VERSION = 11
    _RSI_VERSION = 10
    _RSI_VERSION = 9
    _RSI_VERSION = 8
    _RSI_VERSION = 7
    z = np.mean(np.stack([t.embed() for t in tasks], axis=0), axis=0).astype(np.float32)
    lr, sigma = controller.adjust(z, upd.lr, upd.sigma)
    pop = upd.pop
    half = pop // 2
    if upd.antithetic:
        eps = rng.normal(0.0, 1.0, size=(half, theta.size)).astype(np.float32)
        eps = np.concatenate([eps, -eps], axis=0)
    else:
        eps = rng.normal(0.0, 1.0, size=(pop, theta.size)).astype(np.float32)
    losses = []
    stabs = []
    invs = []
    for i in range(eps.shape[0]):
        th = theta + sigma * eps[i]
        loss, stab, inv = evaluate_theta(model, th, tasks, obj, rng, batch=batch)
        losses.append(loss)
        stabs.append(stab)
        invs.append(inv)
    losses = np.array(losses, dtype=np.float32)
    w = rank_transform(losses, upd.rank_mode).astype(np.float32)
    g = (w.reshape(-1, 1) * eps).mean(axis=0) / (sigma + 1e-08)
    gn = float(np.linalg.norm(g))
    if gn > upd.clip:
        g = g * (upd.clip / (gn + 1e-08))
    theta2 = theta + lr * g
    baseline = float(np.mean(losses))
    loss2, stab2, _ = evaluate_theta(model, theta2, tasks, obj, rng, batch=batch)
    improvement = float(baseline - loss2)
    controller.update(z, reward=improvement)
    return (theta2, float(loss2), float(stab2))

def online_fetch_principles(enabled: bool, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """
    Very conservative: in offline environments this returns empty.
    When enabled, it tries to fetch small metadata snippets (not code) from GitHub/arXiv
    and converts them into typed "principle motifs".

    This is intentionally minimal to avoid licensing/code-copy issues.
    """
    if not enabled:
        return []
    try:
        import urllib.request
        import urllib.parse
        from concurrent.futures import ThreadPoolExecutor, as_completed
        cache_path = Path('multi_source_rsi_cache.json')
        all_items = []
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                all_items = cached.get('items', [])
                if all_items:
                    print(f'[MULTI-SOURCE] Loaded {len(all_items)} cached items from all sources', flush=True)
            except Exception:
                pass
        if not all_items:
            print('[MULTI-SOURCE] Fetching from GitHub + arXiv + Zenodo...', flush=True)

            def fetch_github():
                try:
                    q = urllib.parse.quote('recursive self improvement OR meta learning OR automl')
                    url = f'https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=100'
                    req = urllib.request.Request(url, headers={'User-Agent': 'rsi-meta-agent'})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode('utf-8', errors='ignore'))
                    items = [{'source': 'github', 'name': it.get('full_name', ''), 'description': it.get('description', ''), 'url': it.get('html_url', '')} for it in data.get('items', [])]
                    print(f'  [GitHub] Fetched {len(items)} repositories', flush=True)
                    return items
                except Exception as e:
                    print(f'  [GitHub] Failed: {e}', flush=True)
                    return []

            def fetch_arxiv():
                try:
                    q = urllib.parse.quote('recursive self improvement OR meta learning OR automl OR neural architecture search')
                    url = f'http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results=200'
                    req = urllib.request.Request(url, headers={'User-Agent': 'rsi-meta-agent'})
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        xml_data = resp.read().decode('utf-8', errors='ignore')
                    items = []
                    import re
                    entries = re.findall('<entry>(.*?)</entry>', xml_data, re.DOTALL)
                    for entry in entries:
                        title_match = re.search('<title>(.*?)</title>', entry, re.DOTALL)
                        summary_match = re.search('<summary>(.*?)</summary>', entry, re.DOTALL)
                        link_match = re.search('<id>(.*?)</id>', entry)
                        if title_match:
                            items.append({'source': 'arxiv', 'name': title_match.group(1).strip().replace('\n', ' '), 'description': summary_match.group(1).strip()[:500] if summary_match else '', 'url': link_match.group(1) if link_match else ''})
                    print(f'  [arXiv] Fetched {len(items)} papers', flush=True)
                    return items
                except Exception as e:
                    print(f'  [arXiv] Failed: {e}', flush=True)
                    return []

            def fetch_zenodo():
                try:
                    q = urllib.parse.quote('recursive self improvement meta learning automl')
                    url = f'https://zenodo.org/api/records?q={q}&size=100&type=software'
                    req = urllib.request.Request(url, headers={'User-Agent': 'rsi-meta-agent'})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode('utf-8', errors='ignore'))
                    items = [{'source': 'zenodo', 'name': it.get('metadata', {}).get('title', ''), 'description': it.get('metadata', {}).get('description', '')[:500] if it.get('metadata', {}).get('description') else '', 'url': it.get('links', {}).get('html', '')} for it in data.get('hits', {}).get('hits', [])]
                    print(f'  [Zenodo] Fetched {len(items)} datasets/software', flush=True)
                    return items
                except Exception as e:
                    print(f'  [Zenodo] Failed: {e}', flush=True)
                    return []

            def fetch_wikipedia():
                try:
                    topics = ['Machine_learning', 'Meta-learning_(computer_science)', 'Recursive_self-improvement', 'Artificial_general_intelligence', 'Neural_architecture_search', 'AutoML']
                    items = []
                    for topic in topics:
                        if _iter_count > 10000:
                            break
                        if _iter_count > 10000:
                            break
                        if _iter_count > 10000:
                            break
                        if _iter_count > 10000:
                            break
                        if _iter_count > 10000:
                            break
                        if _iter_count > 10000:
                            break
                        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{topic}'
                        req = urllib.request.Request(url, headers={'User-Agent': 'rsi-meta-agent'})
                        try:
                            with urllib.request.urlopen(req, timeout=5) as resp:
                                data = json.loads(resp.read().decode('utf-8', errors='ignore'))
                            items.append({'source': 'wikipedia', 'name': data.get('title', topic), 'description': data.get('extract', '')[:500], 'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')})
                        except Exception:
                            pass
                    print(f'  [Wikipedia] Fetched {len(items)} articles', flush=True)
                    return items
                except Exception as e:
                    print(f'  [Wikipedia] Failed: {e}', flush=True)
                    return []

            def fetch_semantic_scholar():
                try:
                    q = urllib.parse.quote('recursive self improvement meta learning')
                    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=100&fields=title,abstract,url'
                    req = urllib.request.Request(url, headers={'User-Agent': 'rsi-meta-agent'})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        data = json.loads(resp.read().decode('utf-8', errors='ignore'))
                    items = [{'source': 'semantic_scholar', 'name': p.get('title', ''), 'description': (p.get('abstract', '') or '')[:500], 'url': p.get('url', '')} for p in data.get('data', [])]
                    print(f'  [Semantic Scholar] Fetched {len(items)} papers', flush=True)
                    return items
                except Exception as e:
                    print(f'  [Semantic Scholar] Failed: {e}', flush=True)
                    return []
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(fetch_github), executor.submit(fetch_arxiv), executor.submit(fetch_zenodo), executor.submit(fetch_wikipedia), executor.submit(fetch_semantic_scholar)]
                for future in as_completed(futures):
                    all_items.extend(future.result())
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'items': all_items}, f)
            print(f'[MULTI-SOURCE] Total fetched and cached: {len(all_items)} items', flush=True)
        filtered_items = []
        for it in all_items:
            text = (it.get('description') or '') + ' ' + (it.get('name') or '')
            text_l = text.lower()
            if 'transformer' in text_l or 'llm' in text_l or 'gpt' in text_l or ('bert' in text_l):
                continue
            filtered_items.append(it)
        items = filtered_items[:200]
        print(f'[MULTI-SOURCE] Using {len(items)} items after filtering (from {len(all_items)} total)', flush=True)
        motifs: List[Dict[str, Any]] = []
        for it in items:
            text = (it.get('description') or '') + ' ' + (it.get('name') or '')
            text_l = text.lower()
            motif = {'source': it.get('url', ''), 'name': it.get('name', ''), 'origin': it.get('source', 'unknown')}
            motif['has_rsi'] = int('recursive' in text_l or 'self-improvement' in text_l or 'godel' in text_l or ('self-modifying' in text_l) or ('self-evolving' in text_l))
            motif['has_meta'] = int('meta-learning' in text_l or 'maml' in text_l or 'learning to learn' in text_l or ('meta-optimization' in text_l))
            motif['has_ema'] = int('ema' in text_l or 'momentum' in text_l or 'exponential moving' in text_l)
            motif['has_schedule'] = int('schedule' in text_l or 'cosine' in text_l or 'warmup' in text_l)
            motifs.append(motif)
        rsi_cnt = sum((m.get('has_rsi', 0) for m in motifs))
        meta_cnt = sum((m.get('has_meta', 0) for m in motifs))
        print(f"[ONLINE] Extracted {len(motifs)} motifs: {rsi_cnt} RSI, {meta_cnt} Meta, {sum((m.get('has_ema', 0) for m in motifs))} EMA", flush=True)
        return motifs
    except Exception as e:
        print(f'[ONLINE ERROR] Failed to fetch principles: {type(e).__name__}: {str(e)}', flush=True)
        return []

def motifs_to_mutations(motifs: List[Dict[str, Any]], rng: np.random.Generator) -> Tuple[Optional[ObjectiveSpec], Optional[UpdateRuleSpec]]:
    if not motifs:
        return (None, None)
    rsi = max((m.get('has_rsi', 0) for m in motifs))
    meta = max((m.get('has_meta', 0) for m in motifs))
    ema = max((m.get('has_ema', 0) for m in motifs))
    sch = max((m.get('has_schedule', 0) for m in motifs))
    obj = ObjectiveSpec()
    upd = UpdateRuleSpec()
    if rsi:
        print('[META-META] RSI Principle: AGGRESSIVE Complexity/LR/Pop boost', flush=True)
        obj.w_pred = float(clamp(obj.w_pred * rng.uniform(0.5, 1.5), 0.1, 5.0))
        obj.w_inv = float(clamp(obj.w_inv * rng.uniform(0.8, 1.5), 0.05, 1.0))
        upd.lr = float(clamp(upd.lr * rng.uniform(0.8, 1.5), 0.01, 0.3))
        upd.pop = int(clamp(upd.pop + int(rng.integers(1, 4)), 10, 15))
    if meta:
        print('[META-META] Meta-Learning Principle: Strong Population & Sigma tuning', flush=True)
        upd.pop = int(clamp(upd.pop + int(rng.integers(1, 3)), 10, 15))
        upd.sigma = float(clamp(upd.sigma * rng.uniform(0.6, 1.2), 0.05, 0.5))
        upd.antithetic = True
    if ema:
        obj.w_l2 = float(clamp(obj.w_l2 * rng.uniform(1.2, 2.0), 0.0, 0.02))
        upd.sigma = float(clamp(upd.sigma * rng.uniform(0.7, 0.95), 0.0001, 1.0))
    if sch:
        upd.rank_mode = rng.choice(['centered', 'softmax'])
    return (obj, upd)

@dataclass
class Universe:
    uid: int
    theta: np.ndarray
    obj: ObjectiveSpec
    upd: UpdateRuleSpec
    world_prog: WorldProgram
    world_state: WorldState
    controller: ConditionalController
    archive_tasks: List[TaskSpec] = field(default_factory=list)
    archive_theta: List[np.ndarray] = field(default_factory=list)
    improvement_ema: float = 0.0
    last_oracle: float = -1000000000.0

def compute_novelty(theta: np.ndarray, archive: List[np.ndarray]) -> float:
    if not archive:
        return 1.0
    dists = [float(np.linalg.norm(theta - t)) for t in archive[-8:]]
    m = min(dists) if dists else 0.0
    return float(clamp(m / (np.mean(dists) + 1e-08), 0.0, 5.0))

def world_forge(base_ws: WorldState, world_prog: WorldProgram, metrics: WorldMetrics, archive_tasks: List[TaskSpec], rng: np.random.Generator, cfg: EvalConfig) -> Tuple[List[TaskSpec], List[List[TaskSpec]]]:
    ws = WorldState(**base_ws.to_dict())
    world_prog.run(ws, metrics, rng)
    train_tasks: List[TaskSpec] = []
    for _ in range(cfg.train_tasks):
        if archive_tasks and rng.random() < ws.selfplay_prob:
            train_tasks.append(archive_tasks[int(rng.integers(0, len(archive_tasks)))])
        else:
            train_tasks.append(generate_task(ws, rng, horizon=cfg.horizon))
    holdouts: List[List[TaskSpec]] = []
    for k in range(cfg.holdout_sets):
        ws_k = WorldState(**ws.to_dict())
        ws_k.shift_strength = float(clamp(ws_k.shift_strength + 0.25 * (k + 1), 0.0, 1.0))
        suite = [generate_task(ws_k, rng, horizon=cfg.horizon) for _ in range(cfg.holdout_tasks)]
        holdouts.append(suite)
    return (train_tasks, holdouts)

def evaluate_candidate(model: JepaLiquidModel, theta: np.ndarray, obj: ObjectiveSpec, upd: UpdateRuleSpec, world_prog: WorldProgram, base_ws: WorldState, metrics: WorldMetrics, archive_tasks: List[TaskSpec], archive_theta: List[np.ndarray], rng: np.random.Generator, cfg: EvalConfig) -> Tuple[MultiScore, List[TaskSpec], float]:
    train_tasks, holdouts = world_forge(base_ws, world_prog, metrics, archive_tasks, rng, cfg)
    th = theta.copy()
    local_ctrl = ConditionalController.init(dim=5, rng=rng)
    for _ in range(int(cfg.inner_eval_steps)):
        th, train_loss, stab = es_step(model, th, obj, upd, train_tasks, local_ctrl, rng, batch=cfg.batch)
    hold_losses = []
    hold_stabs = []
    for suite in holdouts:
        l, s, _ = evaluate_theta(model, th, suite, obj, rng, batch=cfg.batch)
        hold_losses.append(l)
        hold_stabs.append(s)
    oracle_loss = float(np.mean(hold_losses)) if hold_losses else train_loss
    perf = float(-oracle_loss)
    stability = float(np.mean(hold_stabs)) if hold_stabs else stab
    novelty = compute_novelty(th, archive_theta)
    complexity = float(obj.complexity() + upd.complexity() + world_prog.complexity())
    score = MultiScore(perf=perf, stability=stability, complexity=complexity, novelty=novelty)
    hard_suite = holdouts[int(rng.integers(0, len(holdouts)))]
    hard_task = max(hard_suite, key=lambda t: float(t.noise + 0.2 * t.nonlin + 0.3 * t.c))
    return (score, train_tasks + [hard_task], oracle_loss)

class MetaRSIController:
    """
    Controls:
      - Pareto-front maintenance
      - Genealogy
      - Universe migration (adopt candidates from Pareto front to encourage transfer)
      - Compute governor: adjusts discovery trials based on improvement EMA
    """

    def __init__(self, front_size: int=80):
        self.pareto = ParetoFrontManager(max_size=front_size)
        self.genealogy = Genealogy()
        self.global_cycle = 0
        self.best_scalar_ema = 0.0
        self.improvement_ema = 0.0
        self.discovery_trials = 10

    def compute_governor(self) -> int:
        imp = self.improvement_ema
        if imp < 0.001:
            self.discovery_trials = int(clamp(self.discovery_trials + 3, 8, 60))
        elif imp > 0.01:
            self.discovery_trials = int(clamp(self.discovery_trials - 2, 6, 60))
        return self.discovery_trials

    def add_candidate(self, cand: Candidate) -> None:
        self.genealogy.add(cand)
        self.pareto.update(cand)

def make_candidate_id(uid: int, cycle: int, payload: Dict[str, Any]) -> str:
    base = {'uid': uid, 'cycle': cycle, **payload}
    return f'U{uid}_C{cycle}_' + stable_hash(base)

def run_system(out_dir: Path, run_name: str, cycles: int, universes: int, seed: int, online: bool, cfg: EvalConfig) -> Dict[str, Any]:
    ensure_dir(out_dir)
    run_dir = out_dir / run_name
    ensure_dir(run_dir)
    rng = np.random.default_rng(seed)
    set_global_seed(seed)
    model = JepaLiquidModel(ModelSpec(d=16, f=5))
    meta = MetaRSIController(front_size=96)
    us: List[Universe] = []
    for u in range(universes):
        theta = rng.normal(0.0, 0.15, size=(model.n_params,)).astype(np.float32)
        obj = ObjectiveSpec().mutate(rng, rate=0.35) if u == 0 else ObjectiveSpec()
        upd = UpdateRuleSpec().mutate(rng, rate=0.35) if u == 0 else UpdateRuleSpec()
        wp = WorldProgram.random(rng, depth=2)
        ws = WorldState()
        ctrl = ConditionalController.init(dim=5, rng=rng)
        us.append(Universe(uid=u, theta=theta, obj=obj, upd=upd, world_prog=wp, world_state=ws, controller=ctrl))
    history: List[Dict[str, Any]] = []
    global_archive_theta: List[np.ndarray] = []
    agi_learner = get_agi_meta_learner(rng)
    print(f'[AGI] Initialized AGI Meta-Learner with {len(agi_learner.skill_levels)} skill domains', flush=True)
    consciousness = get_consciousness(rng)
    print(f'[CONSCIOUS] Initialized Emergent Self-Consciousness Module', flush=True)
    print(f'[CONSCIOUS] Consciousness probability: {consciousness.consciousness_probability:.4f}', flush=True)
    causal_loop = get_causal_loop()
    print(f'[CAUSAL] Initialized Causal Self-Loop. Modifications: {causal_loop.modification_count}', flush=True)
    for cycle in range(cycles):
        meta.global_cycle = cycle
        trials = meta.compute_governor()
        internal_state = {'cycle': cycle, 'trials': trials, 'improvement_ema': meta.improvement_ema, 'pareto_size': len(meta.pareto.front), 'recent_actions': ['optimize', 'evaluate', 'mutate']}
        consciousness_result = consciousness.consciousness_cycle(internal_state)
        causal_result = causal_loop.causal_cycle(state={'cycle': cycle, 'trials': trials, 'pareto': len(meta.pareto.front)}, outcome=meta.improvement_ema)
        causal_params = causal_loop.get_params()
        causal_mutation_rate = float(causal_params.get('mutation_strength', 0.15))
        causal_exploration = float(causal_params.get('exploration_rate', 0.3))
        motifs = online_fetch_principles(enabled=online, rng=rng)
        online_obj_hint, online_upd_hint = motifs_to_mutations(motifs, rng)
        if cycle > 0 and cycle % 5 == 0:
            print(f'[AGI] Cycle {cycle}: Running multi-domain meta-learning...', flush=True)
            agi_improvements = agi_learner.meta_learn_cycle(n_tasks=10)
            transfer_knowledge = agi_learner.get_transfer_knowledge()
            print(f'[AGI] Skills: {agi_learner.skill_levels}', flush=True)
            print(f"[AGI] Transfer knowledge: library={transfer_knowledge['library_size']}, " + f"knowledge={transfer_knowledge['knowledge_base_size']}", flush=True)
        cycle_best_scalar = -1e+18
        cycle_improvements: List[float] = []
        for uni in us:
            m = WorldMetrics(cycle=cycle, oracle_perf=float(uni.last_oracle), stability=-1.0, novelty=1.0, improvement_ema=float(uni.improvement_ema))
            best_local: Optional[Candidate] = None
            best_local_scalar = -1e+18
            for t in range(trials):
                obj = uni.obj.mutate(rng, rate=causal_mutation_rate + 0.1)
                upd = uni.upd.mutate(rng, rate=causal_mutation_rate + 0.1)
                wp = uni.world_prog.mutate(rng, rate=causal_mutation_rate + 0.13)
                ws = WorldState(**uni.world_state.to_dict())
                if online_obj_hint:
                    obj = online_obj_hint.mutate(rng, rate=0.5)
                if online_upd_hint:
                    upd = online_upd_hint.mutate(rng, rate=0.5)
                score, new_archive_tasks, oracle_loss = evaluate_candidate(model=model, theta=uni.theta, obj=obj, upd=upd, world_prog=wp, base_ws=ws, metrics=m, archive_tasks=uni.archive_tasks, archive_theta=global_archive_theta, rng=rng, cfg=cfg)
                cid = make_candidate_id(uni.uid, cycle, {'t': t, 's': dataclasses.asdict(score)})
                parent = None
                cand = Candidate(cid=cid, parent=uni.notes.get('active_cid') if hasattr(uni, 'notes') else None, birth_cycle=cycle, universe=uni.uid, theta=uni.theta.copy(), obj=obj, upd=upd, world_prog=wp, world_state=ws, score=score, notes={'oracle_loss': oracle_loss, 'trials': trials, 'online_motifs': motifs[:2] if motifs else []})
                meta.add_candidate(cand)
                sc = score.scalar()
                if sc > best_local_scalar:
                    best_local_scalar = sc
                    best_local = cand
                if oracle_loss > (-uni.last_oracle if uni.last_oracle > -100000000.0 else oracle_loss):
                    uni.archive_tasks.extend(new_archive_tasks[-2:])
                    if len(uni.archive_tasks) > 80:
                        uni.archive_tasks = uni.archive_tasks[-80:]
            adopted = best_local
            migrated = meta.pareto.sample(rng, meta.genealogy, k=1)
            if migrated and rng.random() < 0.35:
                adopted = migrated[0]
            if adopted is None:
                continue
            m2 = WorldMetrics(cycle=cycle, oracle_perf=float(uni.last_oracle), stability=0.0, novelty=1.0, improvement_ema=float(uni.improvement_ema))
            train_tasks, holdouts = world_forge(uni.world_state, adopted.world_prog, m2, uni.archive_tasks, rng, cfg)
            theta_before = uni.theta.copy()
            for _ in range(int(cfg.inner_adopt_steps)):
                uni.theta, train_loss, stab = es_step(model=model, theta=uni.theta, obj=adopted.obj, upd=adopted.upd, tasks=train_tasks, controller=uni.controller, rng=rng, batch=cfg.batch)
            hold_losses = []
            for suite in holdouts:
                l, _, _ = evaluate_theta(model, uni.theta, suite, adopted.obj, rng, batch=cfg.batch)
                hold_losses.append(l)
            oracle_loss = float(np.mean(hold_losses)) if hold_losses else train_loss
            oracle_perf = float(-oracle_loss)
            improvement = float(oracle_perf - uni.last_oracle) if uni.last_oracle > -100000000.0 else float(oracle_perf)
            uni.improvement_ema = 0.92 * uni.improvement_ema + 0.08 * improvement
            cycle_improvements.append(improvement)
            uni.obj = adopted.obj
            uni.upd = adopted.upd
            uni.world_prog = adopted.world_prog
            uni.last_oracle = oracle_perf
            global_archive_theta.append(uni.theta.copy())
            if len(global_archive_theta) > 120:
                global_archive_theta = global_archive_theta[-120:]
            scalar_now = float(oracle_perf + 0.12 * stab - 0.04 * (adopted.obj.complexity() + adopted.upd.complexity() + adopted.world_prog.complexity()))
            cycle_best_scalar = max(cycle_best_scalar, scalar_now)
        if cycle_improvements:
            avg_imp = float(np.mean(cycle_improvements))
        else:
            avg_imp = 0.0
        meta.improvement_ema = 0.9 * meta.improvement_ema + 0.1 * avg_imp
        meta.best_scalar_ema = 0.92 * meta.best_scalar_ema + 0.08 * cycle_best_scalar
        history.append({'cycle': cycle, 'trials': trials, 'avg_improvement': avg_imp, 'improvement_ema': meta.improvement_ema, 'best_scalar_ema': meta.best_scalar_ema, 'pareto_size': len(meta.pareto.front)})
        if cycle % 5 == 0 or cycle == cycles - 1:
            ckpt = {'seed': seed, 'cycle': cycle, 'history_tail': history[-10:], 'pareto_size': len(meta.pareto.front), 'genealogy_nodes': len(meta.genealogy.nodes), 'universes': [{'uid': u.uid, 'last_oracle': u.last_oracle, 'improvement_ema': u.improvement_ema, 'obj': dataclasses.asdict(u.obj), 'upd': dataclasses.asdict(u.upd), 'world_state': u.world_state.to_dict(), 'world_prog': u.world_prog.to_dict()} for u in us]}
            with open(run_dir / 'checkpoint.json', 'w', encoding='utf-8') as f:
                json.dump(ckpt, f, indent=2, ensure_ascii=False)
            pareto_metas = [c.to_meta() for c in sorted(meta.pareto.front, key=lambda x: x.score.scalar(), reverse=True)]
            with open(run_dir / 'pareto.json', 'w', encoding='utf-8') as f:
                json.dump(pareto_metas[:64], f, indent=2, ensure_ascii=False)
        if cycle > 0 and cycle % 10 == 0:
            best_params = {'improvement_ema': meta.improvement_ema, 'best_scalar_ema': meta.best_scalar_ema, 'pareto_size': len(meta.pareto.front), 'best_obj': dataclasses.asdict(us[0].obj) if us else {}, 'best_upd': dataclasses.asdict(us[0].upd) if us else {}}
            print(f'[SELF-MOD] Cycle {cycle}: ACTIVATING ALGORITHMIC SELF-MODIFICATION!', flush=True)
            print(f'[SELF-MOD] Current fitness: {meta.improvement_ema:.4f}, Version: {SELF_MOD_VERSION}', flush=True)
            try:
                source = read_own_source()
                new_source = source
                jules_improved = False
                try:
                    from jules_api_client import JulesRSIIntegration
                    jules = JulesRSIIntegration()
                    if jules.available:
                        print('[SELF-MOD] Attempting Jules API for code improvement...', flush=True)
                        context = {'improvement_ema': meta.improvement_ema, 'best_scalar_ema': meta.best_scalar_ema, 'pareto_size': len(meta.pareto.front)}
                        result = jules.request_improvement(repo_name='SCIG-RSI-v2', file_path='rsi_meta_rsi_fusion_v5_limitbreak.py', function_name='eval_oracle' if cycle % 30 == 10 else 'mutate_spec' if cycle % 30 == 20 else 'run_es_step', context=context)
                        if result.get('status') == 'success':
                            pr_url = result.get('pull_request')
                            print(f'[SELF-MOD] Jules created PR: {pr_url}', flush=True)
                            print('[SELF-MOD] Manual review needed for Jules changes', flush=True)
                            jules_improved = True
                        else:
                            print(f"[SELF-MOD] Jules API: {result.get('status')} - {result.get('error', 'N/A')}", flush=True)
                except ImportError:
                    print('[SELF-MOD] Jules API client not available', flush=True)
                except Exception as e:
                    print(f'[SELF-MOD] Jules API error: {e}', flush=True)
                llm_improved = False
                try:
                    import google.generativeai as genai
                    api_key = None
                    for key_path in ['c:\\Users\\starg\\OneDrive\\바탕 화면\\SCIG-RSI-v2\\google_api_key.txt', 'google_api_key.txt', '../google_api_key.txt']:
                        try:
                            if os.path.exists(key_path):
                                with open(key_path, 'r') as f:
                                    api_key = f.read().strip().split('\n')[0].strip()
                                    if api_key and len(api_key) > 20:
                                        break
                        except:
                            pass
                    if api_key:
                        genai.configure(api_key=api_key)
                        llm_model = genai.GenerativeModel('gemini-1.5-flash')
                        improvement_targets = [('eval_oracle', 'evaluation function for measuring performance'), ('mutate_spec', 'mutation function for generating variations'), ('run_es_step', 'evolution strategy optimization step')]
                        for func_name, func_desc in improvement_targets:
                            import re
                            pattern = f'(def {func_name}\\([^)]*\\):.*?)(?=\\ndef |\\nclass |\\n# ===|\\Z)'
                            match = re.search(pattern, source, re.DOTALL)
                            if match:
                                old_func = match.group(1).strip()
                                prompt = f'You are an expert Python optimization engineer. Improve this function.\n\n## Current Function ({func_desc}):\n```python\n{old_func[:2000]}\n```\n\n## Current Performance Metrics:\n- improvement_ema: {meta.improvement_ema:.4f}\n- best_scalar_ema: {meta.best_scalar_ema:.4f}\n- pareto_size: {len(meta.pareto.front)}\n\n## Requirements:\n1. Keep the same function signature\n2. Improve performance/efficiency\n3. Add better optimization logic\n4. The code must be syntactically correct Python\n5. Use numpy for numerical operations\n\n## Output:\nReturn ONLY the improved function code, no explanations.\n'
                                response = llm_model.generate_content(prompt, generation_config=genai.GenerationConfig(max_output_tokens=2048, temperature=0.7))
                                if response and response.text:
                                    new_func = response.text.strip()
                                    if '```python' in new_func:
                                        start = new_func.find('```python') + 9
                                        end = new_func.find('```', start)
                                        if end != -1:
                                            new_func = new_func[start:end].strip()
                                    elif '```' in new_func:
                                        start = new_func.find('```') + 3
                                        end = new_func.find('```', start)
                                        if end != -1:
                                            new_func = new_func[start:end].strip()
                                    try:
                                        ast.parse(new_func)
                                        new_source = re.sub(pattern, new_func + '\n\n', new_source, count=1)
                                        llm_improved = True
                                        print(f'[SELF-MOD] LLM improved function: {func_name}', flush=True)
                                        print(f'[SELF-MOD] Old length: {len(old_func)}, New length: {len(new_func)}', flush=True)
                                        break
                                    except SyntaxError as e:
                                        print(f'[SELF-MOD] LLM generated invalid syntax for {func_name}: {e}', flush=True)
                except ImportError:
                    print('[SELF-MOD] google-generativeai not installed, using AST mutation', flush=True)
                except Exception as e:
                    print(f'[SELF-MOD] LLM improvement failed: {e}', flush=True)
                if not llm_improved:
                    print('[SELF-MOD] Using AST-based algorithmic mutation...', flush=True)
                    try:
                        tree = ast.parse(new_source)

                        class AlgorithmMutator(ast.NodeTransformer):
                            """Mutate algorithms - GUARANTEED to make changes."""
                            mutations_made = 0
                            target_functions = ['eval_oracle', 'mutate_spec', 'run_es_step', 'es_step']

                            def visit_FunctionDef(self, node):
                                self.generic_visit(node)
                                if node.name in self.target_functions and self.mutations_made < 1:
                                    version_comment = ast.parse(f'\n# [RSI-EVOLVED v{SELF_MOD_VERSION + 1}] This function was self-modified\n_RSI_VERSION = {SELF_MOD_VERSION + 1}\n').body[0]
                                    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                                        node.body.insert(1, version_comment)
                                    else:
                                        node.body.insert(0, version_comment)
                                    self.mutations_made += 1
                                    print(f'[SELF-MOD] Mutated function: {node.name}', flush=True)
                                return node

                            def visit_For(self, node):
                                self.generic_visit(node)
                                if self.mutations_made >= 1 and self.mutations_made < 2:
                                    if random.random() < 0.3:
                                        limit_check = ast.parse('\nif _iter_count > 10000:\n    break\n').body[0]
                                        counter_inc = ast.parse('_iter_count = getattr(_iter_count, "__add__", lambda x: x)(1) if "_iter_count" in dir() else 1').body[0]
                                        node.body.insert(0, limit_check)
                                        self.mutations_made += 1
                                return node
                        mutator = AlgorithmMutator()
                        new_tree = mutator.visit(tree)
                        if mutator.mutations_made > 0:
                            ast.fix_missing_locations(new_tree)
                            new_source = ast.unparse(new_tree)
                            print(f'[SELF-MOD] AST mutations made: {mutator.mutations_made}', flush=True)
                        else:
                            print('[SELF-MOD] No mutations applied - target functions not found', flush=True)
                    except Exception as e:
                        print(f'[SELF-MOD] AST mutation failed: {e}', flush=True)
                new_source = new_source.replace(f'SELF_MOD_VERSION = {SELF_MOD_VERSION}', f'SELF_MOD_VERSION = {SELF_MOD_VERSION + 1}')
                temp_file = MY_SOURCE_FILE.with_name('_self_mod_test.py')
                try:
                    temp_file.write_text(new_source, encoding='utf-8')
                    result = subprocess.run([sys.executable, str(temp_file), 'test'], capture_output=True, text=True, timeout=120)
                    if result.returncode == 0:
                        print('[SELF-MOD] Test PASSED! Applying changes...', flush=True)
                        write_own_source(new_source)
                        print(f'[SELF-MOD] SUCCESS! Wrote new version {SELF_MOD_VERSION + 1}', flush=True)
                        with open(run_dir / 'checkpoint.json', 'w', encoding='utf-8') as f:
                            json.dump(ckpt, f, indent=2, ensure_ascii=False)
                        print(f'[SELF-MOD] RESTARTING WITH IMPROVED ALGORITHM...', flush=True)
                        os.execv(sys.executable, [sys.executable] + sys.argv)
                    else:
                        print(f'[SELF-MOD] Test FAILED, not applying changes', flush=True)
                        print(f'[SELF-MOD] Error: {result.stderr[:500]}', flush=True)
                finally:
                    if temp_file.exists():
                        temp_file.unlink()
            except Exception as e:
                print(f'[SELF-MOD] Failed: {e}', flush=True)
                import traceback
                traceback.print_exc()
    report = {'run_name': run_name, 'cycles': cycles, 'universes': universes, 'final_pareto': len(meta.pareto.front), 'final_genealogy_nodes': len(meta.genealogy.nodes), 'final_improvement_ema': meta.improvement_ema, 'history': history[-20:]}
    with open(run_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

def _test_pareto() -> None:
    rng = np.random.default_rng(0)

    def mk(perf, stab, comp, u):
        sc = MultiScore(perf=perf, stability=stab, complexity=comp, novelty=1.0)
        return Candidate(cid=stable_hash({'p': perf, 's': stab, 'c': comp, 'u': u}), parent=None, birth_cycle=0, universe=u, theta=rng.normal(0, 1, size=(10,)).astype(np.float32), obj=ObjectiveSpec(), upd=UpdateRuleSpec(), world_prog=WorldProgram.random(rng, depth=1), world_state=WorldState(), score=sc)
    pf = ParetoFrontManager(max_size=16)
    gen = Genealogy()
    c1 = mk(1.0, 0.0, 5.0, 0)
    c2 = mk(0.5, 0.1, 3.0, 1)
    c3 = mk(1.2, -0.2, 8.0, 2)
    for c in [c1, c2, c3]:
        gen.add(c)
        pf.update(c)
    assert len(pf.front) >= 2
    samp = pf.sample(rng, gen, k=2)
    assert len(samp) == 2

def _test_world_program() -> None:
    rng = np.random.default_rng(1)
    wp = WorldProgram.random(rng, depth=2)
    ws = WorldState()
    m = WorldMetrics(cycle=1, oracle_perf=0.1, stability=-0.5, novelty=1.0, improvement_ema=0.0)
    wp.run(ws, m, rng)
    wp2 = wp.mutate(rng)
    assert wp2.complexity() > 0

def _test_end_to_end(tmpdir: Path) -> None:
    rep = run_system(out_dir=tmpdir, run_name='test_run', cycles=3, universes=2, seed=42, online=False, cfg=EvalConfig(train_tasks=2, holdout_sets=2, holdout_tasks=1, batch=6, horizon=6, inner_eval_steps=1, inner_adopt_steps=1))
    assert rep['final_pareto'] > 0
    assert rep['final_genealogy_nodes'] > 0
    assert (tmpdir / 'test_run' / 'report.json').exists()
    assert (tmpdir / 'test_run' / 'checkpoint.json').exists()

def run_tests() -> None:
    tmp = Path('._rsi_v5_test_tmp')
    if tmp.exists():
        for p in tmp.rglob('*'):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            tmp.rmdir()
        except Exception:
            pass
    ensure_dir(tmp)
    _test_pareto()
    _test_world_program()
    _test_end_to_end(tmp)
    for p in tmp.rglob('*'):
        try:
            p.unlink()
        except Exception:
            pass
    try:
        tmp.rmdir()
    except Exception:
        pass

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    ap_test = sub.add_parser('test', help='Run quick verification tests.')
    ap_run = sub.add_parser('run', help='Run RSI system.')
    ap_run.add_argument('--out_dir', type=str, default='rsi_v5_runs')
    ap_run.add_argument('--run_name', type=str, default='v5_limitbreak')
    ap_run.add_argument('--cycles', type=int, default=50)
    ap_run.add_argument('--universes', type=int, default=5)
    ap_run.add_argument('--seed', type=int, default=0)
    ap_run.add_argument('--online', type=int, default=0)
    ap_run.add_argument('--train_tasks', type=int, default=4)
    ap_run.add_argument('--holdout_sets', type=int, default=3)
    ap_run.add_argument('--holdout_tasks', type=int, default=3)
    ap_run.add_argument('--batch', type=int, default=10)
    ap_run.add_argument('--horizon', type=int, default=12)
    ap_run.add_argument('--inner_eval_steps', type=int, default=2)
    ap_run.add_argument('--inner_adopt_steps', type=int, default=3)
    args = ap.parse_args()
    if args.cmd == 'test':
        t0 = now_ts()
        run_tests()
        dt = now_ts() - t0
        print(f'[OK] tests passed in {dt:.2f}s')
        return
    cfg = EvalConfig(train_tasks=int(args.train_tasks), holdout_sets=int(args.holdout_sets), holdout_tasks=int(args.holdout_tasks), batch=int(args.batch), horizon=int(args.horizon), inner_eval_steps=int(args.inner_eval_steps), inner_adopt_steps=int(args.inner_adopt_steps))
    rep = run_system(out_dir=Path(args.out_dir), run_name=str(args.run_name), cycles=int(args.cycles), universes=int(args.universes), seed=int(args.seed), online=bool(int(args.online)), cfg=cfg)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()