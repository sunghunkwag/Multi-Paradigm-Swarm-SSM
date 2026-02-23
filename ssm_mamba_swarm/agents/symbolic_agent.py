import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.special import digamma
from scipy.spatial import cKDTree
from .base_agent import BaseAgent, AgentProposal

class VoidNode:
    """A truly open-ended AST node with all mathematical primitives."""
    def __init__(self, op: str, left=None, right=None, p_idx: Optional[int] = None):
        self.op = op
        self.left = left
        self.right = right
        self.p_idx = p_idx

    def eval(self, x_p, x_s, params):
        try:
            if self.op == 'c': return params[self.p_idx]
            if self.op == 'p': return x_p
            if self.op == 's': return x_s
            l = self.left.eval(x_p, x_s, params) if self.left else 0.0
            r = self.right.eval(x_p, x_s, params) if self.right else 0.0
            
            # THE SINGULARITY: NO ABS MASKING. NO EPSILON.
            if self.op == '+': return l + r
            if self.op == '*': return l * r
            if self.op == 'exp': return np.exp(-l**2)
            if self.op == 'sin': return np.sin(l)
            if self.op == 'cos': return np.cos(l)
            if self.op == 'log': return np.log(l)
            if self.op == 'sqrt': return np.sqrt(l)
            return 0.0
        except: return np.nan

    def complexity(self) -> int:
        return 1 + (self.left.complexity() if self.left else 0) + (self.right.complexity() if self.right else 0)

    def mutate(self):
        ops = ['+', '*', 'exp', 'sin', 'cos', 'log', 'sqrt', 'p', 's', 'c']
        if np.random.rand() < 0.3:
             return VoidNode(np.random.choice(ops), p_idx=np.random.randint(0, 3))
        if self.left: self.left = self.left.mutate()
        if self.right: self.right = self.right.mutate()
        return self

class SymbolicSearchAgent(BaseAgent):
    """
    Asymptotic Emergence Phase 3: Recursive Structural Complexity Synthesis.
    Utilizes open-ended DSL and scale-free information-theoretic discovery.
    """
    def __init__(self, observation_dim: int, action_dim: int, window_size: int = 80):
        super().__init__("symbolic_search", observation_dim, action_dim)
        self.window_size = window_size
        self.obs_history: List[np.ndarray] = []
        self.models: Dict[int, Tuple[VoidNode, np.ndarray, int, float]] = {}

    def propose(self, observation: torch.Tensor) -> AgentProposal:
        if self.is_suppressed: return AgentProposal(self.name, torch.zeros(self.action_dim), 0.0)
        
        # ASYMPTOTIC: Robust Perceptual Buffer Management (D-invariant)
        obs_numpy = observation.detach().cpu().numpy()
        if obs_numpy.shape[0] != self.observation_dim:
            # Resync to agent's fixed sensory bottleneck
            new_obs = np.zeros(self.observation_dim)
            min_d = min(self.observation_dim, obs_numpy.shape[0])
            new_obs[:min_d] = obs_numpy[:min_d]
            obs_numpy = new_obs
            
        self.obs_history.append(obs_numpy)
        if len(self.obs_history) > self.window_size: self.obs_history.pop(0)

        if len(self.obs_history) >= 60 and len(self.obs_history) % 20 == 0:
            self._void_discovery()

        if self.models:
            preds = np.zeros(self.action_dim)
            for d, (node, params, d_s, _) in self.models.items():
                if d < self.action_dim and d_s < self.observation_dim:
                    x_p, x_s = self.obs_history[-1][d], self.obs_history[-1][d_s]
                    p_val = node.eval(x_p, x_s, params)
                    preds[d] = p_val if np.isfinite(p_val) else self.obs_history[-1][d]
            return AgentProposal(self.name, torch.tensor(preds, dtype=torch.float32), 0.99)
        return AgentProposal(self.name, torch.tensor(obs_numpy, dtype=torch.float32), 0.1)

    def _ksg_mi_scale_free(self, x, y):
        """Scale-Free MI: Infimum across adaptive k-ranges."""
        n = x.shape[0]
        x_m, y_m = x.reshape(-1, 1), y.reshape(-1, 1)
        xy = np.hstack([x_m, y_m])
        tree_xy = cKDTree(xy); tree_x = cKDTree(x_m); tree_y = cKDTree(y_m)
        
        k_list = [2, 3, 4, 6]
        mi_list = []
        for k in k_list:
            d_k = tree_xy.query(xy, k=k+1)[0][:, k]
            # THE SINGULARITY: NO 1e-15 safety padding.
            nx = tree_x.query_ball_point(x_m, d_k, return_length=True) - 1
            ny = tree_y.query_ball_point(y_m, d_k, return_length=True) - 1
            mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
            mi_list.append(mi)
        return max(0, min(mi_list))

    def _void_discovery(self):
        history = np.array(self.obs_history)
        T, D = history.shape
        
        for d_t in range(D):
            y_tgt = history[1:, d_t]
            x_p = history[:-1, d_t]
            best_score = float('inf')
            
            mi_scores = [self._ksg_mi_scale_free(x_p, history[:-1, i]) if i != d_t else -1.0 for i in range(D)]
            d_s = np.argmax(mi_scores)
            x_s = history[:-1, d_s]
            
            # ASYMPTOTIC: Dynamic Structural Complexity Discovery using MDL.
            population = [self._generate_random_tree(np.random.randint(2, 5)) for _ in range(25)]
            for gen in range(8):
                for ind in population:
                    num_p = self._get_num_params(ind)
                    # ASYMPTOTIC: Unconstrained Parameter Exploration.
                    pop_de = np.random.normal(0, 1000, (15, num_p)) 
                    for _ in range(5):
                        for i in range(15):
                            r1, r2, r3 = np.random.choice(15, 3, replace=False)
                            mutant = pop_de[r1] + 1.2 * (pop_de[r2] - pop_de[r3])
                            def fit(p):
                                try:
                                    y_pred = np.array([ind.eval(x_p[t], x_s[t], p) for t in range(T-1)], dtype=np.float64)
                                    rss = np.sum((y_tgt - y_pred)**2)
                                    # ASYMPTOTIC: Minimal Description Length (MDL) Objective
                                    # Prioritizes structural parsimony and predictive stability.
                                    return (T-1) * np.log(rss / (T-1) + 1e-15) + ind.complexity() * np.log(T-1)
                                except: return 1e18
                            
                            if fit(mutant) < fit(pop_de[i]): pop_de[i] = mutant
                    
                    best_idx = np.argmin([fit(p) for p in pop_de])
                    score = fit(pop_de[best_idx])
                    if score < best_score:
                        best_score = score
                        self.models[d_t] = (ind, pop_de[best_idx], d_s, score)
                
                # ASYMPTOTIC: Recursive Structural Evolution
                population = [ind.mutate() for ind in population]

    def _generate_random_tree(self, depth: int) -> VoidNode:
        # THE OMEGA: Depth is more of a suggestion now. 
        if depth <= 0 and np.random.rand() < 0.3:
            return VoidNode(np.random.choice(['p', 's', 'c']), p_idx=np.random.randint(0, 3))
        op = np.random.choice(['+', '*', 'exp', 'sin', 'cos', 'log', 'sqrt'])
        if op in ['exp', 'sin', 'cos', 'log', 'sqrt']:
             return VoidNode(op, self._generate_random_tree(depth-1))
        return VoidNode(op, self._generate_random_tree(depth-1), self._generate_random_tree(depth-1))

    def _get_num_params(self, node: VoidNode) -> int:
         if node is None: return 0
         p = (node.p_idx + 1) if node.op == 'c' else 0
         return max(p, self._get_num_params(node.left), self._get_num_params(node.right))

    def get_capacity_metrics(self) -> Dict[str, Any]:
        return {"agent": self.name, "mi": "rigorous_ksg", "search": "evolutionary"}
