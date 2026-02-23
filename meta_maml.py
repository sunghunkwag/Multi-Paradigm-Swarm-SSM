"""Meta-MAML Module — ported from SSM-MetaRL-TestCompute.

Model-Agnostic Meta-Learning with functional forward passes.
Supports stateful models (MambaSSM) via hidden_state propagation.

Origin: https://github.com/sunghunkwag/SSM-MetaRL-TestCompute/blob/main/meta_rl/meta_maml.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from typing import Dict, List, Tuple, Optional, OrderedDict as OrderedDictType, Union
from collections import OrderedDict

try:
    from torch.func import functional_call
except ImportError:
    functional_call = None


class MetaMAML:
    """Model-Agnostic Meta-Learning (MAML) for the SSM-Mamba Swarm.

    Enables few-shot adaptation of any nn.Module (including stateful MambaSSM).
    Uses torch.func.functional_call for efficient gradient computation.

    Args:
        model: The base model to meta-learn
        inner_lr: Learning rate for task-level adaptation
        outer_lr: Learning rate for meta-optimization
        first_order: Use first-order approximation (faster, less accurate)
    """

    def __init__(self, model: nn.Module, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, first_order: bool = False):
        if functional_call is None:
            raise ImportError("MetaMAML requires torch.func (PyTorch >= 2.0).")
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

        # Detect if model requires hidden state
        sig = inspect.signature(self.model.forward)
        self._stateful = 'hidden_state' in sig.parameters

    def functional_forward(self, x: torch.Tensor,
                           hidden_state: Optional[torch.Tensor],
                           params: Optional[OrderedDictType[str, torch.Tensor]] = None
                           ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with custom parameters (fast_weights)."""
        args = (x, hidden_state) if self._stateful else (x,)
        if params is None:
            return self.model(*args)
        else:
            return functional_call(self.model, params, args)

    def adapt_task(self, support_x: torch.Tensor, support_y: torch.Tensor,
                   initial_hidden_state: Optional[torch.Tensor] = None,
                   loss_fn=None, num_steps: int = 1
                   ) -> OrderedDictType[str, torch.Tensor]:
        """Inner loop: adapt model parameters on support set.

        Returns:
            fast_weights: Adapted parameters as OrderedDict
        """
        if loss_fn is None:
            loss_fn = F.mse_loss

        fast_weights = OrderedDict((name, param.clone())
                                   for name, param in self.model.named_parameters())

        time_dim_present = support_x.ndim == 3

        for step in range(num_steps):
            hidden_state = initial_hidden_state

            if time_dim_present:
                T = support_x.shape[1]
                outputs = []
                for t in range(T):
                    x_t = support_x[:, t, :]
                    if hidden_state is not None and hidden_state.shape[0] == 1 and x_t.shape[0] > 1:
                        hidden_state = hidden_state.expand(x_t.shape[0], *hidden_state.shape[1:])
                    output_t, hidden_state = self.functional_forward(x_t, hidden_state, fast_weights)
                    outputs.append(output_t)
                pred = torch.stack(outputs, dim=1)
                step_loss = loss_fn(pred, support_y)
            else:
                if self._stateful:
                    pred, _ = self.functional_forward(support_x, hidden_state, fast_weights)
                else:
                    pred = self.functional_forward(support_x, None, fast_weights)
                step_loss = loss_fn(pred, support_y)

            grads = torch.autograd.grad(step_loss, fast_weights.values(),
                                        create_graph=not self.first_order)
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )

        return fast_weights

    def meta_update(self, tasks: List[Tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]],
                   initial_hidden_state: Optional[torch.Tensor] = None,
                   loss_fn=None) -> float:
        """Outer loop: meta-update across multiple tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y)
            initial_hidden_state: Initial hidden state for stateful models
            loss_fn: Loss function (default: MSE)

        Returns:
            Average meta-loss across tasks
        """
        if loss_fn is None:
            loss_fn = F.mse_loss

        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            fast_weights = self.adapt_task(support_x, support_y,
                                           initial_hidden_state, loss_fn)

            hidden_state = initial_hidden_state
            time_dim_present = query_x.ndim == 3

            if time_dim_present:
                T = query_x.shape[1]
                outputs = []
                for t in range(T):
                    x_t = query_x[:, t, :]
                    if hidden_state is not None and hidden_state.shape[0] == 1 and x_t.shape[0] > 1:
                        hidden_state = hidden_state.expand(x_t.shape[0], *hidden_state.shape[1:])
                    output_t, hidden_state = self.functional_forward(x_t, hidden_state, fast_weights)
                    outputs.append(output_t)
                pred = torch.stack(outputs, dim=1)
                query_loss = loss_fn(pred, query_y)
            else:
                if self._stateful:
                    pred, _ = self.functional_forward(query_x, hidden_state, fast_weights)
                else:
                    pred = self.functional_forward(query_x, None, fast_weights)
                query_loss = loss_fn(pred, query_y)

            meta_loss += query_loss

        meta_loss = meta_loss / len(tasks)
        meta_loss.backward()
        self.meta_optimizer.step()
        return meta_loss.item()

    def get_fast_weights(self) -> OrderedDictType[str, torch.Tensor]:
        """Get current model parameters as OrderedDict."""
        return OrderedDict((name, param.clone())
                           for name, param in self.model.named_parameters())
