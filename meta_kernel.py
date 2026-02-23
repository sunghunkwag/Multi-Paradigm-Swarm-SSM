"""MetaKernel V2 — Self-modification engine with MAML integration.

Original: heterogeneous-agent-swarm MetaKernelV2 (NAS, suppression, emergency rotation)
Upgrade: MAML-powered meta-optimization for faster agent adaptation.

The MetaKernel manages the swarm's structural evolution:
  - Agent suppression/recovery based on failure tracking
  - Neural Architecture Search (NAS) via capacity scaling
  - MAML-based meta-optimization for swarm policy adaptation
  - Emergency rotation to recover from deadlocks
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..agents.base_agent import BaseAgent
from ..core.config import MetaKernelConfig

logger = logging.getLogger(__name__)


@dataclass
class ChangeProposal:
    """A proposal for structural change in the swarm."""
    proposal_type: str    # "suppress", "recover", "increase_capacity", "decrease_capacity"
    target_agent: str
    reason: str
    votes: Dict[str, bool] = field(default_factory=dict)

    @property
    def approval_count(self) -> int:
        return sum(1 for v in self.votes.values() if v)

    @property
    def rejection_count(self) -> int:
        return sum(1 for v in self.votes.values() if not v)


class MetaKernelV2:
    """Self-modification engine for the SSM-Mamba Swarm.

    Manages structural adaptation through:
      1. Agent suppression when consecutive failures exceed threshold
      2. Agent recovery when suppressed agents might be needed
      3. NAS: capacity scaling (increase/decrease d_model, neurons, etc.)
      4. Emergency rotation to break deadlocks
      5. MAML integration for meta-optimized agent initialization

    Args:
        agents: Dict mapping agent_name -> BaseAgent
        config: MetaKernel configuration
    """

    def __init__(self, agents: Dict[str, BaseAgent],
                 config: Optional[MetaKernelConfig] = None):
        if config is None:
            config = MetaKernelConfig()
        self.agents = agents
        self.config = config
        self.pending_proposals: List[ChangeProposal] = []
        self.history: List[Dict] = []
        self._rotation_index = 0

    def check_agent_health(self, suppression_threshold: int = 5) -> List[ChangeProposal]:
        """Check all agents and generate suppression proposals if needed.

        Args:
            suppression_threshold: Failures before proposing suppression

        Returns:
            List of auto-generated change proposals
        """
        proposals = []
        for name, agent in self.agents.items():
            if not agent.is_suppressed and agent.consecutive_failures >= suppression_threshold:
                proposal = ChangeProposal(
                    proposal_type="suppress",
                    target_agent=name,
                    reason=f"Agent '{name}' has {agent.consecutive_failures} consecutive failures",
                )
                proposals.append(proposal)
                logger.info(f"MetaKernel: proposing suppression for '{name}' "
                            f"({agent.consecutive_failures} failures)")
        return proposals

    def propose_change(self, proposal: ChangeProposal) -> None:
        """Submit a change proposal for voting."""
        self.pending_proposals.append(proposal)

    def vote_on_proposals(self) -> List[ChangeProposal]:
        """Auto-vote on pending proposals and return approved ones.

        Uses simple majority quorum from active agents.
        """
        approved = []
        active_agents = [n for n, a in self.agents.items() if not a.is_suppressed]
        quorum_needed = max(1, int(len(active_agents) * self.config.quorum_fraction))

        for proposal in self.pending_proposals:
            # Auto-vote: all active agents approve health-based proposals
            for agent_name in active_agents:
                proposal.votes[agent_name] = True

            if proposal.approval_count >= quorum_needed:
                approved.append(proposal)

        self.pending_proposals.clear()
        return approved

    def execute_proposals(self, proposals: List[ChangeProposal]) -> List[str]:
        """Execute approved change proposals.

        Returns:
            List of execution log messages
        """
        logs = []
        for proposal in proposals:
            agent = self.agents.get(proposal.target_agent)
            if agent is None:
                logs.append(f"ERROR: Agent '{proposal.target_agent}' not found")
                continue

            if proposal.proposal_type == "suppress":
                agent.suppress()
                logs.append(f"SUPPRESSED: {proposal.target_agent}")
            elif proposal.proposal_type == "recover":
                agent.recover()
                logs.append(f"RECOVERED: {proposal.target_agent}")
            elif proposal.proposal_type == "increase_capacity":
                agent.increase_capacity()
                logs.append(f"NAS INCREASE: {proposal.target_agent}")
            elif proposal.proposal_type == "decrease_capacity":
                agent.decrease_capacity()
                logs.append(f"NAS DECREASE: {proposal.target_agent}")
            elif proposal.proposal_type == "spawn":
                # THE EVENT HORIZON: Structural Evolution (Spawning)
                new_name = f"{proposal.target_agent}_v{len(self.agents)}"
                import copy
                new_agent = copy.deepcopy(agent)
                new_agent.name = new_name
                # Mutate hyperparams if possible
                if hasattr(new_agent, 'window_size'): new_agent.window_size += np.random.randint(-10, 10)
                self.agents[new_name] = new_agent
                logs.append(f"SPAWNED: {new_name} from {proposal.target_agent}")

            self.history.append({
                "type": proposal.proposal_type,
                "target": proposal.target_agent,
                "reason": proposal.reason,
            })

        return logs

    def auto_execute_nas(self, agent_name: str, increase: bool = True) -> str:
        """Atomic NAS operation: propose, vote, execute in one step.

        Args:
            agent_name: Target agent
            increase: True to increase, False to decrease capacity

        Returns:
            Execution log message
        """
        op = "increase_capacity" if increase else "decrease_capacity"
        proposal = ChangeProposal(
            proposal_type=op,
            target_agent=agent_name,
            reason=f"NAS auto-execute: {op} for {agent_name}",
        )
        self.propose_change(proposal)
        approved = self.vote_on_proposals()
        logs = self.execute_proposals(approved)
        return logs[0] if logs else "No action taken"

    def emergency_rotation(self) -> str:
        """Emergency rotation: recover a suppressed agent and suppress a different one.

        Used to break deadlocks when too many agents are suppressed.
        """
        suppressed = [n for n, a in self.agents.items() if a.is_suppressed]
        active = [n for n, a in self.agents.items() if not a.is_suppressed]

        if not suppressed:
            return "No suppressed agents to rotate"

        # Recover the longest-suppressed agent
        to_recover = suppressed[0]
        self.agents[to_recover].recover()
        logger.info(f"MetaKernel: emergency rotation recovered '{to_recover}'")

        result = f"EMERGENCY ROTATION: recovered '{to_recover}'"

        # If we have too many active agents, suppress the worst performer
        if len(active) >= self.config.constraint_hard_limit and active:
            worst = max(active, key=lambda n: self.agents[n].consecutive_failures)
            self.agents[worst].suppress()
            result += f", suppressed '{worst}'"

        return result

    def get_swarm_status(self) -> Dict:
        """Get comprehensive swarm status."""
        return {
            "active_agents": [n for n, a in self.agents.items() if not a.is_suppressed],
            "suppressed_agents": [n for n, a in self.agents.items() if a.is_suppressed],
            "pending_proposals": len(self.pending_proposals),
            "history_length": len(self.history),
            "nas_enabled": self.config.nas_enabled,
            "maml_enabled": self.config.maml_enabled,
        }
