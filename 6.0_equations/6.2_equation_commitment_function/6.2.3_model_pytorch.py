"""
6.2.3_model_pytorch.py

Purpose: Differentiable PyTorch implementation of the Commitment Function
Author: Funnel Function Institute
Created: 2025-12-11

The Commitment Equation (Differentiable):
    f(Commitment) = P_Transactional × P_Enduring

This implementation enables:
    - Gradient-based optimization of loyalty drivers
    - Time-series modeling of commitment decay
    - Neural network integration for churn prediction
    - Inverse problem solving (what changes maximize commitment?)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class SoulAlignment(nn.Module):
    """
    Differentiable Soul Alignment calculation.

    S(τ) = ω_Π · Π(Archetype) + ω_ι · ι(Identity)
    """

    def __init__(self, archetype_weight: float = 0.5):
        """
        Initialize Soul Alignment module.

        Args:
            archetype_weight: Weight for archetype resonance (ω_Π)
        """
        super().__init__()
        self.archetype_weight = archetype_weight

    def forward(
        self,
        archetype_resonance: torch.Tensor,
        identity_utility: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soul alignment.

        Args:
            archetype_resonance: Π(Archetype) [0, 1]
            identity_utility: ι(Identity) [0, 1]

        Returns:
            Soul alignment score S(τ)
        """
        identity_weight = 1.0 - self.archetype_weight
        return (self.archetype_weight * archetype_resonance +
                identity_weight * identity_utility)


class PostPurchaseFriction(nn.Module):
    """
    Differentiable Post-Purchase Friction calculation.

    Σ_post = L_service + F_unforeseen + Θ_support
    """

    def forward(
        self,
        service_latency: torch.Tensor,
        unforeseen_costs: torch.Tensor,
        support_friction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute post-purchase friction.

        Args:
            service_latency: L_service
            unforeseen_costs: F_unforeseen
            support_friction: Θ_support

        Returns:
            Total post-purchase friction Σ_post
        """
        return service_latency + unforeseen_costs + support_friction


class EnduringLoyalty(nn.Module):
    """
    Differentiable Enduring Loyalty calculation.

    P_instant = σ( S / (Σ_post)^k − β )
    """

    def __init__(
        self,
        sensitivity_k: float = 1.0,
        baseline_beta: float = 0.5,
        epsilon: float = 1e-8
    ):
        """
        Initialize Enduring Loyalty module.

        Args:
            sensitivity_k: Sensitivity exponent (k ≥ 1)
            baseline_beta: Threshold for positive loyalty
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.sensitivity_k = sensitivity_k
        self.baseline_beta = baseline_beta
        self.epsilon = epsilon

    def forward(
        self,
        soul: torch.Tensor,
        post_friction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute instantaneous enduring loyalty probability.

        Args:
            soul: Soul alignment S(τ)
            post_friction: Post-purchase friction Σ_post

        Returns:
            Enduring loyalty probability P_Enduring
        """
        # Ensure numerical stability
        friction_term = torch.clamp(post_friction, min=self.epsilon) ** self.sensitivity_k

        # Core calculation
        x = (soul / friction_term) - self.baseline_beta

        # Sigmoid activation
        return torch.sigmoid(x)


class CommitmentFunction(nn.Module):
    """
    Full Differentiable Commitment Function.

    f(Commitment) = P_Transactional × P_Enduring

    Where P_Enduring = σ( S(τ) / (Σ_post)^k − β )
    """

    def __init__(
        self,
        archetype_weight: float = 0.5,
        sensitivity_k: float = 1.0,
        baseline_beta: float = 0.5,
        epsilon: float = 1e-8
    ):
        """
        Initialize Commitment Function.

        Args:
            archetype_weight: Weight for archetype vs identity (ω_Π)
            sensitivity_k: Friction sensitivity exponent
            baseline_beta: Baseline loyalty threshold
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.soul_alignment = SoulAlignment(archetype_weight)
        self.post_friction = PostPurchaseFriction()
        self.enduring_loyalty = EnduringLoyalty(sensitivity_k, baseline_beta, epsilon)

    def forward(
        self,
        p_transactional: torch.Tensor,
        archetype_resonance: torch.Tensor,
        identity_utility: torch.Tensor,
        service_latency: torch.Tensor,
        unforeseen_costs: torch.Tensor,
        support_friction: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute commitment score.

        Args:
            p_transactional: Initial purchase probability [0, 1]
            archetype_resonance: Π(Archetype) [0, 1]
            identity_utility: ι(Identity) [0, 1]
            service_latency: L_service [0, ∞)
            unforeseen_costs: F_unforeseen [0, ∞)
            support_friction: Θ_support [0, ∞)

        Returns:
            Commitment score [0, 1]
        """
        # Soul alignment
        soul = self.soul_alignment(archetype_resonance, identity_utility)

        # Post-purchase friction
        friction = self.post_friction(service_latency, unforeseen_costs, support_friction)

        # Enduring loyalty
        p_enduring = self.enduring_loyalty(soul, friction)

        # Final commitment
        commitment = p_transactional * p_enduring

        return commitment

    def forward_detailed(
        self,
        p_transactional: torch.Tensor,
        archetype_resonance: torch.Tensor,
        identity_utility: torch.Tensor,
        service_latency: torch.Tensor,
        unforeseen_costs: torch.Tensor,
        support_friction: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute commitment with full breakdown.

        Returns:
            Dictionary with all intermediate values
        """
        soul = self.soul_alignment(archetype_resonance, identity_utility)
        friction = self.post_friction(service_latency, unforeseen_costs, support_friction)
        p_enduring = self.enduring_loyalty(soul, friction)
        commitment = p_transactional * p_enduring

        return {
            "commitment": commitment,
            "p_transactional": p_transactional,
            "p_enduring": p_enduring,
            "soul_alignment": soul,
            "post_friction": friction
        }


class LearnableCommitmentFunction(nn.Module):
    """
    Commitment Function with learnable parameters.

    Learns optimal weights for:
        - Archetype vs Identity balance
        - Friction sensitivity
        - Baseline threshold

    Use cases:
        - Fit to historical churn data
        - Learn domain-specific loyalty drivers
        - Customer segment analysis
    """

    def __init__(self):
        """Initialize with learnable parameters."""
        super().__init__()

        # Learnable archetype weight (sigmoid-constrained to [0, 1])
        self.archetype_weight_logit = nn.Parameter(torch.tensor(0.0))

        # Learnable sensitivity (softplus to ensure k ≥ 1)
        self.sensitivity_k_raw = nn.Parameter(torch.tensor(0.0))

        # Learnable baseline (no constraint)
        self.baseline_beta = nn.Parameter(torch.tensor(0.5))

        self.epsilon = 1e-8

    def forward(
        self,
        p_transactional: torch.Tensor,
        archetype_resonance: torch.Tensor,
        identity_utility: torch.Tensor,
        service_latency: torch.Tensor,
        unforeseen_costs: torch.Tensor,
        support_friction: torch.Tensor
    ) -> torch.Tensor:
        """Compute commitment with learned parameters."""
        # Get constrained parameters
        archetype_weight = torch.sigmoid(self.archetype_weight_logit)
        sensitivity_k = 1.0 + torch.nn.functional.softplus(self.sensitivity_k_raw)

        # Soul alignment
        identity_weight = 1.0 - archetype_weight
        soul = archetype_weight * archetype_resonance + identity_weight * identity_utility

        # Post-purchase friction
        friction = service_latency + unforeseen_costs + support_friction

        # Enduring loyalty
        friction_term = torch.clamp(friction, min=self.epsilon) ** sensitivity_k
        x = (soul / friction_term) - self.baseline_beta
        p_enduring = torch.sigmoid(x)

        # Commitment
        return p_transactional * p_enduring

    def get_parameters(self) -> Dict[str, float]:
        """Get current learned parameter values."""
        return {
            "archetype_weight": torch.sigmoid(self.archetype_weight_logit).item(),
            "sensitivity_k": (1.0 + torch.nn.functional.softplus(self.sensitivity_k_raw)).item(),
            "baseline_beta": self.baseline_beta.item()
        }


class CommitmentTimeSeries(nn.Module):
    """
    Time-series commitment model with decay dynamics.

    Models how commitment evolves over time as:
        - Soul alignment decays (novelty wears off)
        - Friction accumulates (issues compound)
        - External shocks occur (competitors, scandals)
    """

    def __init__(
        self,
        soul_decay_rate: float = 0.02,
        friction_growth_rate: float = 0.05,
        sensitivity_k: float = 1.0,
        baseline_beta: float = 0.5
    ):
        """
        Initialize time-series model.

        Args:
            soul_decay_rate: Rate of soul alignment decay per time step
            friction_growth_rate: Rate of friction accumulation per time step
            sensitivity_k: Friction sensitivity exponent
            baseline_beta: Baseline loyalty threshold
        """
        super().__init__()
        self.soul_decay_rate = soul_decay_rate
        self.friction_growth_rate = friction_growth_rate
        self.commitment_fn = CommitmentFunction(
            sensitivity_k=sensitivity_k,
            baseline_beta=baseline_beta
        )

    def forward(
        self,
        initial_p_trans: torch.Tensor,
        initial_archetype: torch.Tensor,
        initial_identity: torch.Tensor,
        initial_service: torch.Tensor,
        initial_unforeseen: torch.Tensor,
        initial_support: torch.Tensor,
        n_steps: int
    ) -> torch.Tensor:
        """
        Simulate commitment trajectory over time.

        Args:
            initial_*: Initial values for each variable
            n_steps: Number of time steps to simulate

        Returns:
            Tensor of commitment values at each time step (shape: n_steps)
        """
        commitments = []

        # Current state
        archetype = initial_archetype.clone()
        identity = initial_identity.clone()
        service = initial_service.clone()
        unforeseen = initial_unforeseen.clone()
        support = initial_support.clone()

        for t in range(n_steps):
            # Calculate commitment at this time step
            c = self.commitment_fn(
                initial_p_trans, archetype, identity,
                service, unforeseen, support
            )
            commitments.append(c)

            # Apply decay/growth dynamics
            archetype = archetype * (1 - self.soul_decay_rate)
            identity = identity * (1 - self.soul_decay_rate * 0.5)  # Identity decays slower
            service = service + self.friction_growth_rate * 0.3
            unforeseen = unforeseen + self.friction_growth_rate * 0.5
            support = support + self.friction_growth_rate * 0.2

        return torch.stack(commitments)


class CommitmentOptimizer:
    """
    Utility class for optimizing commitment drivers.

    Given a target commitment level, find what changes would achieve it.
    """

    def __init__(self, commitment_fn: Optional[CommitmentFunction] = None):
        """Initialize with a Commitment Function instance."""
        self.commitment_fn = commitment_fn or CommitmentFunction()

    def optimize_for_retention(
        self,
        current_p_trans: float,
        current_archetype: float,
        current_identity: float,
        current_service: float,
        current_unforeseen: float,
        current_support: float,
        target_commitment: float = 0.7,
        n_steps: int = 500,
        lr: float = 0.05
    ) -> Dict[str, float]:
        """
        Find minimal changes to achieve target commitment.

        Optimizes friction reduction (easier) before soul enhancement (harder).

        Returns:
            Dictionary with recommended changes
        """
        # Create optimizable variables for friction (start from current)
        service_delta = nn.Parameter(torch.tensor(0.0))
        unforeseen_delta = nn.Parameter(torch.tensor(0.0))
        support_delta = nn.Parameter(torch.tensor(0.0))

        optimizer = torch.optim.Adam([service_delta, unforeseen_delta, support_delta], lr=lr)
        target = torch.tensor(target_commitment)

        for _ in range(n_steps):
            optimizer.zero_grad()

            # Apply changes (ReLU to prevent increasing friction)
            new_service = torch.tensor(current_service) - torch.relu(service_delta)
            new_unforeseen = torch.tensor(current_unforeseen) - torch.relu(unforeseen_delta)
            new_support = torch.tensor(current_support) - torch.relu(support_delta)

            # Clamp to non-negative
            new_service = torch.clamp(new_service, min=0.01)
            new_unforeseen = torch.clamp(new_unforeseen, min=0.01)
            new_support = torch.clamp(new_support, min=0.01)

            commitment = self.commitment_fn(
                torch.tensor(current_p_trans),
                torch.tensor(current_archetype),
                torch.tensor(current_identity),
                new_service, new_unforeseen, new_support
            )

            # Loss = distance from target + small penalty for large changes
            loss = (commitment - target) ** 2 + 0.01 * (
                service_delta ** 2 + unforeseen_delta ** 2 + support_delta ** 2
            )

            loss.backward()
            optimizer.step()

        return {
            "current_commitment": self.commitment_fn(
                torch.tensor(current_p_trans),
                torch.tensor(current_archetype),
                torch.tensor(current_identity),
                torch.tensor(current_service),
                torch.tensor(current_unforeseen),
                torch.tensor(current_support)
            ).item(),
            "target_commitment": target_commitment,
            "achieved_commitment": commitment.item(),
            "reduce_service_latency_by": max(0, torch.relu(service_delta).item()),
            "reduce_unforeseen_costs_by": max(0, torch.relu(unforeseen_delta).item()),
            "reduce_support_friction_by": max(0, torch.relu(support_delta).item())
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Commitment Function Demo")
    print("=" * 60)

    cf = CommitmentFunction()

    # Example: High-commitment brand
    print("\n1. High-Commitment Brand (Mint Mobile-like):")
    result = cf.forward_detailed(
        p_transactional=torch.tensor(0.85),
        archetype_resonance=torch.tensor(0.95),
        identity_utility=torch.tensor(0.90),
        service_latency=torch.tensor(0.1),
        unforeseen_costs=torch.tensor(0.05),
        support_friction=torch.tensor(0.15)
    )
    print(f"   Soul Alignment: {result['soul_alignment'].item():.3f}")
    print(f"   Post Friction: {result['post_friction'].item():.3f}")
    print(f"   P_Enduring: {result['p_enduring'].item():.3f}")
    print(f"   Commitment: {result['commitment'].item():.3f}")

    # Example: Low-commitment brand
    print("\n2. Low-Commitment Brand (Temu-like):")
    result = cf.forward_detailed(
        p_transactional=torch.tensor(0.70),
        archetype_resonance=torch.tensor(0.20),
        identity_utility=torch.tensor(0.15),
        service_latency=torch.tensor(0.60),
        unforeseen_costs=torch.tensor(0.40),
        support_friction=torch.tensor(0.70)
    )
    print(f"   Soul Alignment: {result['soul_alignment'].item():.3f}")
    print(f"   Post Friction: {result['post_friction'].item():.3f}")
    print(f"   P_Enduring: {result['p_enduring'].item():.3f}")
    print(f"   Commitment: {result['commitment'].item():.3f}")

    # Time series simulation
    print("\n3. Commitment Decay Over 12 Months:")
    ts_model = CommitmentTimeSeries()
    trajectory = ts_model(
        initial_p_trans=torch.tensor(0.80),
        initial_archetype=torch.tensor(0.85),
        initial_identity=torch.tensor(0.80),
        initial_service=torch.tensor(0.15),
        initial_unforeseen=torch.tensor(0.10),
        initial_support=torch.tensor(0.20),
        n_steps=12
    )
    print(f"   Month 1: {trajectory[0].item():.3f}")
    print(f"   Month 6: {trajectory[5].item():.3f}")
    print(f"   Month 12: {trajectory[11].item():.3f}")
    print(f"   Decay: {((trajectory[0] - trajectory[11]) / trajectory[0] * 100).item():.1f}%")

    # Optimization
    print("\n4. Optimize for Retention (target 70%):")
    optimizer = CommitmentOptimizer(cf)
    plan = optimizer.optimize_for_retention(
        current_p_trans=0.70,
        current_archetype=0.20,
        current_identity=0.15,
        current_service=0.60,
        current_unforeseen=0.40,
        current_support=0.70,
        target_commitment=0.50  # Realistic target for troubled brand
    )
    print(f"   Current: {plan['current_commitment']:.3f}")
    print(f"   Target: {plan['target_commitment']:.3f}")
    print(f"   Achievable: {plan['achieved_commitment']:.3f}")
    print(f"   Recommendations:")
    print(f"     - Reduce service latency by: {plan['reduce_service_latency_by']:.2f}")
    print(f"     - Reduce unforeseen costs by: {plan['reduce_unforeseen_costs_by']:.2f}")
    print(f"     - Reduce support friction by: {plan['reduce_support_friction_by']:.2f}")
