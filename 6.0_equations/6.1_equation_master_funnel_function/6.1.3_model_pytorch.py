"""
6.1.3_model_pytorch.py

Purpose: Differentiable PyTorch implementation of the Master Funnel Function
Author: Funnel Function Institute
Created: 2025-12-11

The Master Equation (Differentiable):
    f(x) = (B · M · S) / (N + L + Θ + ε) × W

This implementation enables:
    - Gradient-based optimization of marketing variables
    - Inverse problem solving (given target attention, find inputs)
    - Neural network integration for attention prediction
    - Batch processing of multiple scenarios
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class FunnelFunction(nn.Module):
    """
    Differentiable Funnel Function as a PyTorch Module.

    f(x) = (B · M · S) / (N + L + Θ + ε) × W

    All operations are differentiable, enabling gradient-based optimization.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize the Funnel Function.

        Args:
            epsilon: Small constant to prevent division by zero
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(
        self,
        body: torch.Tensor,
        mind: torch.Tensor,
        soul: torch.Tensor,
        noise: torch.Tensor,
        load: torch.Tensor,
        friction: torch.Tensor,
        writability: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the Funnel Function.

        Args:
            body: Somatic certainty [0, 1] - shape (batch_size,) or scalar
            mind: Prediction confidence [0, 1]
            soul: Identity congruence [0, 1]
            noise: Environmental interference [0, ∞)
            load: Cognitive burden [0, ∞)
            friction: Barriers to action [0, ∞)
            writability: Receptivity gate [0, 1] (default: 1.0)

        Returns:
            Gated attention value
        """
        # Default writability to 1.0 if not provided
        if writability is None:
            writability = torch.ones_like(body)

        # Signal = B × M × S (multiplicative)
        signal = body * mind * soul

        # Suppression = N + L + Θ + ε (additive)
        suppression = noise + load + friction + self.epsilon

        # Raw attention
        attention = signal / suppression

        # Apply writability gate
        gated_attention = attention * writability

        return gated_attention

    def forward_detailed(
        self,
        body: torch.Tensor,
        mind: torch.Tensor,
        soul: torch.Tensor,
        noise: torch.Tensor,
        load: torch.Tensor,
        friction: torch.Tensor,
        writability: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Funnel Function with detailed breakdown.

        Returns:
            Dictionary with signal, suppression, attention, and gated_attention
        """
        if writability is None:
            writability = torch.ones_like(body)

        signal = body * mind * soul
        suppression = noise + load + friction + self.epsilon
        attention = signal / suppression
        gated_attention = attention * writability

        return {
            "signal": signal,
            "suppression": suppression,
            "attention": attention,
            "gated_attention": gated_attention
        }


class LearnableFunnelFunction(nn.Module):
    """
    Funnel Function with learnable channel weights.

    Extends the base equation to learn optimal weighting:
        f(x) = (w_b·B · w_m·M · w_s·S) / (w_n·N + w_l·L + w_f·Θ + ε) × W

    Use cases:
        - Fit to historical conversion data
        - Learn domain-specific channel importance
        - A/B test analysis
    """

    def __init__(
        self,
        init_signal_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        init_suppression_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        epsilon: float = 1e-8
    ):
        """
        Initialize with optional starting weights.

        Args:
            init_signal_weights: Initial (body, mind, soul) weights
            init_suppression_weights: Initial (noise, load, friction) weights
            epsilon: Small constant for numerical stability
        """
        super().__init__()

        # Learnable signal weights
        self.w_body = nn.Parameter(torch.tensor(init_signal_weights[0]))
        self.w_mind = nn.Parameter(torch.tensor(init_signal_weights[1]))
        self.w_soul = nn.Parameter(torch.tensor(init_signal_weights[2]))

        # Learnable suppression weights
        self.w_noise = nn.Parameter(torch.tensor(init_suppression_weights[0]))
        self.w_load = nn.Parameter(torch.tensor(init_suppression_weights[1]))
        self.w_friction = nn.Parameter(torch.tensor(init_suppression_weights[2]))

        self.epsilon = epsilon

    def forward(
        self,
        body: torch.Tensor,
        mind: torch.Tensor,
        soul: torch.Tensor,
        noise: torch.Tensor,
        load: torch.Tensor,
        friction: torch.Tensor,
        writability: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted Funnel Function."""
        if writability is None:
            writability = torch.ones_like(body)

        # Weighted signal (use softplus to ensure positive weights)
        w_b = torch.nn.functional.softplus(self.w_body)
        w_m = torch.nn.functional.softplus(self.w_mind)
        w_s = torch.nn.functional.softplus(self.w_soul)

        signal = (w_b * body) * (w_m * mind) * (w_s * soul)

        # Weighted suppression
        w_n = torch.nn.functional.softplus(self.w_noise)
        w_l = torch.nn.functional.softplus(self.w_load)
        w_f = torch.nn.functional.softplus(self.w_friction)

        suppression = (w_n * noise) + (w_l * load) + (w_f * friction) + self.epsilon

        # Compute attention
        attention = signal / suppression
        gated_attention = attention * writability

        return gated_attention

    def get_weights(self) -> Dict[str, float]:
        """Get current weight values (after softplus activation)."""
        return {
            "body": torch.nn.functional.softplus(self.w_body).item(),
            "mind": torch.nn.functional.softplus(self.w_mind).item(),
            "soul": torch.nn.functional.softplus(self.w_soul).item(),
            "noise": torch.nn.functional.softplus(self.w_noise).item(),
            "load": torch.nn.functional.softplus(self.w_load).item(),
            "friction": torch.nn.functional.softplus(self.w_friction).item(),
        }


class AttentionOptimizer:
    """
    Utility class for optimizing marketing variables.

    Given constraints and a target attention score, find optimal input values.
    """

    def __init__(self, funnel_fn: Optional[FunnelFunction] = None):
        """Initialize with a Funnel Function instance."""
        self.funnel_fn = funnel_fn or FunnelFunction()

    def optimize_signal(
        self,
        target_attention: float,
        fixed_noise: float = 0.2,
        fixed_load: float = 0.2,
        fixed_friction: float = 0.2,
        writability: float = 1.0,
        n_steps: int = 1000,
        lr: float = 0.01
    ) -> Dict[str, float]:
        """
        Find optimal Body, Mind, Soul values to achieve target attention.

        Args:
            target_attention: Desired attention score
            fixed_noise/load/friction: Suppression values (held constant)
            writability: Receptivity gate
            n_steps: Optimization iterations
            lr: Learning rate

        Returns:
            Dictionary with optimal body, mind, soul values
        """
        # Initialize variables to optimize (use sigmoid to constrain to [0, 1])
        body_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        mind_logit = nn.Parameter(torch.tensor(0.0))
        soul_logit = nn.Parameter(torch.tensor(0.0))

        optimizer = torch.optim.Adam([body_logit, mind_logit, soul_logit], lr=lr)
        target = torch.tensor(target_attention)

        # Fixed tensors
        noise = torch.tensor(fixed_noise)
        load = torch.tensor(fixed_load)
        friction = torch.tensor(fixed_friction)
        w = torch.tensor(writability)

        for _ in range(n_steps):
            optimizer.zero_grad()

            # Convert logits to [0, 1] range
            body = torch.sigmoid(body_logit)
            mind = torch.sigmoid(mind_logit)
            soul = torch.sigmoid(soul_logit)

            # Compute attention
            attention = self.funnel_fn(body, mind, soul, noise, load, friction, w)

            # MSE loss
            loss = (attention - target) ** 2

            loss.backward()
            optimizer.step()

        return {
            "body": torch.sigmoid(body_logit).item(),
            "mind": torch.sigmoid(mind_logit).item(),
            "soul": torch.sigmoid(soul_logit).item(),
            "achieved_attention": attention.item(),
            "target_attention": target_attention
        }

    def sensitivity_analysis(
        self,
        body: float = 0.7,
        mind: float = 0.7,
        soul: float = 0.7,
        noise: float = 0.2,
        load: float = 0.2,
        friction: float = 0.2,
        writability: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute gradients to understand sensitivity to each variable.

        Returns:
            Dictionary of variable names to their gradient magnitudes
        """
        # Create tensors that require gradients
        b = torch.tensor(body, requires_grad=True)
        m = torch.tensor(mind, requires_grad=True)
        s = torch.tensor(soul, requires_grad=True)
        n = torch.tensor(noise, requires_grad=True)
        l = torch.tensor(load, requires_grad=True)
        f = torch.tensor(friction, requires_grad=True)
        w = torch.tensor(writability, requires_grad=True)

        # Compute attention
        attention = self.funnel_fn(b, m, s, n, l, f, w)

        # Compute gradients
        attention.backward()

        return {
            "body_sensitivity": abs(b.grad.item()),
            "mind_sensitivity": abs(m.grad.item()),
            "soul_sensitivity": abs(s.grad.item()),
            "noise_sensitivity": abs(n.grad.item()),
            "load_sensitivity": abs(l.grad.item()),
            "friction_sensitivity": abs(f.grad.item()),
            "writability_sensitivity": abs(w.grad.item()),
            "attention": attention.item()
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Funnel Function Demo")
    print("=" * 60)

    # Basic usage
    ff = FunnelFunction()

    body = torch.tensor(0.8)
    mind = torch.tensor(0.7)
    soul = torch.tensor(0.9)
    noise = torch.tensor(0.2)
    load = torch.tensor(0.3)
    friction = torch.tensor(0.1)

    attention = ff(body, mind, soul, noise, load, friction)
    print(f"\n1. Basic Calculation:")
    print(f"   Inputs: B={body:.1f}, M={mind:.1f}, S={soul:.1f}")
    print(f"   Suppression: N={noise:.1f}, L={load:.1f}, Θ={friction:.1f}")
    print(f"   Attention: {attention.item():.4f}")

    # Batch processing
    print(f"\n2. Batch Processing (3 scenarios):")
    batch_body = torch.tensor([0.8, 0.5, 0.9])
    batch_mind = torch.tensor([0.7, 0.8, 0.6])
    batch_soul = torch.tensor([0.9, 0.6, 0.7])
    batch_noise = torch.tensor([0.2, 0.4, 0.1])
    batch_load = torch.tensor([0.3, 0.2, 0.2])
    batch_friction = torch.tensor([0.1, 0.3, 0.1])

    batch_attention = ff(batch_body, batch_mind, batch_soul,
                         batch_noise, batch_load, batch_friction)
    for i, att in enumerate(batch_attention):
        print(f"   Scenario {i+1}: Attention = {att.item():.4f}")

    # Sensitivity analysis
    print(f"\n3. Sensitivity Analysis:")
    optimizer = AttentionOptimizer(ff)
    sensitivity = optimizer.sensitivity_analysis()
    print(f"   Current Attention: {sensitivity['attention']:.4f}")
    print(f"   Most sensitive to:")
    sensitivities = {k: v for k, v in sensitivity.items() if k.endswith('_sensitivity')}
    sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_sens[:3]:
        print(f"     - {name.replace('_sensitivity', '')}: {value:.4f}")

    # Optimization
    print(f"\n4. Optimize Signal for Target Attention:")
    result = optimizer.optimize_signal(
        target_attention=0.8,
        fixed_noise=0.2,
        fixed_load=0.2,
        fixed_friction=0.2
    )
    print(f"   Target: {result['target_attention']:.2f}")
    print(f"   Achieved: {result['achieved_attention']:.4f}")
    print(f"   Optimal B={result['body']:.3f}, M={result['mind']:.3f}, S={result['soul']:.3f}")

    # Learnable model
    print(f"\n5. Learnable Model (fitting to data):")
    learnable = LearnableFunnelFunction()
    print(f"   Initial weights: {learnable.get_weights()}")
