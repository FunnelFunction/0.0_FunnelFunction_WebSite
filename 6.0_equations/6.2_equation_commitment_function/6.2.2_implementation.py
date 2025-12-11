"""
6.2.2_implementation.py

Purpose: Pure Python implementation of the Commitment Function
Author: Funnel Function Institute
Created: 2025-12-11

The Commitment Equation (replaces "Trust"):
    f(Commitment) = P_Transactional × P_Enduring

Where:
    P_Transactional = Probability of initial purchase (from Master Funnel Function)
    P_Enduring = Time-averaged loyalty probability

Enduring Loyalty:
                    1   t₀+T
    P_Enduring(T) = ─── ∫     σ( S(τ) / (Σ_post(τ))^k − β ) dτ
                     T   t₀

Key insight: "Trust" is not a binary state but a continuous function of:
    - Soul alignment over time
    - Post-purchase friction accumulation
    - Time horizon of the relationship
"""

from dataclasses import dataclass
from typing import List, Optional, Callable
import math


@dataclass
class CommitmentInputs:
    """Input parameters for the Commitment Function."""
    # Transactional probability (from Master Funnel Function)
    p_transactional: float  # [0, 1]

    # Soul alignment components
    archetype_resonance: float  # Π(Archetype) [0, 1]
    identity_utility: float  # ι(Identity) [0, 1]
    archetype_weight: float = 0.5  # ω_Π

    # Post-purchase friction components
    service_latency: float = 0.1  # L_service [0, ∞)
    unforeseen_costs: float = 0.1  # F_unforeseen [0, ∞)
    support_friction: float = 0.1  # Θ_support [0, ∞)

    # Model parameters
    sensitivity_k: float = 1.0  # k ≥ 1
    baseline_beta: float = 0.5  # Threshold


@dataclass
class CommitmentResult:
    """Result of Commitment Function calculation."""
    commitment: float  # Final commitment score
    p_transactional: float  # Initial purchase probability
    p_enduring: float  # Long-term loyalty probability
    soul_alignment: float  # S(τ) value
    post_friction: float  # Σ_post value


def sigmoid(x: float) -> float:
    """Standard sigmoid activation."""
    return 1.0 / (1.0 + math.exp(-x))


def soul_alignment(
    archetype_resonance: float,
    identity_utility: float,
    archetype_weight: float = 0.5
) -> float:
    """
    Calculate Soul alignment S(τ).

    S(τ) = ω_Π · Π(Archetype) + ω_ι · ι(Identity)

    Args:
        archetype_resonance: How well brand archetype matches customer's desired archetype
        identity_utility: How much the brand enhances customer's self-concept
        archetype_weight: Weight for archetype (ω_Π), identity weight is 1 - ω_Π

    Returns:
        Soul alignment score [0, 1]
    """
    identity_weight = 1.0 - archetype_weight
    return archetype_weight * archetype_resonance + identity_weight * identity_utility


def post_purchase_friction(
    service_latency: float,
    unforeseen_costs: float,
    support_friction: float
) -> float:
    """
    Calculate post-purchase friction Σ_post.

    Σ_post = L_service + F_unforeseen + Θ_support

    Args:
        service_latency: Time to resolve issues (MTTR-like)
        unforeseen_costs: Hidden fees, unexpected charges
        support_friction: Effort required to get help

    Returns:
        Total post-purchase friction
    """
    return service_latency + unforeseen_costs + support_friction


def enduring_loyalty_instant(
    soul: float,
    post_friction: float,
    sensitivity_k: float = 1.0,
    baseline_beta: float = 0.5,
    epsilon: float = 1e-8
) -> float:
    """
    Calculate instantaneous enduring loyalty probability.

    P_instant = σ( S / (Σ_post)^k − β )

    Args:
        soul: Soul alignment S(τ)
        post_friction: Post-purchase friction Σ_post
        sensitivity_k: Sensitivity exponent (k ≥ 1)
        baseline_beta: Threshold for positive loyalty
        epsilon: Numerical stability constant

    Returns:
        Instantaneous loyalty probability [0, 1]
    """
    # Avoid division by zero
    friction_term = max(post_friction, epsilon) ** sensitivity_k

    # Core calculation
    x = (soul / friction_term) - baseline_beta

    return sigmoid(x)


def enduring_loyalty_integrated(
    soul_trajectory: List[float],
    friction_trajectory: List[float],
    sensitivity_k: float = 1.0,
    baseline_beta: float = 0.5
) -> float:
    """
    Calculate time-integrated enduring loyalty.

                    1   t₀+T
    P_Enduring(T) = ─── ∫     σ( S(τ) / (Σ_post(τ))^k − β ) dτ
                     T   t₀

    Uses trapezoidal integration over discrete time steps.

    Args:
        soul_trajectory: Soul alignment values over time
        friction_trajectory: Post-friction values over time (same length)
        sensitivity_k: Sensitivity exponent
        baseline_beta: Baseline threshold

    Returns:
        Time-averaged enduring loyalty probability
    """
    if len(soul_trajectory) != len(friction_trajectory):
        raise ValueError("Soul and friction trajectories must have same length")

    if len(soul_trajectory) < 2:
        if len(soul_trajectory) == 1:
            return enduring_loyalty_instant(
                soul_trajectory[0], friction_trajectory[0],
                sensitivity_k, baseline_beta
            )
        return 0.0

    # Calculate instantaneous loyalty at each time point
    loyalty_values = [
        enduring_loyalty_instant(s, f, sensitivity_k, baseline_beta)
        for s, f in zip(soul_trajectory, friction_trajectory)
    ]

    # Trapezoidal integration (normalized by time horizon)
    total = sum(
        (loyalty_values[i] + loyalty_values[i + 1]) / 2
        for i in range(len(loyalty_values) - 1)
    )

    return total / (len(loyalty_values) - 1)


def commitment_function(inputs: CommitmentInputs) -> float:
    """
    Calculate the full Commitment Function.

    f(Commitment) = P_Transactional × P_Enduring

    Args:
        inputs: CommitmentInputs dataclass with all parameters

    Returns:
        Commitment score [0, 1]
    """
    # Calculate soul alignment
    soul = soul_alignment(
        inputs.archetype_resonance,
        inputs.identity_utility,
        inputs.archetype_weight
    )

    # Calculate post-purchase friction
    friction = post_purchase_friction(
        inputs.service_latency,
        inputs.unforeseen_costs,
        inputs.support_friction
    )

    # Calculate instantaneous enduring loyalty
    p_enduring = enduring_loyalty_instant(
        soul, friction,
        inputs.sensitivity_k,
        inputs.baseline_beta
    )

    # Final commitment
    commitment = inputs.p_transactional * p_enduring

    return commitment


def commitment_function_detailed(inputs: CommitmentInputs) -> CommitmentResult:
    """
    Calculate Commitment Function with full breakdown.

    Returns:
        CommitmentResult with all intermediate values
    """
    soul = soul_alignment(
        inputs.archetype_resonance,
        inputs.identity_utility,
        inputs.archetype_weight
    )

    friction = post_purchase_friction(
        inputs.service_latency,
        inputs.unforeseen_costs,
        inputs.support_friction
    )

    p_enduring = enduring_loyalty_instant(
        soul, friction,
        inputs.sensitivity_k,
        inputs.baseline_beta
    )

    commitment = inputs.p_transactional * p_enduring

    return CommitmentResult(
        commitment=commitment,
        p_transactional=inputs.p_transactional,
        p_enduring=p_enduring,
        soul_alignment=soul,
        post_friction=friction
    )


def diagnose_commitment(inputs: CommitmentInputs) -> dict:
    """
    Diagnose commitment issues and provide recommendations.

    Returns:
        Dictionary with diagnosis and recommendations
    """
    result = commitment_function_detailed(inputs)

    issues = []
    recommendations = []

    # Check transactional probability
    if result.p_transactional < 0.5:
        issues.append("Low initial conversion probability")
        recommendations.append("Focus on Master Funnel Function optimization (B, M, S channels)")

    # Check soul alignment
    if result.soul_alignment < 0.5:
        if inputs.archetype_resonance < inputs.identity_utility:
            issues.append("Weak archetype resonance")
            recommendations.append("Clarify brand archetype and ensure consistency")
        else:
            issues.append("Low identity utility")
            recommendations.append("Increase status/identity value proposition")

    # Check friction
    if result.post_friction > 0.5:
        friction_breakdown = {
            "service_latency": inputs.service_latency,
            "unforeseen_costs": inputs.unforeseen_costs,
            "support_friction": inputs.support_friction
        }
        worst_friction = max(friction_breakdown, key=friction_breakdown.get)
        issues.append(f"High post-purchase friction ({worst_friction})")
        recommendations.append(f"Reduce {worst_friction} to improve retention")

    # Check enduring loyalty
    if result.p_enduring < 0.5:
        issues.append("Low long-term loyalty probability")
        recommendations.append("Improve soul/friction ratio for sustained engagement")

    return {
        "commitment_score": result.commitment,
        "p_transactional": result.p_transactional,
        "p_enduring": result.p_enduring,
        "soul_alignment": result.soul_alignment,
        "post_friction": result.post_friction,
        "issues": issues,
        "recommendations": recommendations,
        "health": "good" if result.commitment > 0.5 else "needs_attention"
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Commitment Function Demo")
    print("=" * 60)

    # Example: High-performing brand (like Mint Mobile)
    print("\n1. High-Performing Brand (Mint Mobile-like):")
    mint_inputs = CommitmentInputs(
        p_transactional=0.85,  # Strong conversion
        archetype_resonance=0.95,  # Ryan Reynolds = perfect Jester archetype
        identity_utility=0.90,  # "Smart saver" identity
        service_latency=0.1,  # Good service
        unforeseen_costs=0.05,  # Transparent pricing
        support_friction=0.15  # Easy to get help
    )
    mint_result = commitment_function_detailed(mint_inputs)
    print(f"   P_Transactional: {mint_result.p_transactional:.2f}")
    print(f"   P_Enduring: {mint_result.p_enduring:.2f}")
    print(f"   Commitment: {mint_result.commitment:.2f}")

    # Example: Low-trust brand (like Temu)
    print("\n2. Low-Trust Brand (Temu-like):")
    temu_inputs = CommitmentInputs(
        p_transactional=0.70,  # Decent conversion (price appeal)
        archetype_resonance=0.20,  # No clear archetype
        identity_utility=0.15,  # No identity value
        service_latency=0.60,  # Slow shipping/support
        unforeseen_costs=0.40,  # Quality surprises
        support_friction=0.70  # Hard to get help
    )
    temu_result = commitment_function_detailed(temu_inputs)
    print(f"   P_Transactional: {temu_result.p_transactional:.2f}")
    print(f"   P_Enduring: {temu_result.p_enduring:.2f}")
    print(f"   Commitment: {temu_result.commitment:.2f}")

    # Diagnosis
    print("\n3. Diagnosis for Low-Trust Brand:")
    diagnosis = diagnose_commitment(temu_inputs)
    print(f"   Health: {diagnosis['health']}")
    print(f"   Issues:")
    for issue in diagnosis['issues']:
        print(f"     - {issue}")
    print(f"   Recommendations:")
    for rec in diagnosis['recommendations']:
        print(f"     - {rec}")

    # Time trajectory simulation
    print("\n4. Time Trajectory (12 months):")
    # Simulate soul degradation and friction accumulation
    months = 12
    soul_trajectory = [0.8 - (0.02 * i) for i in range(months)]  # Slow soul decay
    friction_trajectory = [0.2 + (0.05 * i) for i in range(months)]  # Friction accumulation

    p_enduring_integrated = enduring_loyalty_integrated(
        soul_trajectory, friction_trajectory
    )
    print(f"   Initial Soul: {soul_trajectory[0]:.2f} → Final: {soul_trajectory[-1]:.2f}")
    print(f"   Initial Friction: {friction_trajectory[0]:.2f} → Final: {friction_trajectory[-1]:.2f}")
    print(f"   Integrated P_Enduring: {p_enduring_integrated:.2f}")
