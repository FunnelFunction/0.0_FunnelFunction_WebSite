"""
6.1.2_implementation.py

Purpose: Pure Python implementation of the Master Funnel Function f(x)
Author: Funnel Function Institute
Created: 2025-12-10

The Master Equation:
    f(x) = (B · M · S) / (N + L + Θ) × W

Where:
    B = Body (somatic certainty)
    M = Mind (prediction confidence)
    S = Soul (identity congruence)
    N = Noise (environmental interference)
    L = Load (cognitive burden)
    Θ = Theta/Friction (barriers to action)
    W = Writability (receptivity gate)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FunnelFunctionInputs:
    """Input parameters for the Funnel Function."""
    body: float  # B: Somatic certainty [0, 1]
    mind: float  # M: Prediction confidence [0, 1]
    soul: float  # S: Identity congruence [0, 1]
    noise: float = 0.1  # N: Environmental noise [0, ∞)
    load: float = 0.1  # L: Cognitive load [0, ∞)
    friction: float = 0.1  # Θ: Friction [0, ∞)
    writability: float = 1.0  # W: Receptivity gate [0, 1]


@dataclass
class FunnelFunctionResult:
    """Result of Funnel Function calculation."""
    attention: float  # Final attention value
    signal: float  # Numerator (B × M × S)
    suppression: float  # Denominator (N + L + Θ)
    gated_attention: float  # attention × writability


def funnel_function(
    body: float,
    mind: float,
    soul: float,
    noise: float = 0.1,
    load: float = 0.1,
    friction: float = 0.1,
    writability: float = 1.0,
    epsilon: float = 1e-8
) -> float:
    """
    Calculate the Master Funnel Function.

    f(x) = (B · M · S) / (N + L + Θ) × W

    Args:
        body: Somatic certainty (σ) - sensory evidence accumulation
        mind: Prediction confidence (π) - expected outcome accuracy
        soul: Identity congruence (Ι) - alignment with self-concept
        noise: Environmental interference - competing signals
        load: Cognitive burden - mental effort required
        friction: Barriers to action - transaction costs
        writability: Receptivity gate - openness to change
        epsilon: Small value to prevent division by zero

    Returns:
        The computed attention value, gated by writability

    Example:
        >>> funnel_function(body=0.8, mind=0.7, soul=0.9,
        ...                 noise=0.2, load=0.3, friction=0.1)
        0.672
    """
    # Validate inputs
    for name, value in [("body", body), ("mind", mind), ("soul", soul), ("writability", writability)]:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in range [0, 1], got {value}")

    for name, value in [("noise", noise), ("load", load), ("friction", friction)]:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    # Calculate signal (multiplicative)
    signal = body * mind * soul

    # Calculate suppression (additive)
    suppression = noise + load + friction + epsilon

    # Calculate raw attention
    attention = signal / suppression

    # Apply writability gate
    gated_attention = attention * writability

    return gated_attention


def funnel_function_detailed(inputs: FunnelFunctionInputs) -> FunnelFunctionResult:
    """
    Calculate the Funnel Function with detailed breakdown.

    Args:
        inputs: FunnelFunctionInputs dataclass

    Returns:
        FunnelFunctionResult with full breakdown
    """
    signal = inputs.body * inputs.mind * inputs.soul
    suppression = inputs.noise + inputs.load + inputs.friction + 1e-8
    attention = signal / suppression
    gated_attention = attention * inputs.writability

    return FunnelFunctionResult(
        attention=attention,
        signal=signal,
        suppression=suppression,
        gated_attention=gated_attention
    )


def diagnose_attention(
    body: float,
    mind: float,
    soul: float,
    noise: float,
    load: float,
    friction: float
) -> dict:
    """
    Diagnose which channel is the bottleneck.

    Returns:
        Dictionary with diagnosis and recommendations
    """
    channels = {"body": body, "mind": mind, "soul": soul}
    suppressors = {"noise": noise, "load": load, "friction": friction}

    # Find weakest channel
    weakest_channel = min(channels, key=channels.get)
    weakest_value = channels[weakest_channel]

    # Find strongest suppressor
    strongest_suppressor = max(suppressors, key=suppressors.get)
    strongest_value = suppressors[strongest_suppressor]

    # Calculate current attention
    attention = funnel_function(body, mind, soul, noise, load, friction)

    return {
        "current_attention": attention,
        "weakest_channel": {
            "name": weakest_channel,
            "value": weakest_value,
            "recommendation": f"Increase {weakest_channel} to improve signal"
        },
        "strongest_suppressor": {
            "name": strongest_suppressor,
            "value": strongest_value,
            "recommendation": f"Reduce {strongest_suppressor} to decrease suppression"
        },
        "signal": body * mind * soul,
        "suppression": noise + load + friction
    }


# Example usage
if __name__ == "__main__":
    # Basic calculation
    result = funnel_function(
        body=0.8,
        mind=0.7,
        soul=0.9,
        noise=0.2,
        load=0.3,
        friction=0.1
    )
    print(f"Attention Score: {result:.3f}")

    # Detailed calculation
    inputs = FunnelFunctionInputs(
        body=0.8,
        mind=0.7,
        soul=0.9,
        noise=0.2,
        load=0.3,
        friction=0.1
    )
    detailed = funnel_function_detailed(inputs)
    print(f"\nDetailed Breakdown:")
    print(f"  Signal (B×M×S): {detailed.signal:.3f}")
    print(f"  Suppression (N+L+Θ): {detailed.suppression:.3f}")
    print(f"  Attention: {detailed.attention:.3f}")

    # Diagnosis
    diagnosis = diagnose_attention(0.8, 0.4, 0.9, 0.2, 0.3, 0.1)
    print(f"\nDiagnosis:")
    print(f"  Weakest channel: {diagnosis['weakest_channel']['name']} ({diagnosis['weakest_channel']['value']})")
    print(f"  Recommendation: {diagnosis['weakest_channel']['recommendation']}")
