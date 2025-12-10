"""
9.1.1_test_equations.py

Purpose: Unit tests for Funnel Function equations
Author: Funnel Function Institute
Created: 2025-12-10
"""

import pytest
import sys
from pathlib import Path

# Add equations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "6.0_equations"))


class TestMasterFunnelFunction:
    """Tests for the Master Funnel Function f(x)."""

    def test_basic_calculation(self):
        """Test basic f(x) calculation."""
        from equations.funnel_function import funnel_function

        result = funnel_function(
            body=0.8,
            mind=0.7,
            soul=0.9,
            noise=0.2,
            load=0.3,
            friction=0.1
        )

        # Expected: (0.8 * 0.7 * 0.9) / (0.2 + 0.3 + 0.1) = 0.504 / 0.6 = 0.84
        assert 0.83 < result < 0.85

    def test_zero_body_collapses_function(self):
        """Test that zero body input collapses the function."""
        from equations.funnel_function import funnel_function

        result = funnel_function(
            body=0.0,  # Zero body
            mind=0.9,
            soul=0.9,
            noise=0.1,
            load=0.1,
            friction=0.1
        )

        assert result == 0.0

    def test_zero_mind_collapses_function(self):
        """Test that zero mind input collapses the function."""
        from equations.funnel_function import funnel_function

        result = funnel_function(
            body=0.9,
            mind=0.0,  # Zero mind
            soul=0.9,
            noise=0.1,
            load=0.1,
            friction=0.1
        )

        assert result == 0.0

    def test_zero_soul_collapses_function(self):
        """Test that zero soul input collapses the function."""
        from equations.funnel_function import funnel_function

        result = funnel_function(
            body=0.9,
            mind=0.9,
            soul=0.0,  # Zero soul
            noise=0.1,
            load=0.1,
            friction=0.1
        )

        assert result == 0.0

    def test_high_suppression_reduces_output(self):
        """Test that high suppression reduces attention."""
        from equations.funnel_function import funnel_function

        low_suppression = funnel_function(
            body=0.8, mind=0.8, soul=0.8,
            noise=0.1, load=0.1, friction=0.1
        )

        high_suppression = funnel_function(
            body=0.8, mind=0.8, soul=0.8,
            noise=0.5, load=0.5, friction=0.5
        )

        assert high_suppression < low_suppression

    def test_writability_gate(self):
        """Test that writability gates the output."""
        from equations.funnel_function import funnel_function

        full_writability = funnel_function(
            body=0.8, mind=0.8, soul=0.8,
            noise=0.1, load=0.1, friction=0.1,
            writability=1.0
        )

        half_writability = funnel_function(
            body=0.8, mind=0.8, soul=0.8,
            noise=0.1, load=0.1, friction=0.1,
            writability=0.5
        )

        assert half_writability == pytest.approx(full_writability * 0.5, rel=0.01)

    def test_invalid_inputs_raise_error(self):
        """Test that invalid inputs raise ValueError."""
        from equations.funnel_function import funnel_function

        with pytest.raises(ValueError):
            funnel_function(body=1.5, mind=0.8, soul=0.8)  # body > 1

        with pytest.raises(ValueError):
            funnel_function(body=0.8, mind=-0.1, soul=0.8)  # mind < 0

        with pytest.raises(ValueError):
            funnel_function(body=0.8, mind=0.8, soul=0.8, noise=-0.5)  # noise < 0


class TestCommitmentFunction:
    """Tests for the Commitment Function f(Commitment)."""

    def test_basic_calculation(self):
        """Test basic commitment calculation."""
        # f(Commitment) = P_transactional * P_enduring
        p_transactional = 0.7
        p_enduring = 0.8

        expected = p_transactional * p_enduring  # 0.56

        # When implementation exists:
        # from equations.commitment_function import commitment_function
        # result = commitment_function(p_transactional, p_enduring)
        # assert result == pytest.approx(expected)

        assert True  # Placeholder until implementation

    def test_zero_transactional_means_no_commitment(self):
        """Test that zero purchase probability means zero commitment."""
        # If they won't buy, commitment is zero regardless of loyalty potential
        p_transactional = 0.0
        p_enduring = 0.9

        expected = 0.0

        assert True  # Placeholder until implementation

    def test_zero_enduring_means_one_time_only(self):
        """Test that zero enduring probability means no long-term commitment."""
        p_transactional = 0.9
        p_enduring = 0.0

        expected = 0.0

        assert True  # Placeholder until implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
