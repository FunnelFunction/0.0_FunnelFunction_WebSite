/**
 * Campaign Commander - Marketing Simulator
 *
 * Purpose: Let users control marketing variables and see real-time effects
 * Based on: Master Funnel Function f(x) = (B·M·S) / (N+L+Θ) × W
 *
 * Cross-Promotion:
 * - Links to whitepapers explaining the math
 * - Links to case studies showing real-world application
 */

import React, { useState, useMemo } from 'react';
import './styles.css';

// ============================================================================
// TYPES
// ============================================================================

interface SimulatorState {
  budget: number;        // Amplifier (doesn't affect conversion rate, just reach)
  body: number;          // B: Creative quality / sensory strength [0-1]
  mind: number;          // M: Targeting precision / intent match [0-1]
  soul: number;          // S: Brand authenticity / identity congruence [0-1]
  noise: number;         // N: Environmental interference [0-1]
  load: number;          // L: Cognitive burden [0-1]
  friction: number;      // Θ: Conversion barriers [0-1]
  writability: number;   // W: Receptivity gate [0-1]
}

// ============================================================================
// EQUATION IMPLEMENTATION
// ============================================================================

function calculateAttention(state: SimulatorState): number {
  const { body, mind, soul, noise, load, friction, writability } = state;

  // Signal = B × M × S (multiplicative - any zero collapses it)
  const signal = body * mind * soul;

  // Suppression = N + L + Θ (additive - they accumulate)
  const suppression = noise + load + friction + 0.01; // Avoid division by zero

  // Raw attention
  const rawAttention = signal / suppression;

  // Apply writability gate
  const gatedAttention = rawAttention * writability;

  return Math.min(gatedAttention, 1); // Cap at 1
}

function diagnoseBottleneck(state: SimulatorState): string {
  const { body, mind, soul, noise, load, friction } = state;

  // Find weakest signal channel
  const channels = [
    { name: 'Body (Creative Quality)', value: body },
    { name: 'Mind (Targeting)', value: mind },
    { name: 'Soul (Authenticity)', value: soul },
  ];
  const weakestChannel = channels.reduce((a, b) => a.value < b.value ? a : b);

  // Find strongest suppressor
  const suppressors = [
    { name: 'Noise', value: noise },
    { name: 'Load', value: load },
    { name: 'Friction', value: friction },
  ];
  const strongestSuppressor = suppressors.reduce((a, b) => a.value > b.value ? a : b);

  if (weakestChannel.value < 0.3) {
    return `⚠️ Bottleneck: ${weakestChannel.name} is critically low (${(weakestChannel.value * 100).toFixed(0)}%). This collapses your signal.`;
  }

  if (strongestSuppressor.value > 0.7) {
    return `⚠️ High Suppression: ${strongestSuppressor.name} is at ${(strongestSuppressor.value * 100).toFixed(0)}%. Consider reducing friction before increasing signal.`;
  }

  return '✅ Balanced configuration. Fine-tune to optimize.';
}

// ============================================================================
// COMPONENTS
// ============================================================================

interface SliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  color: string;
  description: string;
}

function Slider({ label, value, onChange, color, description }: SliderProps) {
  return (
    <div className="slider-container">
      <div className="slider-header">
        <span className="slider-label">{label}</span>
        <span className="slider-value" style={{ color }}>{(value * 100).toFixed(0)}%</span>
      </div>
      <input
        type="range"
        min="0"
        max="100"
        value={value * 100}
        onChange={(e) => onChange(Number(e.target.value) / 100)}
        className="slider"
        style={{ '--slider-color': color } as React.CSSProperties}
      />
      <div className="slider-description">{description}</div>
    </div>
  );
}

function AttentionMeter({ value }: { value: number }) {
  const percentage = value * 100;
  const getColor = () => {
    if (percentage < 30) return '#ef4444';
    if (percentage < 60) return '#f59e0b';
    return '#10b981';
  };

  return (
    <div className="attention-meter">
      <div className="meter-label">Attention Score</div>
      <div className="meter-value" style={{ color: getColor() }}>
        {percentage.toFixed(1)}%
      </div>
      <div className="meter-bar">
        <div
          className="meter-fill"
          style={{ width: `${percentage}%`, background: getColor() }}
        />
      </div>
      <div className="meter-sublabel">
        Probability of capturing attention with current configuration
      </div>
    </div>
  );
}

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const [state, setState] = useState<SimulatorState>({
    budget: 50000,
    body: 0.7,
    mind: 0.6,
    soul: 0.5,
    noise: 0.3,
    load: 0.2,
    friction: 0.2,
    writability: 0.8,
  });

  const attention = useMemo(() => calculateAttention(state), [state]);
  const diagnosis = useMemo(() => diagnoseBottleneck(state), [state]);

  const updateState = (key: keyof SimulatorState) => (value: number) => {
    setState(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎯 Campaign Commander</h1>
        <p>Control the variables. See the math in action.</p>
      </header>

      <main className="main">
        {/* Results Panel */}
        <section className="results-panel">
          <AttentionMeter value={attention} />
          <div className="diagnosis">{diagnosis}</div>

          <div className="equation-display">
            <div className="equation-label">The Equation</div>
            <div className="equation">
              𝒜 = ({state.body.toFixed(2)} × {state.mind.toFixed(2)} × {state.soul.toFixed(2)}) /
              ({state.noise.toFixed(2)} + {state.load.toFixed(2)} + {state.friction.toFixed(2)}) × {state.writability.toFixed(2)}
            </div>
            <div className="equation-result">
              = {(state.body * state.mind * state.soul).toFixed(3)} / {(state.noise + state.load + state.friction + 0.01).toFixed(3)} × {state.writability.toFixed(2)}
              = <strong>{(attention * 100).toFixed(1)}%</strong>
            </div>
          </div>
        </section>

        {/* Controls */}
        <section className="controls">
          <div className="control-group">
            <h3 className="control-group-title signal">📡 Signal Channels (Multiplicative)</h3>
            <Slider
              label="Body (B)"
              value={state.body}
              onChange={updateState('body')}
              color="#10b981"
              description="Creative quality, sensory impact"
            />
            <Slider
              label="Mind (M)"
              value={state.mind}
              onChange={updateState('mind')}
              color="#3b82f6"
              description="Targeting precision, intent match"
            />
            <Slider
              label="Soul (S)"
              value={state.soul}
              onChange={updateState('soul')}
              color="#8b5cf6"
              description="Brand authenticity, identity alignment"
            />
          </div>

          <div className="control-group">
            <h3 className="control-group-title suppression">🔇 Suppression (Additive)</h3>
            <Slider
              label="Noise (N)"
              value={state.noise}
              onChange={updateState('noise')}
              color="#ef4444"
              description="Competitive clutter, distractions"
            />
            <Slider
              label="Load (L)"
              value={state.load}
              onChange={updateState('load')}
              color="#f59e0b"
              description="Cognitive effort required"
            />
            <Slider
              label="Friction (Θ)"
              value={state.friction}
              onChange={updateState('friction')}
              color="#ec4899"
              description="Conversion barriers, transaction cost"
            />
          </div>

          <div className="control-group">
            <h3 className="control-group-title gate">🚪 Writability Gate</h3>
            <Slider
              label="Writability (W)"
              value={state.writability}
              onChange={updateState('writability')}
              color="#06b6d4"
              description="Target's openness to change"
            />
          </div>
        </section>
      </main>

      {/* Cross-Promotion Footer */}
      <footer className="footer">
        <div className="crosspromo">
          <a href="https://funnelfunction.com/whitepapers/mathematical-decomposition-of-trust/" className="crosspromo-link blue">
            📚 Read the Whitepaper
          </a>
          <a href="https://funnelfunction.com/2025/12/05/the-purest-execution-of-fx-ever-recorded-the-mint-mobile-case-study/" className="crosspromo-link orange">
            📊 See Real Case Study
          </a>
        </div>
        <div className="attribution">
          Built by <a href="https://funnelfunction.com">Funnel Function Institute</a>
        </div>
      </footer>
    </div>
  );
}
