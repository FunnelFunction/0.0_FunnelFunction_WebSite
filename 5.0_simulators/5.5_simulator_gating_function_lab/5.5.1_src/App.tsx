/**
 * Gating Function Lab - Writability Simulator
 *
 * Purpose: Explore what makes audiences receptive to change
 * Based on: f(Commitment) = (σ × π × Ι) / (F + R + SQ)
 *
 * This determines the Writability (W) gate in the Master Funnel Function
 *
 * Cross-Promotion:
 * - Links to whitepapers on commitment psychology
 * - Links to Campaign Commander for full equation
 */

import React, { useState, useMemo } from 'react';
import './styles.css';

// ============================================================================
// TYPES
// ============================================================================

interface CommitmentState {
  // Numerator - Approach Forces (multiplicative)
  signalStrength: number;    // σ: How clear/strong is the message [0-1]
  personalRelevance: number; // π: How much does it matter to them [0-1]
  identityAlignment: number; // Ι: How well does it fit their self-image [0-1]

  // Denominator - Resistance Forces (additive)
  fear: number;              // F: Fear of change, uncertainty [0-1]
  resourceCost: number;      // R: Time, money, effort required [0-1]
  statusQuo: number;         // SQ: Comfort with current state [0-1]
}

interface Persona {
  name: string;
  description: string;
  state: CommitmentState;
}

// ============================================================================
// PRESETS - PERSONA ARCHETYPES
// ============================================================================

const PERSONAS: Persona[] = [
  {
    name: "Early Adopter",
    description: "Loves new things, low fear, high identity with innovation",
    state: {
      signalStrength: 0.7,
      personalRelevance: 0.8,
      identityAlignment: 0.9,
      fear: 0.2,
      resourceCost: 0.3,
      statusQuo: 0.1,
    }
  },
  {
    name: "Skeptical Professional",
    description: "Needs proof, high resource consciousness, moderate fear",
    state: {
      signalStrength: 0.5,
      personalRelevance: 0.6,
      identityAlignment: 0.4,
      fear: 0.5,
      resourceCost: 0.7,
      statusQuo: 0.5,
    }
  },
  {
    name: "Satisfied Customer",
    description: "Happy with current solution, high status quo bias",
    state: {
      signalStrength: 0.6,
      personalRelevance: 0.4,
      identityAlignment: 0.5,
      fear: 0.3,
      resourceCost: 0.4,
      statusQuo: 0.9,
    }
  },
  {
    name: "Crisis Mode",
    description: "Current situation is painful, desperate for change",
    state: {
      signalStrength: 0.8,
      personalRelevance: 0.95,
      identityAlignment: 0.6,
      fear: 0.4,
      resourceCost: 0.3,
      statusQuo: 0.05,
    }
  },
];

// ============================================================================
// EQUATION IMPLEMENTATION
// ============================================================================

function calculateWritability(state: CommitmentState): number {
  const { signalStrength, personalRelevance, identityAlignment, fear, resourceCost, statusQuo } = state;

  // Approach = σ × π × Ι (multiplicative - any zero kills it)
  const approach = signalStrength * personalRelevance * identityAlignment;

  // Resistance = F + R + SQ (additive - they accumulate)
  const resistance = fear + resourceCost + statusQuo + 0.01; // Avoid division by zero

  // Writability = Approach / Resistance
  const writability = approach / resistance;

  return Math.min(writability, 1); // Cap at 1
}

function diagnoseResistance(state: CommitmentState): string {
  const { signalStrength, personalRelevance, identityAlignment, fear, resourceCost, statusQuo } = state;

  // Find weakest approach factor
  const approaches = [
    { name: 'Signal Strength', value: signalStrength, fix: 'Make your message clearer and more compelling' },
    { name: 'Personal Relevance', value: personalRelevance, fix: 'Show them why this matters to THEM specifically' },
    { name: 'Identity Alignment', value: identityAlignment, fix: 'Frame it as consistent with who they are' },
  ];
  const weakestApproach = approaches.reduce((a, b) => a.value < b.value ? a : b);

  // Find strongest resistance
  const resistances = [
    { name: 'Fear', value: fear, fix: 'Reduce uncertainty with guarantees, testimonials, trials' },
    { name: 'Resource Cost', value: resourceCost, fix: 'Lower the barrier: easier payment, less time, simpler onboarding' },
    { name: 'Status Quo', value: statusQuo, fix: 'Highlight the cost of inaction, show what they\'re missing' },
  ];
  const strongestResistance = resistances.reduce((a, b) => a.value > b.value ? a : b);

  if (weakestApproach.value < 0.3) {
    return `🔴 Critical Gap: ${weakestApproach.name} is only ${(weakestApproach.value * 100).toFixed(0)}%. ${weakestApproach.fix}`;
  }

  if (strongestResistance.value > 0.7) {
    return `🟠 High Resistance: ${strongestResistance.name} at ${(strongestResistance.value * 100).toFixed(0)}%. ${strongestResistance.fix}`;
  }

  const writability = calculateWritability(state);
  if (writability > 0.5) {
    return '🟢 High Writability: This audience is receptive. Focus on your core message.';
  }

  return '🟡 Moderate: Balance approach strength with resistance reduction.';
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

function WritabilityMeter({ value }: { value: number }) {
  const percentage = value * 100;
  const getColor = () => {
    if (percentage < 25) return '#ef4444';
    if (percentage < 50) return '#f59e0b';
    if (percentage < 75) return '#3b82f6';
    return '#10b981';
  };

  const getLabel = () => {
    if (percentage < 25) return 'Resistant';
    if (percentage < 50) return 'Guarded';
    if (percentage < 75) return 'Open';
    return 'Receptive';
  };

  return (
    <div className="writability-meter">
      <div className="meter-label">Writability Score</div>
      <div className="meter-value" style={{ color: getColor() }}>
        {percentage.toFixed(1)}%
      </div>
      <div className="meter-status" style={{ color: getColor() }}>
        {getLabel()}
      </div>
      <div className="meter-bar">
        <div
          className="meter-fill"
          style={{ width: `${percentage}%`, background: getColor() }}
        />
      </div>
      <div className="meter-sublabel">
        How receptive is this audience to your message?
      </div>
    </div>
  );
}

function PersonaSelector({ onSelect }: { onSelect: (state: CommitmentState) => void }) {
  return (
    <div className="persona-selector">
      <div className="persona-label">Quick Presets</div>
      <div className="persona-grid">
        {PERSONAS.map((persona) => (
          <button
            key={persona.name}
            className="persona-button"
            onClick={() => onSelect(persona.state)}
            title={persona.description}
          >
            {persona.name}
          </button>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const [state, setState] = useState<CommitmentState>({
    signalStrength: 0.6,
    personalRelevance: 0.5,
    identityAlignment: 0.5,
    fear: 0.4,
    resourceCost: 0.3,
    statusQuo: 0.4,
  });

  const writability = useMemo(() => calculateWritability(state), [state]);
  const diagnosis = useMemo(() => diagnoseResistance(state), [state]);

  const updateState = (key: keyof CommitmentState) => (value: number) => {
    setState(prev => ({ ...prev, [key]: value }));
  };

  const loadPersona = (newState: CommitmentState) => {
    setState(newState);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🚪 Gating Function Lab</h1>
        <p>Explore what makes audiences receptive to change</p>
      </header>

      <main className="main">
        {/* Results Panel */}
        <section className="results-panel">
          <WritabilityMeter value={writability} />
          <div className="diagnosis">{diagnosis}</div>

          <div className="equation-display">
            <div className="equation-label">The Commitment Equation</div>
            <div className="equation">
              W = ({state.signalStrength.toFixed(2)} × {state.personalRelevance.toFixed(2)} × {state.identityAlignment.toFixed(2)}) /
              ({state.fear.toFixed(2)} + {state.resourceCost.toFixed(2)} + {state.statusQuo.toFixed(2)})
            </div>
            <div className="equation-result">
              = {(state.signalStrength * state.personalRelevance * state.identityAlignment).toFixed(3)} / {(state.fear + state.resourceCost + state.statusQuo + 0.01).toFixed(3)}
              = <strong>{(writability * 100).toFixed(1)}%</strong>
            </div>
          </div>

          <PersonaSelector onSelect={loadPersona} />
        </section>

        {/* Controls */}
        <section className="controls">
          <div className="control-group">
            <h3 className="control-group-title approach">✨ Approach Forces (Multiplicative)</h3>
            <Slider
              label="Signal Strength (σ)"
              value={state.signalStrength}
              onChange={updateState('signalStrength')}
              color="#10b981"
              description="How clear and compelling is your message?"
            />
            <Slider
              label="Personal Relevance (π)"
              value={state.personalRelevance}
              onChange={updateState('personalRelevance')}
              color="#3b82f6"
              description="How much does this matter to them personally?"
            />
            <Slider
              label="Identity Alignment (Ι)"
              value={state.identityAlignment}
              onChange={updateState('identityAlignment')}
              color="#8b5cf6"
              description="How well does this fit who they see themselves as?"
            />
          </div>

          <div className="control-group">
            <h3 className="control-group-title resistance">🛡️ Resistance Forces (Additive)</h3>
            <Slider
              label="Fear (F)"
              value={state.fear}
              onChange={updateState('fear')}
              color="#ef4444"
              description="Uncertainty, risk aversion, fear of making a mistake"
            />
            <Slider
              label="Resource Cost (R)"
              value={state.resourceCost}
              onChange={updateState('resourceCost')}
              color="#f59e0b"
              description="Time, money, effort, switching costs"
            />
            <Slider
              label="Status Quo (SQ)"
              value={state.statusQuo}
              onChange={updateState('statusQuo')}
              color="#ec4899"
              description="Comfort with current situation, inertia"
            />
          </div>
        </section>
      </main>

      {/* Cross-Promotion Footer */}
      <footer className="footer">
        <div className="crosspromo">
          <a href="/5.0_simulators/5.1_simulator_campaign_commander/" className="crosspromo-link purple">
            🎯 Try Campaign Commander
          </a>
          <a href="https://funnelfunction.com/whitepapers/mathematical-decomposition-of-trust/" className="crosspromo-link blue">
            📚 Read the Whitepaper
          </a>
          <a href="https://funnelfunction.com/2025/12/07/the-death-of-trust-and-why-temu-will-die/" className="crosspromo-link orange">
            📊 Trust & Commitment Article
          </a>
        </div>
        <div className="attribution">
          Built by <a href="https://funnelfunction.com">Funnel Function Institute</a>
        </div>
      </footer>
    </div>
  );
}
