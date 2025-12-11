# Funnel Function Website

> **The Mathematical Architecture of Commercial Attention**

```
f(x) = (B · M · S) / (N + L + Θ) × W
```

Body × Mind × Soul, divided by Noise + Load + Friction, gated by Writability.
The equation that explains why attention converts.

---

## Live Sites

| Site | Description | URL |
|------|-------------|-----|
| **Main Landing** | Portal to all Funnel Function resources | [funnelfunction.github.io/0.0_FunnelFunction_WebSite](https://funnelfunction.github.io/0.0_FunnelFunction_WebSite/) |
| **Campaign Commander** | Interactive simulator - control B, M, S variables | [Launch Simulator](https://funnelfunction.github.io/0.0_FunnelFunction_WebSite/simulators/campaign-commander/) |
| **Gating Function Lab** | Explore the W (Writability) gating mechanism | [Launch Lab](https://funnelfunction.github.io/0.0_FunnelFunction_WebSite/simulators/gating-function-lab/) |
| **f(x) Social** | The Origin Hub - where content is born | [Visit Hub](https://funnelfunction.github.io/0.0_FunnelFunction_WebSite/social/4.1_social_aggregator/) |
| **Blog & Articles** | Case studies and real-world analysis | [funnelfunction.com/blog](https://funnelfunction.com/blog/) |
| **Whitepapers** | Academic foundations and formal proofs | [funnelfunction.com/whitepapers](https://funnelfunction.com/whitepapers/) |

---

## The Principals

This website implements concepts from the **Funnel Function Marketing Principals** - a comprehensive framework for computational marketing science.

**[View the Marketing Principals Repository](https://github.com/FunnelFunction/0.0_git_funnelfunction_marketing_Principals)**

The principals define:
- The master equation `f(x)` and all its components
- The Commitment Function `f(Commitment)`
- Variable definitions (Body, Mind, Soul, Noise, Load, Friction, Writability)
- Mathematical proofs and derivations
- Application methodology

Everything in this website - the simulators, the equations, the analysis - flows from those principals.

---

## Repository Structure

```
0.0_FunnelFunction_WebSite/
├── index.html                      # Main landing page
├── 0.1_root_documentation/         # Deployment guides, architecture docs
├── 4.0_social/
│   └── 4.1_social_aggregator/      # f(x) Social - The Origin Hub
├── 5.0_simulators/
│   ├── 5.1_simulator_campaign_commander/   # React app - Campaign Commander
│   └── 5.5_simulator_gating_function_lab/  # React app - Gating Function Lab
├── 6.0_equations/
│   ├── 6.1_equation_master_fx/     # Master Funnel Function (Python + PyTorch)
│   └── 6.2_equation_commitment/    # Commitment Function (Python + PyTorch)
├── 8.0_scripts/
│   └── 8.1_script_wordpress_sync/  # WordPress integration scripts
└── .github/workflows/              # GitHub Actions deployment
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Simulators | React + TypeScript + Vite |
| Equations | Python + PyTorch (differentiable) |
| Static Pages | HTML + CSS |
| Hosting | GitHub Pages |
| Content | WordPress.com (blog/whitepapers) |
| CI/CD | GitHub Actions |

---

## Local Development

```bash
# Clone the repository
git clone https://github.com/FunnelFunction/0.0_FunnelFunction_WebSite.git
cd 0.0_FunnelFunction_WebSite

# Run Campaign Commander locally
cd 5.0_simulators/5.1_simulator_campaign_commander
npm install
npm run dev
# Opens at http://localhost:5173

# Run Gating Function Lab locally
cd ../5.5_simulator_gating_function_lab
npm install
npm run dev
```

---

## Deployment

Deployment is automatic via GitHub Actions on push to `main`.

See [Deployment Guide](0.1_root_documentation/0.1.4_deployment_guide.md) for details.

---

## The Equation

```
f(x) = (B · M · S) / (N + L + Θ) × W

Where:
  B = Body    (Physical/sensory engagement)
  M = Mind    (Cognitive processing)
  S = Soul    (Emotional resonance)
  N = Noise   (Competing signals)
  L = Load    (Cognitive burden)
  Θ = Theta   (Friction/barriers)
  W = Write   (Ability to act - the gate)
```

The numerator is **multiplicative** - if any channel (B, M, or S) hits zero, the entire function collapses. This is why "almost-sales" fail.

The denominator is **additive** - noise, load, and friction compound against conversion.

W is the **gate** - regardless of attention score, if writability is zero, nothing converts.

---

## Links

- **Website**: [funnelfunction.github.io/0.0_FunnelFunction_WebSite](https://funnelfunction.github.io/0.0_FunnelFunction_WebSite/)
- **WordPress**: [funnelfunction.com](https://funnelfunction.com)
- **Marketing Principals**: [github.com/FunnelFunction/0.0_git_funnelfunction_marketing_Principals](https://github.com/FunnelFunction/0.0_git_funnelfunction_marketing_Principals)
- **LinkedIn**: [linkedin.com/in/funnelfunction](https://www.linkedin.com/in/funnelfunction/)
- **Twitter**: [@FunnelFunction](https://twitter.com/FunnelFunction)
- **GitHub**: [github.com/FunnelFunction](https://github.com/FunnelFunction)

---

*Marketing science that actually computes.*
