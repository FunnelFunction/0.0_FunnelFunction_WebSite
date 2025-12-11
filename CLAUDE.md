# CLAUDE.md

> Guidelines for AI assistants working with the Funnel Function Website repository

---

## Project Overview

**Funnel Function Website** is a computational marketing science platform implementing the master equation:

```
f(x) = (B · M · S) / (N + L + Θ) × W
```

Body × Mind × Soul, divided by Noise + Load + Friction, gated by Writability.

### Architecture

```
WordPress (funnelfunction.com)  ←→  GitHub Repository (Source of Truth)
         ↓                                      ↓
    Content Delivery                   Code, Equations, Simulators
    SEO, Analytics                     GitHub Pages Hosting
```

- **WordPress.com**: Blog, whitepapers, CMS content
- **GitHub Pages**: Static hosting for React apps and HTML pages
- **GitHub Repository**: Version control, source of truth for all code

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Simulators | React 18 + TypeScript + Vite |
| Equations | Python + PyTorch (differentiable) |
| Static Pages | HTML + CSS (BEM methodology) |
| Build/Bundle | Vite 5.x |
| Testing | Vitest |
| CI/CD | GitHub Actions |
| Hosting | GitHub Pages |

---

## Directory Structure

The repository uses a **semantic hierarchical naming system**:

```
[MAJOR].[MINOR].[PATCH]_[type]_[descriptor]
```

### Major Categories (0-9)

| # | Domain | Description |
|---|--------|-------------|
| 0 | Root | Configuration, documentation, meta |
| 1 | Pages | Core static pages (Home, About) |
| 2 | Deep Knowledge | Whitepapers, research, library |
| 3 | Articles | Blog posts, Medium, Substack |
| 4 | Social | Social media feeds, aggregator |
| 5 | Simulators | Interactive React tools |
| 6 | Equations | Mathematical implementations |
| 7 | Assets | Images, styles, fonts |
| 8 | Scripts | Build, deploy, utility scripts |
| 9 | Tests | Unit, integration, E2E tests |

### Key Directories

```
0.0_FunnelFunction_WebSite/
├── index.html                              # Main landing page (sales-focused)
├── 0.0_root_config/                        # Configs, README, LICENSE
├── 0.1_root_documentation/                 # Architecture, standards docs
├── 1.0_pages_core/                         # Static HTML pages
│   ├── 1.1_page_home/
│   └── 1.2_page_about/
├── 4.0_social/
│   └── 4.1_social_aggregator/              # f(Social) hub page
├── 5.0_simulators/
│   ├── 5.1_simulator_campaign_commander/   # React app
│   └── 5.5_simulator_gating_function_lab/  # React app
├── 6.0_equations/
│   ├── 6.1_equation_master_funnel_function/
│   └── 6.2_equation_commitment_function/
├── 7.0_assets/                             # Static resources
├── 8.0_scripts/                            # Build/deploy scripts
├── 9.0_tests/                              # Test suite
└── .github/workflows/deploy-pages.yml      # CI/CD pipeline
```

---

## Naming Conventions

### Files

```
[MAJOR].[MINOR].[PATCH]_[type]_[descriptor].[ext]
```

**Type Prefixes:**
- `index_` - Entry points, table of contents
- `content_` - Written prose content
- `config_` - Configuration files
- `script_` - Executable scripts
- `embed_` - Embeddable HTML widgets
- `styles_` - CSS files
- `util_` - Utility functions
- `test_` - Test files
- `model_` - ML/PyTorch models

**Examples:**
- `0.1.0_index_documentation.md`
- `5.1.0_README.md`
- `6.1.2_implementation.py`
- `6.1.3_model_pytorch.py`

### Folders

```
[MAJOR].[MINOR]_[descriptor]/
```

**Examples:**
- `5.1_simulator_campaign_commander/`
- `6.2_equation_commitment_function/`

---

## Development Workflows

### Running Simulators Locally

```bash
# Campaign Commander
cd 5.0_simulators/5.1_simulator_campaign_commander
npm install
npm run dev
# → http://localhost:5173

# Gating Function Lab
cd 5.0_simulators/5.5_simulator_gating_function_lab
npm install
npm run dev
# → http://localhost:5173
```

### Building Simulators

```bash
cd 5.0_simulators/5.1_simulator_campaign_commander
npm run build
# Output: dist/
```

### Running Tests

```bash
cd 5.0_simulators/5.1_simulator_campaign_commander
npm test
```

---

## Deployment

### Automatic Deployment

Push to `main` branch triggers GitHub Actions workflow:
1. Builds all React simulators
2. Copies static assets to `_site/`
3. Deploys to GitHub Pages

### Live URLs

| Resource | URL |
|----------|-----|
| Main Landing | `https://funnelfunction.github.io/0.0_FunnelFunction_WebSite/` |
| Campaign Commander | `/simulators/campaign-commander/` |
| Gating Function Lab | `/simulators/gating-function-lab/` |
| f(Social) Hub | `/social/4.1_social_aggregator/` |

### Vite Base Paths

Each simulator's `vite.config.ts` has a specific base path for GitHub Pages:
- Campaign Commander: `/0.0_FunnelFunction_WebSite/simulators/campaign-commander/`
- Gating Function Lab: `/0.0_FunnelFunction_WebSite/simulators/gating-function-lab/`

---

## Coding Standards

### Python

- Follow **PEP 8**
- Use **type hints** for all functions
- Max line length: 88 characters (Black formatter)
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

```python
def calculate_attention(body: float, mind: float, soul: float) -> float:
    """
    Calculate the funnel function value.

    Args:
        body: Somatic certainty (σ)
        mind: Prediction confidence (π)
        soul: Identity congruence (Ι)

    Returns:
        The computed attention value
    """
    return (body * mind * soul) / suppression()
```

### TypeScript/React

- Use **ESLint + Prettier**
- Prefer **TypeScript** over JavaScript
- Use **functional components** with hooks
- Components: `PascalCase`
- Functions/variables: `camelCase`
- Constants: `UPPER_SNAKE_CASE`

```typescript
interface SimulatorState {
  body: number;
  mind: number;
  soul: number;
}

const CampaignCommander: React.FC<Props> = ({ initialConfig }) => {
  const [state, setState] = useState<SimulatorState>(initialState);
  // ...
};
```

### CSS

- Use **BEM methodology** (Block Element Modifier)
- Use **CSS variables** for theming

```css
/* Block */
.campaign-commander { }

/* Element */
.campaign-commander__header { }

/* Modifier */
.campaign-commander--dark { }
```

**Design Tokens:**
```css
:root {
  --primary: #8b5cf6;
  --bg: #0a0a0f;
  --text: #f1f5f9;
  --accent: #ef4444;
}
```

---

## The Core Equations

### Master Funnel Function `f(x)`

```
f(x) = (B · M · S) / (N + L + Θ) × W

Where:
  B = Body (sensory engagement) [0-1]
  M = Mind (cognitive processing) [0-1]
  S = Soul (emotional resonance) [0-1]
  N = Noise (competing signals) [0, ∞)
  L = Load (cognitive burden) [0, ∞)
  Θ = Friction (barriers to action) [0, ∞)
  W = Writability (ability to act - the gate) [0-1]
```

**Key Insight:**
- Numerator is **multiplicative** - any zero collapses the function
- Denominator is **additive** - suppressors compound
- W is a **gate** - zero writability means zero conversion regardless of attention

### Commitment Function `f(Commitment)`

```
f(Commitment) = P_Transactional × P_Enduring
```

Implemented in `6.0_equations/6.2_equation_commitment_function/`

---

## Important Files

| File | Purpose |
|------|---------|
| `index.html` | Main sales-focused landing page |
| `.github/workflows/deploy-pages.yml` | CI/CD deployment pipeline |
| `5.0_simulators/*/package.json` | Simulator dependencies |
| `5.0_simulators/*/vite.config.ts` | Vite build configuration |
| `6.0_equations/*/6.x.2_implementation.py` | Pure Python implementations |
| `6.0_equations/*/6.x.3_model_pytorch.py` | Differentiable PyTorch models |
| `0.1_root_documentation/0.1.3_coding_standards.md` | Full coding standards |
| `0.1_root_documentation/0.1.2_naming_conventions.md` | Naming convention reference |

---

## Common Tasks

### Adding a New Simulator

1. Create folder: `5.0_simulators/5.X_simulator_[name]/`
2. Initialize React project with Vite + TypeScript
3. Set base path in `vite.config.ts`
4. Add build step to `.github/workflows/deploy-pages.yml`
5. Create `5.X.0_README.md` documenting the simulator

### Adding a New Equation

1. Create folder: `6.0_equations/6.X_equation_[name]/`
2. Add files:
   - `6.X.0_README.md` - Documentation
   - `6.X.1_definition.md` - Mathematical definition
   - `6.X.2_implementation.py` - Pure Python
   - `6.X.3_model_pytorch.py` - Differentiable PyTorch

### Modifying Landing Page

Edit `index.html` directly. It contains:
- Hero section with value proposition
- Services grid
- Vibe Coding event promotion
- Simulator showcase
- Contact section
- Footer with navigation

---

## Git Conventions

### Commit Messages

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code restructure
- `test` - Adding tests
- `chore` - Maintenance

**Example:**
```
feat(simulators): add persona presets to Gating Function Lab

- Adds 4 psychological persona presets
- Implements instant parameter loading
- Updates documentation

Closes #42
```

### Branch Naming

```
<type>/<description>

feature/add-campaign-commander-simulator
fix/social-aggregator-rate-limit
docs/update-naming-conventions
```

---

## External Resources

- **WordPress Site**: [funnelfunction.com](https://funnelfunction.com)
- **Marketing Principals**: [github.com/FunnelFunction/0.0_git_funnelfunction_marketing_Principals](https://github.com/FunnelFunction/0.0_git_funnelfunction_marketing_Principals)
- **LinkedIn**: [linkedin.com/in/funnelfunction](https://www.linkedin.com/in/funnelfunction/)
- **Twitter**: [@FunnelFunction](https://twitter.com/FunnelFunction)
- **GitHub Org**: [github.com/FunnelFunction](https://github.com/FunnelFunction)

---

## Principles for AI Assistants

1. **Follow the naming convention** - Every file gets a semantic version number
2. **No orphan files** - Everything has a numbered place in the hierarchy
3. **Self-documenting code** - Names tell you category, type, and purpose
4. **Type safety** - Use TypeScript for JS, type hints for Python
5. **KISS** - Clarity over cleverness, one function one purpose
6. **Documentation first** - Every public function needs a docstring
7. **Cross-promote** - Simulators link to whitepapers, case studies, and each other
8. **Test before commit** - Run `npm run build` for simulators before pushing

---

*Last updated: 2025-12-11*
