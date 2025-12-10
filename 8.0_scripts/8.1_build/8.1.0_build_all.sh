#!/bin/bash
# =============================================================================
# 8.1.0_build_all.sh
#
# Purpose: Build all components of the Funnel Function website
# Author: Funnel Function Institute
# Created: 2025-12-10
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Funnel Function - Build All"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment variables
if [ -f "$ROOT_DIR/.env" ]; then
    export $(cat "$ROOT_DIR/.env" | grep -v '^#' | xargs)
    echo "✓ Environment loaded"
else
    echo "⚠ No .env file found. Using defaults."
fi

# Step 1: Validate structure
echo ""
echo "Step 1: Validating structure..."
echo "──────────────────────────────"

# Check required directories exist
REQUIRED_DIRS=(
    "0.0_root_config"
    "0.1_root_documentation"
    "1.0_pages_core"
    "2.0_deep_knowledge"
    "5.0_simulators"
    "6.0_equations"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$ROOT_DIR/$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir (missing)"
        exit 1
    fi
done

# Step 2: Build simulators
echo ""
echo "Step 2: Building simulators..."
echo "──────────────────────────────"

if [ -f "$SCRIPT_DIR/8.1.1_build_simulators.sh" ]; then
    bash "$SCRIPT_DIR/8.1.1_build_simulators.sh"
else
    echo "  ⚠ Simulator build script not found (skipping)"
fi

# Step 3: Build static pages
echo ""
echo "Step 3: Building static pages..."
echo "──────────────────────────────"

if [ -f "$SCRIPT_DIR/8.1.2_build_static_pages.sh" ]; then
    bash "$SCRIPT_DIR/8.1.2_build_static_pages.sh"
else
    echo "  ⚠ Static pages build script not found (skipping)"
fi

# Step 4: Run tests
echo ""
echo "Step 4: Running tests..."
echo "──────────────────────────────"

if [ -d "$ROOT_DIR/9.0_tests" ]; then
    echo "  ⚠ Tests not yet implemented (skipping)"
else
    echo "  ⚠ Test directory not found (skipping)"
fi

# Done
echo ""
echo "=============================================="
echo "  Build Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  • Deploy to GitHub Pages: ./8.2_deploy/8.2.0_deploy_github_pages.sh"
echo "  • Sync to WordPress: ./8.2_deploy/8.2.1_deploy_wordpress_sync.sh"
echo ""
