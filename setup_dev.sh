#!/bin/bash
# AIC Development Setup — source this in any distrobox terminal
# Usage: source ~/ws_aic/setup_dev.sh

source /opt/ros/kilted/setup.bash
source /ws_aic/install/setup.bash
_PIXI_SITE_PACKAGES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src/aic/.pixi/envs/default/lib/python3.12/site-packages"
export PYTHONPATH="${_PIXI_SITE_PACKAGES}:$PYTHONPATH"

echo "✓ AIC dev environment ready (system ROS + pixi packages)"
