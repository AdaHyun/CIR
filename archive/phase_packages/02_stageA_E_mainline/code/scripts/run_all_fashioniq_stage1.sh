#!/usr/bin/env bash
set -euo pipefail

for category in dress shirt toptee; do
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_fashioniq_stage1.sh" "${category}"
done
