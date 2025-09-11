# Runs lint and tests
param([switch]$noLint)

$ErrorActionPreference = 'Stop'
. ..\.venv\Scripts\Activate.ps1
if (-not $noLint) { ruff check src tests --fix }
python -m pytest -q
