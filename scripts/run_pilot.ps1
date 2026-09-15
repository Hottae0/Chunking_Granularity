$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))
python -m rq1.experiments.run_grid --config configs/pilot.yaml
