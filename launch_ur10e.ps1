Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$workdir = Join-Path $root "catkin_ws"
$conda = "C:\conda-forge\condabin\conda.bat"

if (-not (Test-Path -LiteralPath $conda)) {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -eq $cmd) {
        throw "Conda not found. Expected C:\conda-forge\condabin\conda.bat or conda in PATH."
    }
    $conda = $cmd.Source
}

$env:PYTHONNOUSERSITE = "1"
Push-Location $workdir
try {
    & $conda run --no-capture-output -n mlc-stack python -I run_pddl_plan.py
}
finally {
    Pop-Location
}
