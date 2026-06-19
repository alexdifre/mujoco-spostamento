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
    & $conda run --no-capture-output -n robot_sim python -I run_pddl_plan.py

    <#
    Grasp experiment command, intentionally masked.
    $graspArgs = @(
        "run_ur10e_acados_grasp.py",
        "--viewer",
        "--max-steps", "2600",
        "--horizon", "15",
        "--mpc-dt", "0.04",
        "--approach-clearance", "0.02",
        "--grasp-z-offset", "0.0",
        "--lift-z-offset", "0.18",
        "--ee-pos-weight", "600",
        "--ee-z-weight", "700",
        "--ee-terminal-weight", "1300",
        "--ee-terminal-z-weight", "1500",
        "--q-weight", "0.05",
        "--qv-weight", "0.03",
        "--qf-weight", "30",
        "--qvf-weight", "0.1",
        "--delta-q-max", "0.14",
        "--delta-dq-max", "0.65",
        "--delta-tau-max", "55",
        "--tau-slew-rate", "900",
        "--delta-tau-cost", "0.01",
        "--ee-upright-weight", "0",
        "--ee-terminal-upright-weight", "0",
        "--reach-tol", "0.105",
        "--grasp-tol", "0.055",
        "--close-steps", "450",
        "--close-ramp-steps", "400",
        "--grasp-aperture-threshold", "0.075",
        "--latch-aperture-threshold", "0.145",
        "--regularization", "1e-5"
    )
    & $conda run --no-capture-output -n robot_sim python -I @graspArgs
    #>
}
finally {
    Pop-Location
}
