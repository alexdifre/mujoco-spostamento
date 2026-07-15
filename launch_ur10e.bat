@echo off
setlocal

set "ROOT=%~dp0"
set "WORKDIR=%ROOT%catkin_ws"
set "CONDA_BAT=C:\conda-forge\condabin\conda.bat"
set "PYTHONNOUSERSITE=1"

if not exist "%CONDA_BAT%" (
    set "CONDA_BAT=conda"
)

cd /d "%WORKDIR%" || exit /b 1
call "%CONDA_BAT%" run --no-capture-output -n mlc-stack python -I run_pddl_plan.py
exit /b %ERRORLEVEL%
