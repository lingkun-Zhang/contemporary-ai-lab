@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"

rem Use the verified local runtime; Q1_PYTHON can point to another Python 3.9+.
if defined Q1_PYTHON goto check_python
set "Q1_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%Q1_PYTHON%" goto check_python
set "Q1_PYTHON=py"
set "Q1_PY_ARGS=-3"

:check_python
"%Q1_PYTHON%" %Q1_PY_ARGS% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
if /I "%~1"=="experiments" goto experiments
if /I "%~1"=="explorations" goto explorations
if /I "%~1"=="bonus" goto bonus
if /I "%~1"=="tuning" goto tuning
if /I "%~1"=="confirmation" goto confirmation
if /I "%~1"=="tests" goto tests

"%Q1_PYTHON%" %Q1_PY_ARGS% maze_astar.py %*
goto finished

:experiments
"%Q1_PYTHON%" %Q1_PY_ARGS% run_experiments.py
goto finished

:tests
"%Q1_PYTHON%" %Q1_PY_ARGS% -m unittest discover -v
goto finished

:explorations
"%Q1_PYTHON%" %Q1_PY_ARGS% run_explorations.py
goto finished

:bonus
"%Q1_PYTHON%" %Q1_PY_ARGS% run_bonus_experiments.py
goto finished

:tuning
"%Q1_PYTHON%" %Q1_PY_ARGS% run_landmark_tuning.py
goto finished

:confirmation
"%Q1_PYTHON%" %Q1_PY_ARGS% run_landmark_confirmation.py
goto finished

:missing_python
echo No usable Python 3.9+ was found.
echo Install Python, or set Q1_PYTHON to the full path of python.exe.
set "Q1_EXIT=1"
goto cleanup

:finished
set "Q1_EXIT=%ERRORLEVEL%"

:cleanup
popd
if not "%Q1_NO_PAUSE%"=="1" pause
exit /b %Q1_EXIT%
