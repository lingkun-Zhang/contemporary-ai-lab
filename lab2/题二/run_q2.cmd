@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if defined Q2_PYTHON goto check_python
set "Q2_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%Q2_PYTHON%" goto check_python
set "Q2_PYTHON=py"
set "Q2_PY_ARGS=-3"

:check_python
"%Q2_PYTHON%" %Q2_PY_ARGS% -c "import sys;sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
if /I "%~1"=="tests" goto tests
if /I "%~1"=="experiments" goto experiments
if /I "%~1"=="supplementary" goto supplementary
if /I "%~1"=="tuning" goto tuning
if /I "%~1"=="confirmation" goto confirmation
"%Q2_PYTHON%" %Q2_PY_ARGS% weighted_astar.py %*
goto finished

:tests
"%Q2_PYTHON%" %Q2_PY_ARGS% -m unittest discover -v
goto finished

:experiments
"%Q2_PYTHON%" %Q2_PY_ARGS% run_experiments.py
goto finished

:supplementary
"%Q2_PYTHON%" %Q2_PY_ARGS% run_supplementary_experiments.py
goto finished

:tuning
"%Q2_PYTHON%" %Q2_PY_ARGS% run_heap_tuning.py
goto finished

:confirmation
"%Q2_PYTHON%" %Q2_PY_ARGS% run_heap_confirmation.py
goto finished

:missing_python
echo No usable Python 3.9+ was found. Set Q2_PYTHON to the Python executable path.
set "Q2_EXIT=1"
goto cleanup

:finished
set "Q2_EXIT=%ERRORLEVEL%"

:cleanup
popd
if not "%Q2_NO_PAUSE%"=="1" pause
exit /b %Q2_EXIT%
