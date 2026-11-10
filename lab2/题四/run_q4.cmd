@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if defined Q4_PYTHON goto check_python
set "Q4_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%Q4_PYTHON%" goto check_python
set "Q4_PYTHON=py"
set "Q4_PY_ARGS=-3"
:check_python
"%Q4_PYTHON%" %Q4_PY_ARGS% -c "import sys;sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
if /I "%~1"=="tests" goto tests
if /I "%~1"=="experiments" goto experiments
if /I "%~1"=="robustness" goto robustness
if /I "%~1"=="lazy" goto lazy
"%Q4_PYTHON%" %Q4_PY_ARGS% treasure_game.py %*
goto finished
:tests
"%Q4_PYTHON%" %Q4_PY_ARGS% -m unittest discover -v
goto finished
:experiments
"%Q4_PYTHON%" %Q4_PY_ARGS% run_experiments.py
goto finished
:robustness
"%Q4_PYTHON%" %Q4_PY_ARGS% run_robustness_study.py
goto finished
:lazy
"%Q4_PYTHON%" %Q4_PY_ARGS% run_lazy_study.py
goto finished
:missing_python
echo No usable Python 3.9+ was found. Set Q4_PYTHON to the Python executable path.
set "Q4_EXIT=1"
goto cleanup
:finished
set "Q4_EXIT=%ERRORLEVEL%"
:cleanup
popd
if not "%Q4_NO_PAUSE%"=="1" pause
exit /b %Q4_EXIT%
