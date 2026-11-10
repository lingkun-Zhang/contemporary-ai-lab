@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
if defined Q3_PYTHON goto check_python
set "Q3_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%Q3_PYTHON%" goto check_python
set "Q3_PYTHON=py"
set "Q3_PY_ARGS=-3"

:check_python
"%Q3_PYTHON%" %Q3_PY_ARGS% -c "import sys;sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
if /I "%~1"=="tests" goto tests
if /I "%~1"=="experiments" goto experiments
if /I "%~1"=="caching" goto caching
if /I "%~1"=="cache-confirmation" goto cache_confirmation
if /I "%~1"=="sampling-confirmation" goto sampling_confirmation
if /I "%~1"=="correlation" goto correlation
if /I "%~1"=="capacity" goto capacity
if /I "%~1"=="extensions" goto extensions
"%Q3_PYTHON%" %Q3_PY_ARGS% alphabeta_tree.py %*
goto finished

:tests
"%Q3_PYTHON%" %Q3_PY_ARGS% -m unittest discover -v
goto finished

:experiments
"%Q3_PYTHON%" %Q3_PY_ARGS% run_experiments.py
goto finished

:missing_python
echo No usable Python 3.9+ was found. Set Q3_PYTHON to the Python executable path.
set "Q3_EXIT=1"
goto cleanup

:caching
"%Q3_PYTHON%" %Q3_PY_ARGS% run_transposition_study.py
goto finished

:cache_confirmation
"%Q3_PYTHON%" %Q3_PY_ARGS% run_transposition_confirmation.py
goto finished

:sampling_confirmation
"%Q3_PYTHON%" %Q3_PY_ARGS% run_sampling_confirmation.py
goto finished

:correlation
"%Q3_PYTHON%" %Q3_PY_ARGS% run_correlation_study.py
goto finished

:capacity
"%Q3_PYTHON%" %Q3_PY_ARGS% run_capacity_study.py
goto finished

:extensions
"%Q3_PYTHON%" %Q3_PY_ARGS% run_extensions.py
goto finished

:finished
set "Q3_EXIT=%ERRORLEVEL%"

:cleanup
popd
if not "%Q3_NO_PAUSE%"=="1" pause
exit /b %Q3_EXIT%
