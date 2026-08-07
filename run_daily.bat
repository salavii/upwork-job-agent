@echo off
REM run_daily.bat
REM
REM Wrapper for Windows Task Scheduler: runs the automatic job-fetch +
REM scoring pipeline (match_llm.py --daily) and appends its output to
REM daily_run_log.txt. Never applies to anything automatically - see
REM README.md's "Applying stays manual" section.
REM
REM Runs with -X utf8 so job titles/descriptions with non-ASCII characters
REM can never crash the run over a console codepage mismatch.
REM
REM Two optional environment variables, so this file contains no paths
REM specific to one machine:
REM
REM   JOB_AGENT_PYTHON  Python interpreter to use. Defaults to `python`.
REM                     Set this to a full interpreter path if `python` on
REM                     PATH resolves to the Windows Store alias stub rather
REM                     than a real install, which fails silently under Task
REM                     Scheduler.
REM   OLLAMA_EXE        Ollama executable. Defaults to the standard per-user
REM                     install location.

REM Run from the repo root regardless of where the scheduler invokes this.
cd /d "%~dp0"

if not defined JOB_AGENT_PYTHON set "JOB_AGENT_PYTHON=python"
if not defined OLLAMA_EXE set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"

echo. >> daily_run_log.txt
echo ===== Run started %date% %time% ===== >> daily_run_log.txt

REM Scoring needs Ollama's local server. An unattended, scheduled run
REM can't assume it's already up (e.g. right after a reboot, or if it
REM was closed since the last run) - a real run on 2026-07-25 crashed
REM here with a connection-refused error. Check it first, and start it
REM if needed, instead of letting match_llm.py fail outright.
curl -s -o nul -w "%%{http_code}" http://localhost:11434/api/tags > "%TEMP%\ollama_check.txt" 2>nul
set /p OLLAMA_STATUS=<"%TEMP%\ollama_check.txt"
if not "%OLLAMA_STATUS%"=="200" (
    echo Ollama not responding - starting it... >> daily_run_log.txt
    start "" /min "%OLLAMA_EXE%" serve
    timeout /t 15 /nobreak > nul
)

"%JOB_AGENT_PYTHON%" -X utf8 src\match_llm.py --daily >> daily_run_log.txt 2>&1
echo ===== Run finished %date% %time% ===== >> daily_run_log.txt
