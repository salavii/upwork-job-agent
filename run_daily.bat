@echo off
REM run_daily.bat
REM
REM Wrapper for Windows Task Scheduler: runs the automatic job-fetch +
REM scoring pipeline (match_llm.py --daily) and appends its output to
REM daily_run_log.txt. Never applies to anything automatically - see
REM README.md's "Applying stays manual" section.
REM
REM Uses the real anaconda3 Python (not the WindowsApps alias stub) and
REM -X utf8 so job titles/descriptions with non-ASCII characters can
REM never crash the run over a console codepage mismatch.

cd /d D:\job\UPWORK-agent

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
    start "" /min "C:\Users\Lenovo\AppData\Local\Programs\Ollama\ollama.exe" serve
    timeout /t 15 /nobreak > nul
)

"C:\Users\Lenovo\anaconda3\python.exe" -X utf8 src\match_llm.py --daily >> daily_run_log.txt 2>&1
echo ===== Run finished %date% %time% ===== >> daily_run_log.txt
