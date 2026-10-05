@echo off
echo Starting EV Battery Thermal Management Flask App...
py -3.13 app.py
if %ERRORLEVEL% NEQ 0 (
    python app.py
)
pause
