@echo off
rem Runs scripts/pod.sh with Git Bash (not WSL). Usage: pod up | wait | status | down
set "GB=%ProgramFiles%\Gitinash.exe"
if not exist "%GB%" set "GB=%LocalAppData%\Programs\Gitinash.exe"
if not exist "%GB%" ( echo Git Bash not found - install Git for Windows & exit /b 1 )
"%GB%" "%~dp0scripts/pod.sh" %*
