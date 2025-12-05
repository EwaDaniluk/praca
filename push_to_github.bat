@echo off
REM Push project to GitHub. Run this after installing Git (and optionally GitHub CLI).

REM Change to script directory
cd /d %~dp0

REM Initialize repo and set a local committer if not set
git init -b main
git config user.name "Praca Owner"
git config user.email "noreply@example.com"
git add .
git commit -m "Initial commit: project praca"

REM If GitHub CLI is installed, create and push the repo automatically
where gh >nul 2>&1
if %ERRORLEVEL%==0 (
  gh repo create praca --public --source=. --remote=origin --push --confirm
) else (
  echo GitHub CLI not found.
  echo Create a public repo named 'praca' on github.com, then run these commands:
  echo.
  echo git remote add origin https://github.com/USERNAME/praca.git
  echo git branch -M main
  echo git push -u origin main
)

pause
