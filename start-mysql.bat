@echo off

:: Ensure that the cluster is running!
:: Use \js mode inside of the mysqlsh.exe (mysqlsh clusteradmin@localhost)

@echo off
:: Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: Your script continues below
echo Running MySQL Router as Administrator

mysqlrouter -c "C:\Program Files\MySQL\MySQL Server 8.4\mysqlrouter.conf"