@echo off
call env_config

title jupyter @ %PROJECT_ENV%
call activate %PROJECT_ENV%

set PROJECT_ROOT=%cd%
set PYTHONPATH=%PYTHONPATH%;%cd%/src
call jupyter lab
