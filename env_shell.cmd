@echo off
call env_config
call activate %PROJECT_ENV%
call conda env list
cmd
