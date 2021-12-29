@echo off
call env_config
call conda create -n %PROJECT_ENV% 
call activate %PROJECT_ENV%
call conda env list
