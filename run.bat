@echo off
set CONDAPATH=<path to Conda>
set ENVNAME=<envname>
set ENVPATH=%CONDAPATH%\envs\%ENVNAME%
call %CONDAPATH%/Scripts/activate.bat %ENVPATH%
python -m src.app
pause