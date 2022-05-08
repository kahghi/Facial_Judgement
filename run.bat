@echo off
set CONDAPATH=C:\Users\KahGhi\anaconda3
set ENVNAME=aiap_wk8
set ENVPATH=%CONDAPATH%\envs\%ENVNAME%
call %CONDAPATH%/Scripts/activate.bat %ENVPATH%
python -m src.app
pause