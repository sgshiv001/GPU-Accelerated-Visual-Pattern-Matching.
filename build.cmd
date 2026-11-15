@echo off
setlocal
pushd "%~dp0" || exit /b 1

where nvcc.exe >nul 2>&1
if errorlevel 1 (
    echo nvcc is not on PATH. Add the CUDA Toolkit bin directory and retry. 1>&2
    goto :fail
)

where cl.exe >nul 2>&1
if not errorlevel 1 goto :compile

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_ROOT=%%I"
)
if not defined VS_ROOT set "VS_ROOT=%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools"
if not exist "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" (
    echo Open an x64 Native Tools Command Prompt for a CUDA-supported Visual Studio installation and retry. 1>&2
    goto :fail
)
call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" || goto :fail

:compile
if not exist build mkdir build || goto :fail
if /i "%~1"=="vision" goto :vision
nvcc -std=c++17 -O3 -lineinfo -arch=native -Xcompiler=/EHsc src\gemm.cu -lcublas -o build\gemm.exe || goto :fail
popd
exit /b 0

:vision
nvcc -std=c++17 -O3 -lineinfo -arch=native -Xcompiler=/EHsc -shared vision\inference.cu -o build\vision_cuda.dll || goto :fail
popd
exit /b 0

:fail
popd
exit /b 1
