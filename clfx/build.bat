@echo off
REM ──────────────────────────────────────────────────────────────────────────
REM  CLFX Windows Native Build Script (MSYS2 / MinGW-w64)
REM
REM  Requirements:
REM    1. Install MSYS2 from https://www.msys2.org/  (default: C:\msys64)
REM    2. Open MSYS2 MINGW64 shell and run:
REM         pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencl-headers
REM    3. Then run THIS script from any terminal.
REM ──────────────────────────────────────────────────────────────────────────

SET MSYS2_PATH=C:\msys64\mingw64\bin
SET GCC=%MSYS2_PATH%\gcc.exe

IF NOT EXIST "%GCC%" (
    echo ERROR: MinGW-w64 GCC not found at %MSYS2_PATH%
    echo Please install MSYS2 from https://www.msys2.org/ and run:
    echo   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-opencl-headers
    exit /b 1
)

echo Building CLFX with MinGW-w64 GCC ...

"%GCC%" -Iinclude main.c -o clfx.exe ^
    -lOpenCL -lm -std=c99 ^
    -D_POSIX_C_SOURCE=200112L ^
    -D_WIN32_WINNT=0x0601

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ Build succeeded: clfx.exe
    echo   No external DLLs required - fully native Windows binary.
    clfx.exe --info
) ELSE (
    echo.
    echo ✗ Build failed. Check output above for errors.
    exit /b 1
)
