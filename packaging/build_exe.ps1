# Usage: pwsh packaging/build_exe.ps1
# The script wraps PyInstaller so the executable can be rebuilt easily.

param(
    [string]$VirtualEnv = ".\.venv",
    [string]$SpecFile = "packaging\gui_main.spec"
)

$pyinstaller = Join-Path -Path $VirtualEnv -ChildPath "Scripts\pyinstaller.exe"
if (-not (Test-Path $pyinstaller)) {
    Write-Host "PyInstaller not found in $VirtualEnv. Installing it..."
    & "$VirtualEnv\Scripts\python.exe" -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to install PyInstaller."
    }
}

Write-Host "Building executable with $SpecFile"
& $pyinstaller $SpecFile --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

Write-Host "Executable available in the dist folder."
