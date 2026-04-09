# Load .env (KEY=VALUE), then run CMake with a Visual Studio generator by default
# so NMake/cl.exe discovery issues are avoided on Windows.
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$EnvFile = Join-Path $Root ".env"

if (-not (Test-Path $EnvFile)) {
    $example = Join-Path $Root ".env.example"
    if (Test-Path $example) {
        Copy-Item $example $EnvFile
        Write-Host "Created .env from .env.example - edit CMAKE_CUDA_ARCHITECTURES and optional compiler paths."
    }
    else {
        Write-Error ".env missing and .env.example not found."
    }
}

Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    if ($line -match '^([^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $val = $matches[2].Trim()
        Set-Item -Path "Env:$key" -Value $val
    }
}

$buildDir = Join-Path $Root "build"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

$gen = $env:CMAKE_GENERATOR
if (-not $gen) { $gen = "Visual Studio 18 2026" }
$plat = $env:CMAKE_GENERATOR_PLATFORM
if (-not $plat) { $plat = "x64" }
$arch = $env:CMAKE_CUDA_ARCHITECTURES
if (-not $arch) { $arch = "80" }

$cmakeArgs = @(
    "-S", $Root,
    "-B", $buildDir,
    "-G", $gen
)
if ($gen -match "Visual Studio") {
    $cmakeArgs += "-A", $plat
    # CMake 4.x + VS generator needs the CUDA MSBuild toolset (-T cuda=...), not only nvcc on PATH.
    $cudaRoot = $env:CUDA_PATH
    if (-not $cudaRoot) {
        $cudaDirs = @(Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue | Sort-Object { $_.Name } -Descending)
        if ($cudaDirs.Count -gt 0) {
            $cudaRoot = $cudaDirs[0].FullName
        }
    }
    if ($cudaRoot) {
        $cudaFwd = ($cudaRoot -replace '\\', '/')
        $cmakeArgs += "-T", "cuda=$cudaFwd"
    }
    else {
        Write-Warning "CUDA_PATH not set and no toolkit under Program Files; Visual Studio CUDA toolset (-T cuda=...) may be missing. Set CUDA_PATH or install CUDA."
    }
}
$cmakeArgs += "-D", "CMAKE_CUDA_ARCHITECTURES=$arch"

if ($env:CMAKE_CUDA_COMPILER) {
    $cmakeArgs += "-D", "CMAKE_CUDA_COMPILER=$($env:CMAKE_CUDA_COMPILER)"
}
if ($env:CMAKE_CXX_COMPILER) {
    $cmakeArgs += "-D", "CMAKE_CXX_COMPILER=$($env:CMAKE_CXX_COMPILER)"
}

Write-Host ('cmake ' + ($cmakeArgs -join ' '))
& cmake @cmakeArgs
