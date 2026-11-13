param(
    [string]$EnvironmentPath,
    [string]$BasePython
)
$ErrorActionPreference = 'Stop'
$a1RepoRoot = Split-Path $PSScriptRoot -Parent
$a1Workspace = Split-Path $a1RepoRoot -Parent
if (-not $EnvironmentPath) { $EnvironmentPath = Join-Path $a1Workspace '.venv-a1' }
if (-not $BasePython) { $BasePython = Join-Path $a1Workspace '.venv\Scripts\python.exe' }
$a1Python = Join-Path $EnvironmentPath 'Scripts\python.exe'
if (Test-Path -LiteralPath $EnvironmentPath) {
    if (-not (Test-Path -LiteralPath (Join-Path $EnvironmentPath 'pyvenv.cfg'))) {
        throw 'Thu muc dich da co nhung khong phai venv. Chon EnvironmentPath khac.'
    }
} else {
    & $BasePython -m venv $EnvironmentPath
    if ($LASTEXITCODE -ne 0) { throw 'Khong tao duoc venv A1.' }
}

function Invoke-A1Pip([string[]]$PipArguments) {
    & $a1Python -X utf8 -m scripts.pip_with_dns_fallback @PipArguments
    if ($LASTEXITCODE -ne 0) { throw 'Cai thu vien A1 that bai.' }
}

Push-Location $a1RepoRoot
try {
    Invoke-A1Pip @('install', '--disable-pip-version-check', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                  'pip==25.3', 'setuptools==80.9.0', 'wheel==0.45.1')
    Invoke-A1Pip @('install', '--disable-pip-version-check', '--only-binary=:all:', '-r', 'requirements-a1-torch.txt')
    Invoke-A1Pip @('install', '--disable-pip-version-check', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                  '-r', 'requirements-a1.txt')
    & $a1Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Phu thuoc A1 chua hop le.' }
    Write-Output ('Moi truong A1: ' + $a1Python)
} finally {
    Pop-Location
}
