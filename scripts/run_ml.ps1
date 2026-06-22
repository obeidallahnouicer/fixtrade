param(
    [ValidateSet('full-pipeline', 'etl', 'train', 'predict', 'warm-cache', 'refresh-predictions')]
    [string]$Mode = 'full-pipeline',

    [string]$Symbol = 'BIAT',
    [int]$Days = 5,
    [switch]$Final,
    [switch]$Incremental,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Get-PythonCommand {
    $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $systemPython) {
        return $systemPython.Source
    }

    throw 'Python was not found. Create .venv or install Python on PATH.'
}

$python = Get-PythonCommand

function Invoke-PythonCommand {
    param([string[]]$Arguments)

    Write-Host "> $python $($Arguments -join ' ')"
    if (-not $DryRun) {
        & $python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $LASTEXITCODE"
        }
    }
}

Write-Host "FixTrade ML automation"
Write-Host "Mode: $Mode"
Write-Host "Python: $python"
Write-Host ""

switch ($Mode) {
    'full-pipeline' { Invoke-PythonCommand -Arguments @('-m', 'prediction', 'scheduler', '--run', 'full_pipeline') }
    'etl' {
        if ($Incremental) {
            Invoke-PythonCommand -Arguments @('-m', 'prediction', 'etl', '--incremental')
        } else {
            Invoke-PythonCommand -Arguments @('-m', 'prediction', 'etl')
        }
    }
    'train' {
        if ($Final) {
            Invoke-PythonCommand -Arguments @('-m', 'prediction', 'train', '--final')
        } else {
            Invoke-PythonCommand -Arguments @('-m', 'prediction', 'train')
        }
    }
    'predict' { Invoke-PythonCommand -Arguments @('-m', 'prediction', 'predict', '--symbol', $Symbol, '--days', "$Days") }
    'warm-cache' { Invoke-PythonCommand -Arguments @('-m', 'prediction', 'warm-cache') }
    'refresh-predictions' { Invoke-PythonCommand -Arguments @('-m', 'prediction', 'scheduler', '--run', 'refresh_predictions') }
}
