# From the project root, run .\scripts\run.ps1 with the project environment active.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot 'data'
$RunName = 'sonics_parts'
$Detectors = @('sonics-alpha-5s', 'sonics-alpha-120s')
$Python = 'python'
$Steps = @(
    'waveform fgsm --epsilon 0.001 0.002 0.005 0.01'
    'waveform pgd --epsilon 0.001 0.002 0.005 0.01 --alpha 0.001 --steps 20'
    'waveform cw --c 1 --steps 100'
    'waveform noise_baseline --snr-db 45 33 28 24'
    'mel fgsm --epsilon 0.02 0.05 0.1'
    'mel pgd --epsilon 0.02 0.05 0.1 --alpha 0.01 --steps 20'
    'mel cw --c 1 --steps 25'
)

function Invoke-App([string[]]$Arguments) {
    & $Python -m src.main @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Arguments" }
}

Push-Location $ProjectRoot
try {
    foreach ($part in @('sonics-part1', 'sonics-part2')) {
        $directory = Join-Path $DataDir "fake\$part"
        if (-not (Test-Path -LiteralPath $directory)) { throw "Missing dataset: $directory" }
        $partSample = Get-ChildItem -LiteralPath $directory -File -Recurse |
            Where-Object { $_.Extension -in '.wav', '.mp3' } |
            Select-Object -First 1
        if (-not $partSample) { throw "No WAV/MP3 files in: $directory" }
    }

    $sample = Get-ChildItem -LiteralPath (Join-Path $DataDir 'fake') -File -Recurse |
        Where-Object { $_.Extension -in '.wav', '.mp3' } |
        Sort-Object FullName |
        Select-Object -First 1
    if (-not $sample) { throw "No WAV/MP3 files under $DataDir\fake" }

    foreach ($detector in $Detectors) {
        $results = Join-Path $ProjectRoot "results\$detector\$RunName"
        if (Test-Path -LiteralPath $results) { throw "Results already exist: $results" }
    }

    Invoke-App @('check-env')
    New-Item -ItemType Directory -Path (Join-Path $ProjectRoot 'results') -Force | Out-Null
    foreach ($detector in $Detectors) {
        $results = Join-Path $ProjectRoot "results\$detector\$RunName"
        New-Item -ItemType Directory -Path (Split-Path $results) -Force | Out-Null
        New-Item -ItemType Directory -Path $results | Out-Null
        $common = @('--detector', $detector, '--data-dir', $DataDir, '--results-dir', $results)

        Invoke-App (@('infer', '--detector', $detector, '--sample-file', $sample.FullName))
        Invoke-App (@('check-gradient', '--detector', $detector, '--sample-file', $sample.FullName))
        foreach ($representation in @('waveform', 'mel')) {
            Invoke-App (@('baseline', '--representation', $representation) + $common)
        }
        foreach ($step in $Steps) {
            $parts = $step -split ' '
            $arguments = @('attack', '--representation', $parts[0], '--attack', $parts[1]) +
                $parts[2..($parts.Length - 1)] + $common
            Invoke-App $arguments
        }
    }
}
finally {
    Pop-Location
}
