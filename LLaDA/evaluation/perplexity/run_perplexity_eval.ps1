# run_perplexity_eval.ps1
# Perplexity evaluation script for LLaDA model outputs
#
# Usage:
#   .\run_perplexity_eval.ps1                          # Use default promptfoo_results.json
#   .\run_perplexity_eval.ps1 -ResultsPath "../promptfoo/promptfoo_results.json"
#   .\run_perplexity_eval.ps1 -GenerateHtml

param(
    [string]$ResultsPath = "../promptfoo/promptfoo_results.json",
    [switch]$GenerateHtml
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

$Colors = @{
    Cyan = "Cyan"
    Green = "Green"
    Yellow = "Yellow"
    Red = "Red"
}

function Write-Status {
    param([string]$Message, [string]$Color = $Colors.Cyan)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Message" -ForegroundColor $Color
}

# Check Python
Write-Status "Checking Python..." $Colors.Cyan
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $python) {
    Write-Status "ERROR: Python not found in PATH" $Colors.Red
    exit 1
}

# Resolve results path
$fullResultsPath = Join-Path $RootDir $ResultsPath
if (-not (Test-Path $fullResultsPath)) {
    # Try relative to script directory
    $fullResultsPath = Join-Path $ScriptDir $ResultsPath
}
if (-not (Test-Path $fullResultsPath)) {
    Write-Status "ERROR: Results file not found. Searched at:" $Colors.Red
    Write-Host "  - $fullResultsPath"
    Write-Host "  $(Join-Path $RootDir $ResultsPath)"
    exit 1
}

Write-Status "Using results file: $fullResultsPath" $Colors.Green

# Run perplexity calculation
Push-Location $ScriptDir

try {
    $outputFile = "perplexity_results.json"
    $htmlFile = if ($GenerateHtml) { "perplexity_report.html" } else { $null }

    Write-Status "Starting perplexity calculation..." $Colors.Cyan
    Write-Status "This will download GPT-2 on first run (~500MB)" $Colors.Yellow

    $args = @(
        "calculate_perplexity.py",
        "--input", $fullResultsPath,
        "--output", $outputFile
    )

    if ($htmlFile) {
        $args += @("--html", $htmlFile)
    }

    & $python.Source @args

    if ($LASTEXITCODE -ne 0) {
        Write-Status "Perplexity evaluation failed" $Colors.Red
        exit 1
    }

    # Display results
    if (Test-Path $outputFile) {
        $results = Get-Content $outputFile | ConvertFrom-Json
        Write-Status "" $Colors.Green
        Write-Status "=== PERPLEXITY RESULTS ===" $Colors.Green
        Write-Host "  Average Perplexity: $($results.average_perplexity)" -ForegroundColor White
        Write-Host "  Samples Evaluated:  $($results.num_samples)" -ForegroundColor White

        $ppl = $results.average_perplexity
        if ($ppl -lt 20) {
            Write-Host "  Quality: EXCELLENT (< 20)" -ForegroundColor Green
        } elseif ($ppl -lt 50) {
            Write-Host "  Quality: GOOD (< 50)" -ForegroundColor Yellow
        } else {
            Write-Host "  Quality: HIGH (> 50)" -ForegroundColor Red
        }

        if ($htmlFile -and (Test-Path $htmlFile)) {
            Write-Status "HTML report generated: $htmlFile" $Colors.Green

            $openReport = Read-Host "Open report in browser? (Y/n)"
            if ($openReport -eq '' -or $openReport -eq 'y' -or $openReport -eq 'Y') {
                Start-Process (Join-Path $ScriptDir $htmlFile)
            }
        }
    }

} catch {
    Write-Status "Error running perplexity evaluation: $_" $Colors.Red
} finally {
    Pop-Location
}

Write-Status "Done!" $Colors.Green
