# run_evaluation.ps1
# Automated evaluation script for LLaDA model using promptfoo
#
# Usage:
#   .\run_evaluation.ps1
#   .\run_evaluation.ps1 -SkipServer
#   .\run_evaluation.ps1 -SkipEval -JustReport
#
# IMPORTANT: Run this script from the LLaDA root directory, not from evaluation/promptfoo/

param(
    [switch]$SkipServer,
    [switch]$SkipEval,
    [switch]$JustReport
)

# Configuration
$ServerPort = 5000
$HealthUrl  = "http://127.0.0.1:$ServerPort/health"
$ServerUrl  = "http://127.0.0.1:$ServerPort/generate"
$ResultsFile = "promptfoo_results.json"
$ReportFile  = "evaluation_report.html"
$Timeout = 3600000  # 1 hour in milliseconds

$OllamaUrl = "http://127.0.0.1:11434"

# Determine our location
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$PromptfooDir = $ScriptDir
$LladaRoot   = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$ServerScript = Join-Path $LladaRoot "serve_llada.py"

# Colors
$Red    = "Red"
$Green  = "Green"
$Yellow = "Yellow"
$Cyan   = "Cyan"

function Write-Status {
    param([string]$Message, [string]$Color = $Cyan)
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $Message" -ForegroundColor $Color
}

function Test-ServerRunning {
    # Uses the lightweight /health endpoint — no inference, returns instantly.
    try {
        $response = Invoke-WebRequest -Uri $HealthUrl -Method GET -TimeoutSec 5 -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Test-OllamaRunning {
    # Check if Ollama is running
    try {
        $response = Invoke-WebRequest -Uri "$OllamaUrl/" -Method GET -TimeoutSec 5 -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

function Start-LladaServer {
    Write-Status "Starting LLaDA server..." $Yellow

    if (-not (Test-Path $ServerScript)) {
        Write-Status "ERROR: serve_llada.py not found at: $ServerScript" $Red
        exit 1
    }

    # Check if Python is available
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        $python = Get-Command python3 -ErrorAction SilentlyContinue
    }
    if (-not $python) {
        Write-Status "ERROR: Python not found in PATH" $Red
        exit 1
    }

    # Start server in background from LLaDA root
    $serverJob = Start-Job -ScriptBlock {
        param($py, $script, $workingDir)
        Set-Location $workingDir
        & $py $script
    } -ArgumentList $python.Source, $ServerScript, $LladaRoot

    Write-Status "Waiting for server to initialize (this may take a few minutes)..." $Yellow

    # Wait up to 10 minutes for model to load — loading 6 shards takes ~4-5 min
    $maxAttempts = 300  # 300 * 2s = 10 minutes
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        Start-Sleep -Seconds 2
        $attempt++

        if (Test-ServerRunning) {
            Write-Status "Server is ready!" $Green
            return $serverJob
        }

        # Check if the job has crashed — only bail on real Python fatal errors,
        # not on the harmless "weights not tied" warning which also contains "Error".
        $jobStatus = Receive-Job -Job $serverJob -Keep
        $isFatal = ($jobStatus -match "Traceback \(most recent call last\)") -or
                   ($jobStatus -match "OutOfMemoryError") -or
                   ($jobStatus -match "ModuleNotFoundError") -or
                   ($jobStatus -match "ImportError") -or
                   ($jobStatus -match "RuntimeError")
        if ($isFatal) {
            Write-Status "Server failed to start. Last output:" $Red
            Write-Host $jobStatus
            Stop-Job  -Job $serverJob -ErrorAction SilentlyContinue
            Remove-Job -Job $serverJob -ErrorAction SilentlyContinue
            exit 1
        }

        if ($attempt % 10 -eq 0) {
            Write-Host "." -NoNewline -ForegroundColor $Yellow
        }
    }

    Write-Host ""
    Write-Status "ERROR: Server failed to start within 10 minutes" $Red
    Stop-Job  -Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job -Job $serverJob -ErrorAction SilentlyContinue
    exit 1
}

# Main execution
Write-Status "=== LLaDA Evaluation Runner ===" $Cyan
Write-Status "LLaDA Root: $LladaRoot" $Cyan
Write-Status "Promptfoo Dir: $PromptfooDir" $Cyan
Write-Host ""

# Check Ollama is running (required for judge)
Write-Status "Checking Ollama..." $Cyan
if (-not (Test-OllamaRunning)) {
    Write-Status "ERROR: Ollama is not running at $OllamaUrl" $Red
    Write-Status "Please start Ollama and ensure llama3.1:8b is pulled:" $Red
    Write-Status "  ollama pull llama3.1:8b" $Yellow
    exit 1
}
Write-Status "Ollama is running" $Green

# Set timeout
$env:PROMPTFOO_REQUEST_TIMEOUT_MS = $Timeout.ToString()
Write-Status "Timeout set to $($Timeout / 1000 / 60) minutes" $Green

# Check Node.js and promptfoo
Write-Status "Checking dependencies..." $Cyan
$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    Write-Status "ERROR: Node.js not found. Please install Node.js from https://nodejs.org/" $Red
    exit 1
}
Write-Status "Node.js found: $(node --version)" $Green

# Check if config exists
$configPath = Join-Path $PromptfooDir "promptfooconfig.yaml"
if (-not (Test-Path $configPath)) {
    Write-Status "ERROR: promptfooconfig.yaml not found at: $configPath" $Red
    exit 1
}
Write-Status "Config found: $configPath" $Green

# Start server if needed
$serverJob = $null
if (-not $SkipServer -and -not $JustReport) {
    if (Test-ServerRunning) {
        Write-Status "Server is already running (detected via /health)" $Green
    } else {
        $serverJob = Start-LladaServer
    }
} elseif ($JustReport) {
    Write-Status "Skipping server startup (JustReport mode)" $Yellow
} else {
    Write-Status "Skipping server startup (SkipServer flag)" $Yellow
    if (-not (Test-ServerRunning)) {
        Write-Status "WARNING: Server does not appear to be running!" $Red
        $continue = Read-Host "Continue anyway? (y/N)"
        if ($continue -ne 'y' -and $continue -ne 'Y') {
            exit 1
        }
    }
}

# Run evaluation
if (-not $JustReport) {
    Write-Host ""
    Write-Status "=== Starting Evaluation ===" $Cyan
    Write-Status "This will take approximately 15-30 minutes (LLaDA generates then Ollama judges)" $Yellow
    Write-Status "Running sequentially to avoid VRAM collision between LLaDA and Ollama" $Cyan
    Write-Host ""

    # Change to promptfoo directory so relative paths in config work
    Push-Location $PromptfooDir

    try {
        # Run with concurrency=1 to ensure sequential execution (avoid VRAM collision)
        npx promptfoo eval -o $ResultsFile --max-concurrency 1

        if ($LASTEXITCODE -ne 0) {
            Write-Status "Evaluation completed with warnings (exit code: $LASTEXITCODE)" $Yellow
        } else {
            Write-Status "Evaluation completed successfully!" $Green
        }
    } catch {
        Write-Status "ERROR during evaluation: $_" $Red
    } finally {
        Pop-Location
    }
}

# Generate report
Write-Host ""
Write-Status "=== Generating Report ===" $Cyan

$resultsPath = Join-Path $PromptfooDir $ResultsFile
if (-not (Test-Path $resultsPath)) {
    Write-Status "ERROR: Results file not found: $resultsPath" $Red
    Write-Status "Evaluation may have failed. Check the output above." $Red

    if ($serverJob) {
        Stop-Job  -Job $serverJob -ErrorAction SilentlyContinue
        Remove-Job -Job $serverJob -ErrorAction SilentlyContinue
    }
    exit 1
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    $python = Get-Command python3 -ErrorAction SilentlyContinue
}

if ($python) {
    Push-Location $PromptfooDir
    & $python.Source generate_report.py
    Pop-Location

    $reportPath = Join-Path $PromptfooDir $ReportFile
    if (Test-Path $reportPath) {
        Write-Status "Report generated: $reportPath" $Green

        $openReport = Read-Host "Open report in browser? (Y/n)"
        if ($openReport -eq '' -or $openReport -eq 'y' -or $openReport -eq 'Y') {
            Start-Process $reportPath
        }
    } else {
        Write-Status "WARNING: Report file was not created" $Yellow
    }
} else {
    Write-Status "WARNING: Python not found, cannot generate HTML report" $Yellow
    Write-Status "You can still view results with: npx promptfoo view" $Cyan
}

# Cleanup
if ($serverJob) {
    Write-Host ""
    $stopServer = Read-Host "Stop the LLaDA server? (Y/n)"
    if ($stopServer -eq '' -or $stopServer -eq 'y' -or $stopServer -eq 'Y') {
        Stop-Job  -Job $serverJob
        Remove-Job -Job $serverJob
        Write-Status "Server stopped" $Green
    } else {
        Write-Status "Server left running (stop manually with Stop-Job)" $Yellow
    }
}

Write-Host ""
Write-Status "=== Evaluation Complete ===" $Green