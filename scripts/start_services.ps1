Param(
    [int]$TimeoutSeconds = 300
)

$LogDir = Join-Path (Get-Location) 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$services = @('postgres','redis','auth-service','ml-service','api','genai-service','scraper','etl-worker')
Write-Output "Starting services: $($services -join ', ')" | Tee-Object -FilePath "$LogDir\startup.log"

$lms = Get-Command lms -ErrorAction SilentlyContinue
if ($lms) {
    $lmStatus = & lms server status 2>&1 | Out-String
    if ($lmStatus -notmatch 'running') {
        & lms server start 2>&1 | Tee-Object -FilePath "$LogDir\startup.log" -Append
    }
    try {
        $models = Invoke-RestMethod -Uri 'http://127.0.0.1:1234/v1/models' -TimeoutSec 10
        $localLlama = $models.data.id | Where-Object { $_ -match 'llama' } | Select-Object -First 1
        if ($localLlama) {
            Write-Output "LM Studio fallback is ready with $localLlama" | Tee-Object -FilePath "$LogDir\startup.log" -Append
        } else {
            Write-Output 'LM Studio is running; no local Llama model is available yet.' | Tee-Object -FilePath "$LogDir\startup.log" -Append
        }
    } catch {
        Write-Output 'LM Studio was found but its API did not become ready on port 1234.' | Tee-Object -FilePath "$LogDir\startup.log" -Append
    }
} else {
    Write-Output 'LM Studio CLI not found; portfolio explanations will use fallback mode.' | Tee-Object -FilePath "$LogDir\startup.log" -Append
}

docker compose up -d --build $services 2>&1 | Tee-Object -FilePath "$LogDir\startup.log" -Append

function Wait-ForUrl {
    param($url, $name, $timeoutSeconds)
    $start = Get-Date
    while ((Get-Date) - $start).TotalSeconds -lt $timeoutSeconds {
        try {
            $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            Write-Output "$name is available" | Tee-Object -FilePath "$LogDir\startup.log" -Append
            return $true
        } catch {
            Start-Sleep -Seconds 2
        }
    }
    Write-Output "Timeout waiting for $name" | Tee-Object -FilePath "$LogDir\startup.log" -Append
    return $false
}

Wait-ForUrl 'http://127.0.0.1:5432' 'postgres' 120 | Out-Null
Wait-ForUrl 'http://127.0.0.1:6379' 'redis' 60 | Out-Null
Wait-ForUrl 'http://127.0.0.1:8000/api/v1/health' 'api' 120 | Out-Null
Wait-ForUrl 'http://127.0.0.1:8001/api/v1/health' 'ml-service' 120 | Out-Null
Wait-ForUrl 'http://127.0.0.1:8002/api/v1/health' 'auth-service' 120 | Out-Null
Wait-ForUrl 'http://127.0.0.1:8003/api/v1/health' 'genai-service' 120 | Out-Null

Write-Output "Automation worker is running ETL, enrichment, predictions, and anomalies in the background." | Tee-Object -FilePath "$LogDir\startup.log" -Append

if (Test-Path './scraped_fallback.jsonl') {
    Copy-Item './scraped_fallback.jsonl' -Destination './data/' -Force
    Write-Output 'copied scraped_fallback.jsonl to data/' | Tee-Object -FilePath "$LogDir\startup.log" -Append
}

Write-Output "Running smoke tests" | Tee-Object -FilePath "$LogDir\startup.log" -Append
python scripts/smoke_test.py "$LogDir\smoke_test.json" 2>&1 | Tee-Object -FilePath "$LogDir\startup.log" -Append

Write-Output "Done. Logs in $LogDir" | Tee-Object -FilePath "$LogDir\startup.log" -Append
