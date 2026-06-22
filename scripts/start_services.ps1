Param(
    [int]$TimeoutSeconds = 300
)

$LogDir = Join-Path (Get-Location) 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

$services = @('postgres','redis','auth-service','ml-service','api','genai-service','scraper','etl-worker')
Write-Output "Starting services: $($services -join ', ')" | Tee-Object -FilePath "$LogDir\startup.log"
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

Write-Output "Running ETL loader inside etl-worker (if available)" | Tee-Object -FilePath "$LogDir\startup.log" -Append
try {
    docker compose exec -T etl-worker python scripts/load_fallback_articles.py 2>&1 | Tee-Object -FilePath "$LogDir\etl_run.log"
} catch {
    Write-Output "ETL run failed: $_" | Tee-Object -FilePath "$LogDir\startup.log" -Append
}

if (Test-Path './scraped_fallback.jsonl') {
    Copy-Item './scraped_fallback.jsonl' -Destination './data/' -Force
    Write-Output 'copied scraped_fallback.jsonl to data/' | Tee-Object -FilePath "$LogDir\startup.log" -Append
}

Write-Output "Running smoke tests" | Tee-Object -FilePath "$LogDir\startup.log" -Append
python scripts/smoke_test.py "$LogDir\smoke_test.json" 2>&1 | Tee-Object -FilePath "$LogDir\startup.log" -Append

Write-Output "Done. Logs in $LogDir" | Tee-Object -FilePath "$LogDir\startup.log" -Append
