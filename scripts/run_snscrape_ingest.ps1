# Simple wrapper to run the snscrape connector periodically (Windows PowerShell)
# Usage: .\run_snscrape_ingest.ps1 -Query "#handmade OR #kangra" -District kangra -IntervalMinutes 5
param(
    [Parameter(Mandatory=$true)]
    [string]$Query,

    [Parameter(Mandatory=$true)]
    [string]$District,

    [int]$IntervalMinutes = 5
)

Write-Host "Starting snscrape connector for query: $Query (district: $District)"
while ($true) {
    python .\scripts\connector_snscrape_ingest.py --query "$Query" --district $District --max 200 --batch 20
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
