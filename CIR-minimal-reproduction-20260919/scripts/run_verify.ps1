$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

python (Join-Path $PSScriptRoot "verify_rankings.py")
if ($LASTEXITCODE -ne 0) { throw "Ranking verification failed." }

$Required = @()
foreach ($Category in @("dress", "shirt", "toptee")) {
    $Required += Join-Path $Root "assets\$Category\${Category}_val_query_features.pt"
    $Required += Join-Path $Root "assets\$Category\${Category}_val_image_features.pt"
    $Required += Join-Path $Root "assets\$Category\${Category}_val_topk.pt"
    $Required += Join-Path $Root "checkpoints\reranker\$Category\${Category}_best.pt"
}

$Missing = $Required | Where-Object { -not (Test-Path -LiteralPath $_) -or (Get-Item -LiteralPath $_).Length -eq 0 }
if ($Missing.Count -eq 0) {
    python (Join-Path $PSScriptRoot "evaluate_cached_reranker.py") --device auto
    if ($LASTEXITCODE -ne 0) { throw "Cached-feature evaluation failed." }
} else {
    Write-Warning "Cached-feature evaluation skipped because one or more optional binary assets are absent."
}
