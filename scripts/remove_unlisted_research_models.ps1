param([Parameter(Mandatory = $true)][string]$PlanPath)

# Native PowerShell only. Validate every absolute target before any recursive removal.
$repositoryRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$projectModelRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot 'data/models'))
$hfModelCacheRoot = [System.IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.cache/huggingface/hub'))
$hfLockRoot = [System.IO.Path]::GetFullPath((Join-Path $hfModelCacheRoot '.locks'))
$expectedRoots = @($projectModelRoot, $hfModelCacheRoot, $hfLockRoot)
$resolvedPlan = (Resolve-Path -LiteralPath $PlanPath -ErrorAction Stop).Path
$plan = Get-Content -LiteralPath $resolvedPlan -Raw -ErrorAction Stop | ConvertFrom-Json
$catalogPath = [System.IO.Path]::GetFullPath([string]$plan.catalog)
if (-not $catalogPath.StartsWith($repositoryRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Catalog must be inside this repository'
}
$catalogHash = (Get-FileHash -LiteralPath $catalogPath -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
if ($catalogHash -ne $plan.catalog_sha256) { throw 'Catalog changed after cleanup planning' }
$catalog = Get-Content -LiteralPath $catalogPath -Raw -ErrorAction Stop | ConvertFrom-Json
$inventory = Get-Content -LiteralPath $plan.download_inventory -Raw -ErrorAction Stop | ConvertFrom-Json
if ($inventory.catalog_sha256 -ne $catalogHash -or @($inventory.models).Count -ne @($catalog.models).Count) {
    throw 'Downloads are not complete for this catalog'
}
if (@($inventory.models | Where-Object { $_.verified -ne $true }).Count -ne 0) {
    throw 'An unverified model remains in download inventory'
}
$retainedModels = @($catalog.models) + @($catalog.support_models)
$keptLocalNames = @($retainedModels | ForEach-Object { $_.repo.Replace('/', '--') })
$keptCacheNames = @($keptLocalNames | ForEach-Object { 'models--' + $_ })
$validatedTargets = @()
foreach ($entry in @($plan.targets)) {
    $allowedRoot = [System.IO.Path]::GetFullPath([string]$entry.allowed_root)
    $targetPath = [System.IO.Path]::GetFullPath([string]$entry.path)
    if ($expectedRoots -notcontains $allowedRoot) { throw "Unexpected root: $allowedRoot" }
    if (-not $targetPath.StartsWith($allowedRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes the allowed root: $targetPath"
    }
    if ([System.IO.Path]::GetDirectoryName($targetPath) -ne $allowedRoot) {
        throw "Target must be a direct model folder: $targetPath"
    }
    $folderName = [System.IO.Path]::GetFileName($targetPath)
    if ($folderName -ne $entry.name) { throw 'Plan folder name mismatch' }
    if ($allowedRoot -eq $projectModelRoot) {
        if ($keptLocalNames -contains $folderName) { throw "Refusing to remove retained model: $folderName" }
    } else {
        if (-not $folderName.StartsWith('models--') -or $keptCacheNames -contains $folderName) {
            throw "Refusing to remove retained or non-model cache: $folderName"
        }
    }
    if (Test-Path -LiteralPath $targetPath) {
        $targetItem = Get-Item -LiteralPath $targetPath -Force -ErrorAction Stop
        $rootItem = Get-Item -LiteralPath $allowedRoot -Force -ErrorAction Stop
        if (-not $targetItem.PSIsContainer -or
            ($targetItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -or
            ($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
            throw "Refusing a linked/non-directory target or root: $targetPath"
        }
    }
    $validatedTargets += [pscustomobject]@{ path = $targetPath; kind = $entry.kind; bytes = $entry.bytes }
}

$auditPath = Join-Path (Split-Path -Parent $resolvedPlan) 'cleanup_result.json'
if (Test-Path -LiteralPath $auditPath) { throw 'Cleanup audit already exists; refusing overwrite' }
$deletedTargets = @()
$audit = [ordered]@{ status = 'executing'; catalog_sha256 = $catalogHash; deleted_targets = @(); failed_target = $null }
$audit | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $auditPath -Encoding utf8 -ErrorAction Stop
try {
    foreach ($target in $validatedTargets) {
        $existed = Test-Path -LiteralPath $target.path
        if ($existed) {
            Remove-Item -LiteralPath $target.path -Recurse -Force -ErrorAction Stop
        }
        if (Test-Path -LiteralPath $target.path) { throw "Target still exists: $($target.path)" }
        $deletedTargets += [pscustomobject]@{ path = $target.path; kind = $target.kind; existed = $existed; bytes = $(if ($existed) { $target.bytes } else { 0 }) }
        $audit.deleted_targets = $deletedTargets
        $audit | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $auditPath -Encoding utf8 -ErrorAction Stop
        Write-Output "Removed: $($target.path)"
    }
    $audit.status = 'completed'
    $audit.removed_bytes = ($deletedTargets | Measure-Object -Property bytes -Sum).Sum
} catch {
    $audit.status = 'failed'
    $audit.failed_target = [string]$_
    throw
} finally {
    $audit | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $auditPath -Encoding utf8 -ErrorAction Stop
}
