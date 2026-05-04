$lines = Get-Content coverage/lcov.info
$inTarget = $false
$recordLines = @()

foreach ($line in $lines) {
  if ($line -match 'SF:src.*speciation.sharing.utils.ts') {
    $inTarget = $true
    $recordLines = @($line)
  } elseif ($inTarget) {
    $recordLines += $line
    if ($line -eq 'end_of_record') {
      break
    }
  }
}

# Find all BRDA lines (branch data) and identify which have 0 hits
$uncoveredBranches = @()
foreach ($line in $recordLines) {
  if ($line -match '^BRDA:(\d+),(\d+),(\d+),(\d+)$') {
    $lineNum = [int]$matches[1]
    $blockId = [int]$matches[2]
    $branchId = [int]$matches[3]
    $hitCount = [int]$matches[4]
    if ($hitCount -eq 0) {
      $uncoveredBranches += $line
    }
  }
}

Write-Host "Uncovered branches in speciation.sharing.utils.ts:"
Write-Host "================================================="
foreach ($branch in $uncoveredBranches) {
  Write-Host $branch
}
$count = $uncoveredBranches.Count
Write-Host "Total uncovered: $count branches"
