$lines = Get-Content coverage/lcov.info
$files = @{}
$currentFile = $null
$brf = 0
$brh = 0

foreach ($line in $lines) {
  if ($line -match '^SF:(.+)$') {
    $currentFile = $matches[1]
  } elseif ($line -match '^BRF:(\d+)$') {
    $brf = [int]$matches[1]
  } elseif ($line -match '^BRH:(\d+)$') {
    $brh = [int]$matches[1]
    if ($currentFile -and $brf -gt 0) {
      $pct = [math]::Round(($brh / $brf) * 100, 2)
      $files[$currentFile] = @{'brf'=$brf; 'brh'=$brh; 'pct'=$pct}
    }
  }
}

# Sort by percentage and show lowest 20
$files.GetEnumerator() | Sort-Object { $_.Value.pct } | Select-Object -First 20 | ForEach-Object {
  $name = $_.Key
  $pct = $_.Value.pct
  $brf = $_.Value.brf
  $brh = $_.Value.brh
  Write-Host ("{0:000}% ({1:D2}/{2:D2}) - {3}" -f [int]$pct, $brh, $brf, $name)
}
