# generate_tree.ps1

$OutputFile = "project_structure.txt"

$ExcludeDirs  = @('.git', '.venv', 'venv', '__pycache__', '.idea', '.vscode', 'node_modules', '.pytest_cache', 'dist', 'build')
$ExcludeFiles = @('*.pyc', '*.pyo', '.DS_Store', 'project_structure.txt', 'generate_tree.ps1')

function Get-DirectoryTree {
    param (
        [string]$Path = "../",
        [string]$Indent = ""
    )

    $items = @(Get-ChildItem -LiteralPath $Path | Where-Object {
        $name = $_.Name
        if ($_.PSIsContainer) {
            return $ExcludeDirs -notcontains $name
        } else {
            foreach ($pattern in $ExcludeFiles) {
                if ($name -like $pattern) { return $false }
            }
            return $true
        }
    } | Sort-Object { -not $_.PSIsContainer }, Name)

    $count = $items.Count
    $i = 0

    foreach ($item in $items) {
        $i++
        $isLast = ($i -eq $count)

        $connector  = if ($isLast) { "└── " } else { "├── " }
        $nextIndent = if ($isLast) { "    " } else { "│   " }

        "$Indent$connector$($item.Name)"

        if ($item.PSIsContainer) {
            Get-DirectoryTree -Path $item.FullName -Indent "$Indent$nextIndent"
        }
    }
}

$rootName = (Get-Item .).Name
$header = @"
==================================================
Project Structure: $rootName
Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
==================================================
"@

$treeLines = Get-DirectoryTree -Path "."
$finalOutput = "$header`n.`n" + ($treeLines -join "`n")

$finalOutput | Out-File -FilePath $OutputFile -Encoding utf8

Write-Host "✅ Successfully saved '$OutputFile' in the root of project." -ForegroundColor Green
