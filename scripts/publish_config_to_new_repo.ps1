<#
.publish_config_to_new_repo.ps1

Creates a temporary folder containing selected config files from the current repo,
initializes a new git repo there, and pushes it to the provided remote.

USAGE:
  .\scripts\publish_config_to_new_repo.ps1 -Remote 'https://github.com/your/repo.git'

This script does NOT copy any `.env` files.
#>

param(
    [string]$Remote = 'https://github.com/znoux46/SEP490.8-WikiChatbot-RAGModel.git',
    [string[]]$FilesToInclude = @(
        'docker-compose.yml',
        'Dockerfile',
        'init_db.sql',
        'migrations',
        'DOCKER_DEPLOYMENT.md',
        'README.md',
        '.github'
    )
)

$TempDir = Join-Path -Path $PWD -ChildPath 'temp_config'
if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
New-Item -ItemType Directory -Path $TempDir | Out-Null

Write-Host "Copying selected config files to $TempDir"
foreach ($f in $FilesToInclude) {
    if (Test-Path $f) {
        Write-Host " - $f"
        Copy-Item -Path $f -Destination $TempDir -Recurse -Force
    } else {
        Write-Host " - (missing) $f"
    }
}

# Add a safe .gitignore to ensure env files won't be committed
Set-Location $TempDir
if (-Not (Test-Path '.gitignore')) {
    @('.env', '.env.*') | Out-File -FilePath .gitignore -Encoding utf8
}

Write-Host "Initializing git repository in $TempDir"
git init | Out-Null
git add .
git commit -m "Publish selected config files (no env)" | Out-Null

Write-Host "Adding remote $Remote and pushing"
git remote add origin $Remote
git branch -M main
git push -u origin main --force

Write-Host "Done. Temporary folder: $TempDir"