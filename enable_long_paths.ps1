# Enable Windows Long Path Support
# Run this script as Administrator

Write-Host "Enabling Windows Long Path Support..." -ForegroundColor Yellow

# Enable Long Paths in Registry
New-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force -ErrorAction SilentlyContinue | Out-Null
Write-Host "Registry updated: LongPathsEnabled = 1" -ForegroundColor Green

# Enable Long Paths for Git (if installed)
if (Get-Command git -ErrorAction SilentlyContinue)
{
    git config --system core.longpaths true
    Write-Host "Git long paths enabled" -ForegroundColor Green
}

Write-Host ""
Write-Host "IMPORTANT: Restart your computer for changes to take effect!" -ForegroundColor Red
Write-Host ""
Write-Host "After restart, reinstall TensorFlow:" -ForegroundColor Cyan
Write-Host "  cd scania_predictive_maintenance" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  pip uninstall -y tensorflow-cpu tensorflow-intel" -ForegroundColor White
Write-Host "  pip install tensorflow-cpu==2.15.0" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to exit"
