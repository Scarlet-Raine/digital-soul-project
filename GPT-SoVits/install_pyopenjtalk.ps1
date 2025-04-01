# pyopenjtalk Installation for Windows - Virtual Environment version

# Check if running in a virtual environment
$in_venv = $env:VIRTUAL_ENV -ne $null
if (-not $in_venv) {
    Write-Error "This script should be run within an activated virtual environment. Please activate your venv first."
    exit 1
}

Write-Host "Using virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green

# Install jq if needed (using Chocolatey, or you can download it manually)
# If you don't have Chocolatey installed, you can download jq manually from: https://github.com/stedolan/jq/releases
# choco install jq -y

$PACKAGE_NAME = "pyopenjtalk"

# Get the latest version from PyPI
$VERSION = (Invoke-WebRequest -Uri "https://pypi.org/pypi/$PACKAGE_NAME/json" | ConvertFrom-Json).info.version

# Download the package
$url = "https://files.pythonhosted.org/packages/source/$($PACKAGE_NAME.Substring(0,1))/$PACKAGE_NAME/$PACKAGE_NAME-$VERSION.tar.gz"
Invoke-WebRequest -Uri $url -OutFile "$PACKAGE_NAME-$VERSION.tar.gz"

$TAR_FILE = "$PACKAGE_NAME-$VERSION.tar.gz"
$DIR_NAME = $TAR_FILE -replace '\.tar\.gz$', ''

# Extract the tarball (requires 7-Zip or similar)
# You can install 7-Zip with: choco install 7zip -y
& 'C:\Program Files\7-Zip\7z.exe' x $TAR_FILE
& 'C:\Program Files\7-Zip\7z.exe' x "$DIR_NAME.tar"

# Modify the CMakeLists.txt file
$CMAKE_FILE = "$DIR_NAME\lib\open_jtalk\src\CMakeLists.txt"
$content = Get-Content $CMAKE_FILE
$content = $content -replace 'cmake_minimum_required\(VERSION[^\)]*\)', 'cmake_minimum_required(VERSION 3.5...3.31)'
$content | Set-Content $CMAKE_FILE

# Repack the tarball
& 'C:\Program Files\7-Zip\7z.exe' a -ttar "$DIR_NAME.tar" $DIR_NAME
& 'C:\Program Files\7-Zip\7z.exe' a -tgzip "$TAR_FILE" "$DIR_NAME.tar"

# Install the package into the virtual environment
Write-Host "Installing pyopenjtalk into virtual environment" -ForegroundColor Cyan
& "$env:VIRTUAL_ENV\Scripts\pip" install $TAR_FILE

# Clean up
Remove-Item $TAR_FILE -Force
Remove-Item "$DIR_NAME.tar" -Force
Remove-Item -Recurse -Force $DIR_NAME

Write-Host "pyopenjtalk installation complete!" -ForegroundColor Green