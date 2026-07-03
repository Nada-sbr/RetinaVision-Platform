# PowerShell script to trigger the automated ML training pipeline

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Starting Automated Ocular AI MLOps Pipeline..." -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Set environment variables for local MLflow tracking
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:ENVIRONMENT = "development"

# Run DVC pull to ensure data is updated
Write-Host "Step 1: Fetching latest data from DVC..." -ForegroundColor Yellow
dvc pull

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: DVC pull failed. Running with existing local data." -ForegroundColor Magenta
}

# Run the automated training pipeline
Write-Host "Step 2: Executing PyTorch model training..." -ForegroundColor Yellow
python ml_pipeline/pipeline.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: ML Pipeline failed!" -ForegroundColor Red
    Exit 1
}

Write-Host "=============================================" -ForegroundColor Green
Write-Host "MLOps Pipeline executed successfully!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
