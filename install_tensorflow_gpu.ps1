# Install TensorFlow with GPU support
cd "C:\Users\Iduma\OneDrive - University of Hertfordshire\Desktop\scania_predictive_maintenance"

Write-Host "Uninstalling CPU-only TensorFlow..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip uninstall -y tensorflow-cpu tensorflow-intel

Write-Host "`nInstalling TensorFlow with GPU support..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip install tensorflow[and-cuda]==2.15.0

Write-Host "`nVerifying GPU detection..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs detected: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]"

Write-Host "`nSetup complete!" -ForegroundColor Green
