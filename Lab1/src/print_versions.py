"""Print Python and library versions (Lab 1, task 1)."""
import sys
import platform
from importlib import metadata

PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory-profiler", "psutil", "codecarbon", "fastapi", "uvicorn",
    "pytest", "httpx", "locust", "requests", "pyarrow", "joblib", "tqdm",
]

print(f"Python {sys.version}")
print(f"Platform {platform.platform()}")
for name in PACKAGES:
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}: NOT INSTALLED")
