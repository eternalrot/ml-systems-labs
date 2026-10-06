# Run from the repo root in PowerShell: .\run_lab01.ps1
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Set-Location lab01
python src\print_versions.py | Out-File -Encoding utf8 results\versions.txt
python src\measure.py
Set-Location ..
