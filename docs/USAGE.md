# Usage

## Install
```powershell
# from repo root on Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt  # if present
conda env create -f environment.yml
conda activate wbkc-calibration-mc
pip install -e .
