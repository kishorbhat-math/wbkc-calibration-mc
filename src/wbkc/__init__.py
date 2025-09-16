# WBKC Calibration MC

![CI](https://github.com/kishorbhat-math/wbkc-calibration-mc/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Monte Carlo–based calibration models for WBKC (Whole Body Potassium).

## Installation
`ash
pip install -e .
from wbkc import run_calibration
run_calibration()

---

### 5. Commit and push all changes
`powershell
git add LICENSE requirements.txt pyproject.toml README.md
git commit -m "Finalize packaging, license, and documentation"
git push
gh release create v0.1.1 --title "v0.1.1" --notes "Initial release of WBKC Calibration MC with CI, packaging, and docs."
@"
# WBKC Calibration MC

![CI](https://github.com/kishorbhat-math/wbkc-calibration-mc/actions/workflows/ci.yml/badge.svg)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Monte Carlo–based calibration models for WBKC (Whole Body Potassium).

## Installation
`ash
pip install -e .
from wbkc import run_calibration
run_calibration()
mkdir src\wbkc
@"
def run_calibration():
    print("WBKC calibration demo running...")
