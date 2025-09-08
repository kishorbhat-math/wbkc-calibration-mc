# Usage:
#   python scripts\\fit_phantoms.py data\\phantoms.csv
from wbkc.calib_fit import fit_from_file
import sys
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/fit_phantoms.py <phantoms.csv>")
        sys.exit(2)
    phantom_csv = sys.argv[1]
    params = fit_from_file(phantom_csv, out_json="docs/calibration/calib_params.json")
    print("Fitted params:", params)
