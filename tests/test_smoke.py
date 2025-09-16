from wbkc import run_calibration


def test_run_calibration(capsys):
    run_calibration()
    out, _ = capsys.readouterr()
    assert "WBKC calibration demo running" in out
