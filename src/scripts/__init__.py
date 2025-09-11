"""Installable `scripts` package for test utilities."""
from .gen_data import make_phantoms, make_pregnancy, synth_spectrum

__all__ = ["make_phantoms", "make_pregnancy", "synth_spectrum"]
