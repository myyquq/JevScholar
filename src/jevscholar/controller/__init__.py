"""The d1-3B decision model and what it decides (docs/05_controller.md).

`state`, `prompt` and `policy` serve the controlled loop (systems/controlled.py) in the main
environment; `d1` (readout and training) and `serve` (the HTTP server the d1 policy calls) run in
`.venv-train`; `data` builds d1's training data from controlled runs.
"""
