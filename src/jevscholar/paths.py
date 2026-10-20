"""Repository locations: every module takes its paths from here."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
CATALOG = CONFIGS / "systems.yaml"  # the system catalog
EXPERIMENTS = CONFIGS / "experiments"
RUNS = ROOT / "runs"
DATA = ROOT / "data"
SOURCES = DATA / "sources.yaml"  # every external file, pinned
DATA_RAW = DATA / "raw"  # fetched, never committed
DATA_DERIVED = DATA / "derived"  # built by `jev data prepare` and the controller tools
SPLITS = DATA / "splits"  # committed task orders
CACHE = ROOT / "cache"  # replay cache, search store, campaign spend
VENDOR = ROOT / "third_party"
SECRETS = Path("~/.config/jevscholar/secrets.env").expanduser()
