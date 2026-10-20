"""External data: pinned downloads (`fetch`), derived artefacts and committed splits (`prepare`).

Every external byte comes from data/sources.yaml at a pinned revision and is sha256-verified, so
each score names exactly what it was computed on. `prepare` is deterministic (re-running it
rewrites identical files) and checks what upstream does not: DeepScholar's June-2025 nugget
folders are misaligned with the CSV rows (upstream issue 16), so nugget files are matched to rows
by qid and by the abstract they were generated from, never by folder index.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import takewhile
from pathlib import Path
from urllib.parse import quote

import yaml

from jevscholar.paths import DATA_DERIVED as DERIVED
from jevscholar.paths import DATA_RAW as RAW
from jevscholar.paths import SOURCES, SPLITS
from jevscholar.util import download, read_json, read_jsonl, squash

DRB2_TASKS = RAW / "drb2" / "tasks_and_rubrics.jsonl"
# DRB2 dev: 12 fixed ids (6 zh / 6 en) that had already been read while building the harness.
DRB2_DEV = ("118", "60", "83", "52", "47", "114", "121", "63", "19", "103", "48", "88")
ABSTRACT_MARK = "Here is the paper abstract:"


# --------------------------------------------------------------------------- fetch
def fetch(pin: bool = False) -> None:
    """Download every source into data/raw/ and verify it; `pin` writes trusted-on-first-use
    hashes back into sources.yaml (its leading comment block is kept)."""
    text = SOURCES.read_text(encoding="utf-8")
    sources = yaml.safe_load(text)
    for name, src in sources.items():
        if "extract" in src:
            archive = RAW / "archives" / f"{name}.tar.gz"
            src["sha256"] = _verified(name, src["url"], archive, src["sha256"], pin)
            _extract(archive, src["extract"])
        else:
            for rel, sha in src["files"].items():
                url, dest = src["url"] + quote(rel), RAW / src["dest"] / rel
                src["files"][rel] = _verified(f"{name}: {rel}", url, dest, sha, pin)
        print(f"{name}: ok")
    if pin:
        head = "".join(takewhile(lambda ln: ln.startswith("#"), text.splitlines(keepends=True)))
        dump = yaml.safe_dump(sources, sort_keys=False, allow_unicode=True, width=100)
        SOURCES.write_text(f"{head}\n{dump}", encoding="utf-8")


def _verified(name: str, url: str, dest: Path, sha256: str | None, pin: bool) -> str:
    """`download`, refusing an unpinned file unless `pin` (then its digest is written back)."""
    digest = download(url, dest, sha256)
    if not sha256 and not pin:
        raise ValueError(
            f"{name} has no pinned sha256 (downloaded {digest}); check it and "
            "re-run `jev data fetch --pin` to pin it"
        )
    return digest


def _extract(archive: Path, members: Mapping[str, str]) -> None:
    """Extract only the listed members of a GitHub tarball: regular files, never outside RAW.

    A member ending in "/" is a directory; its destination is replaced, not merged."""
    for dst in members.values():
        if dst.endswith("/"):
            shutil.rmtree(RAW / dst, ignore_errors=True)
    found = set()
    with tarfile.open(archive) as tar:
        for m in tar:
            rel = m.name.partition("/")[2]  # drop the "<repo>-<sha>/" top-level folder
            for src, dst in members.items():
                is_dir = src.endswith("/")
                if not m.isfile() or not (rel.startswith(src) if is_dir else rel == src):
                    continue
                out = RAW / dst / rel.removeprefix(src) if is_dir else RAW / dst
                if not out.resolve().is_relative_to((RAW / dst).resolve()):
                    raise ValueError(f"unsafe archive member {m.name!r} in {archive.name}")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tar.extractfile(m).read())  # type: ignore[union-attr]
                found.add(src)
    if missing := set(members) - found:
        raise ValueError(f"{archive.name}: members not found: {sorted(missing)}")


# --------------------------------------------------------------------------- readers
def deepscholar_rows(dataset: str) -> list[dict[str, str]]:
    """Rows of papers_with_related_works.csv ("jun2025" | "nov2025"); the index is the row id
    the official evaluator uses (`dataset.iloc[int(folder)]`)."""
    import pandas as pd

    path = RAW / "deepscholar" / dataset / "papers_with_related_works.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")


# --------------------------------------------------------------------------- prepare
def prepare() -> None:
    """Build data/derived/ and data/splits/ from data/raw/ (run `fetch` first)."""
    for dataset in ("jun2025", "nov2025"):
        mapping = materialise_nuggets(dataset)
        moved = sum(src != str(row) for row, src in mapping.items())
        print(
            f"deepscholar {dataset} nuggets: {len(mapping)}/{len(deepscholar_rows(dataset))} "
            f"rows matched by qid ({moved} not at their row folder)"
        )
    SPLITS.mkdir(parents=True, exist_ok=True)
    for bench, splits in build_splits().items():
        (SPLITS / f"{bench}.json").write_text(json.dumps(splits, indent=1) + "\n")
        print(f"splits {bench}: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    print(f"drb2 reference reports: {prepare_drb2_reports()} files")


def materialise_nuggets(dataset: str) -> dict[int, str]:
    """Copy each CSV row's nugget file, found by qid, to derived/deepscholar/<dataset>/nuggets/
    <row>/res.json (bytes unchanged). Returns {row: source folder under gt_nuggets_outputs}.

    Errors: two byte-different files for one qid, a row without nuggets, or a nugget query that
    was generated from a different abstract than the row's."""
    src = RAW / "deepscholar" / dataset / "gt_nuggets_outputs"
    by_qid: dict[str, Path] = {}
    for path in sorted(src.rglob("res.json"), key=lambda p: (len(p.parts), p)):
        first = by_qid.setdefault(read_json(path)["qid"], path)
        if first.read_bytes() != path.read_bytes():
            raise ValueError(f"byte-different nugget files for one qid: {first}, {path}")
    out = DERIVED / "deepscholar" / dataset / "nuggets"
    shutil.rmtree(out, ignore_errors=True)
    mapping = {}
    for row, paper in enumerate(deepscholar_rows(dataset)):
        path = by_qid[paper["arxiv_id"]]
        query = read_json(path)["query"]
        if squash(query.partition(ABSTRACT_MARK)[2]) != squash(paper["abstract"]):
            raise ValueError(f"{dataset} row {row}: {path} was not built from the row's abstract")
        (out / str(row)).mkdir(parents=True)
        shutil.copyfile(path, out / str(row) / "res.json")
        mapping[row] = path.parent.relative_to(src).as_posix()
    return mapping


def prepare_drb2_reports() -> int:
    """Reference reports in the official evaluator layout (<system>/idx-<n>.md|.docx): md and
    docx as published; html and pdf, which the evaluator cannot read, as extracted text."""
    src, out = RAW / "drb2" / "reports", DERIVED / "drb2" / "reports"
    shutil.rmtree(out, ignore_errors=True)
    paths = sorted(src.glob("*/idx-*.*"))
    for path in paths:
        dest = out / path.parent.name / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".md", ".docx"):
            shutil.copyfile(path, dest)
        else:
            dest.with_suffix(".md").write_text(_EXTRACT[path.suffix](path), encoding="utf-8")
    return len(paths)


def _html_text(path: Path) -> str:
    """Visible text, each link followed by its URL (the judge checks cited URLs)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for a in soup.find_all("a", href=True):
        a.append(f" ({a['href']})")
    return soup.get_text("\n", strip=True)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() for page in PdfReader(path).pages)


_EXTRACT = {".html": _html_text, ".pdf": _pdf_text}


# --------------------------------------------------------------------------- splits
def order(
    bench: str,
    ids: Iterable[str],
    strata: Mapping[str, str] | None = None,
    *,
    proportional: bool = False,
) -> list[str]:
    """Deterministic order: sha256("jevscholar|<bench>|<id>") within each stratum, strata
    interleaved round-robin (in the order of their first member) so that every prefix is
    balanced. `proportional` interleaves in proportion to stratum size instead of equally."""
    ranked = sorted(ids, key=lambda i: hashlib.sha256(f"jevscholar|{bench}|{i}".encode()).digest())
    if strata is None:
        return ranked
    groups: dict[str, list[str]] = defaultdict(list)
    rank = {}
    for i in ranked:
        rank[i] = len(groups[strata[i]])
        groups[strata[i]].append(i)
    first = {g: n for n, g in enumerate(groups)}

    def key(i: str) -> tuple[float, int]:
        g = strata[i]
        return (rank[i] + 0.5) / (len(groups[g]) if proportional else 1), first[g]

    return sorted(ranked, key=key)


def build_splits() -> dict[str, dict[str, list[str]]]:
    """All split lists from data/raw/ (docs/04_data_and_evaluation.md: Splits)."""
    jun, nov = deepscholar_rows("jun2025"), deepscholar_rows("nov2025")
    primary = {p["arxiv_id"]: p["categories"].split(",")[0].strip() for p in [*jun, *nov]}
    jun_order = order("deepscholar", [p["arxiv_id"] for p in jun], primary, proportional=True)
    nov_order = order("deepscholar", [p["arxiv_id"] for p in nov], primary, proportional=True)

    rqa = RAW / "researchqa"
    valid = read_json(rqa / "valid.json")
    test = {q["id"]: q for q in read_json(rqa / "test.json")}
    mini = read_json(rqa / "researchqa_official_subset_ids.json")
    domain = {q["id"]: q["general_domain"] for q in [*valid, *(test[i] for i in mini)]}

    language = {str(t["idx"]): t["language"] for t in read_jsonl(DRB2_TASKS)}
    return {
        "deepscholar": {
            "dev": jun_order[:20],
            # the controller's tasks, disjoint from dev and test: validation (selection), training
            "val": jun_order[20:29],
            "train": jun_order[29:],
            "dev2": nov_order[:20],
            "test": nov_order[20:],
        },
        "researchqa": {
            "dev": order("researchqa", [q["id"] for q in valid], domain),
            "test": order("researchqa", mini, domain),
        },
        "drb2": {
            "dev": order("drb2", DRB2_DEV, language),
            "test": order("drb2", [i for i in language if i not in DRB2_DEV], language),
        },
    }
