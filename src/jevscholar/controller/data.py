"""The training-loop check's data for d1.

`arxiv_categories` only exercises the training loop: real, free, leak-free data with a known answer
that the released checkpoint gets partly wrong (a paper's primary arXiv category among four
neighbouring ones).
"""

from __future__ import annotations

import random
import sqlite3
from contextlib import closing

from jevscholar.controller.d1 import Example

CATEGORIES = {
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.AI": "Artificial Intelligence",
    "cs.IR": "Information Retrieval",
    "cs.RO": "Robotics",
    "cs.CR": "Cryptography and Security",
    "stat.ML": "Machine Learning (Statistics)",
}
QUESTION = "Which arXiv category is this paper's primary category?"


def arxiv_categories(n: int, *, seed: int = 0, options: int = 4) -> list[Example]:
    """`n` papers from the local arXiv index whose primary category is in CATEGORIES, each asked
    to pick it among `options` categories (the others drawn at random, all in random order)."""
    from jevscholar.search import arxiv_local

    rng = random.Random(seed)
    uri = f"file:{arxiv_local.INDEX / 'meta.sqlite'}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    out: list[Example] = []
    seen: set[int] = set()
    while len(out) < n:
        rows = [r for r in rng.sample(range(total), 4 * n) if r not in seen]
        seen.update(rows)
        papers = arxiv_local._papers(arxiv_local.INDEX, "row", rows)
        for row in rows:  # in sampled order, so the result is deterministic per seed
            paper = papers[row]
            primary = paper["categories"][0]
            if primary not in CATEGORIES or len(out) == n:
                continue
            names = [primary, *rng.sample(sorted(set(CATEGORIES) - {primary}), options - 1)]
            rng.shuffle(names)
            question = {
                "type": "choice",
                "instructions": QUESTION,
                "criteria": {c: CATEGORIES[c] for c in names},
            }
            state = {"title": paper["title"], "abstract": paper["abstract"]}
            out.append(Example.labelled(state, question, primary))
    return out
