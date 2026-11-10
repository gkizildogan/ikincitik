"""Experiments: which questions are run and where the results go.

An experiment is a name plus a list of question files, declared in
``experiments.toml`` at the repo root. The name is the results folder; everything
the experiment produces is under ``results/<name>/``:

  runs/<run_id>/     run records (``runs.py``; notebooks 07 and 08)
  qrels.jsonl        span-based qrels of its questions (notebook 06), unless the
                     experiment names another path with ``qrels``
  run_metrics.parquet, question_metrics.parquet, <index>/metrics.{csv,xlsx},
  granularity.csv, diagrams/     (``report.py``; notebook 09)

The name is not a parameter version. One experiment holds the v1 runs and, later,
the v2 runs of the same questions, so both are reported side by side. A run id does
not depend on the questions, so two question sets must never share a folder:
``runs.execute`` refuses to reuse a run that was made with other questions.

The active experiment is ``active`` in ``experiments.toml``; the environment
variable ``CSRAG_EXPERIMENT`` overrides it. It is resolved once per process, so
editing the file does not move a running notebook to another folder. The caches in
``data/cache/`` and the query embeddings are keyed by text and shared by all
experiments.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CONFIG = Path("experiments.toml")  # relative to the repo root, like every path in src/
RESULTS_ROOT = Path("results")
ENV = "CSRAG_EXPERIMENT"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")


def read_questions(paths) -> list[dict]:
    """The records of one question file, or of several concatenated in the order given."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out += [json.loads(ln) for ln in f if ln.strip()]
    return out


@dataclass(frozen=True)
class Experiment:
    name: str
    questions: tuple[Path, ...]
    description: str = ""
    qrels: Path | None = None
    root: Path = RESULTS_ROOT

    def __post_init__(self):
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"experiment name {self.name!r}: use letters, digits and _ . + - only (it is a folder name)")
        if not self.questions:
            raise ValueError(f"experiment {self.name!r} has no question files")

    @property
    def dir(self) -> Path:
        """``results/<name>``: every file of this experiment."""
        return Path(self.root) / self.name

    @property
    def runs_dir(self) -> Path:
        return self.dir / "runs"

    @property
    def qrels_path(self) -> Path:
        return Path(self.qrels) if self.qrels else self.dir / "qrels.jsonl"

    def read_questions(self) -> list[dict]:
        """All question records of the experiment (raw, as in the files)."""
        qs = read_questions(self.questions)
        dup = sorted(i for i, n in Counter(q["question_id"] for q in qs).items() if n > 1)
        if dup:
            raise ValueError(f"experiment {self.name!r}: question ids occur more than once: {dup[:10]}")
        return qs

    def summary(self) -> str:
        lines = [f"experiment {self.name!r} -> {self.dir}/" + (f"  ({self.description})" if self.description else "")]
        total = answerable = 0
        for p in self.questions:
            qs = read_questions(p)
            total += len(qs)
            answerable += sum(bool(q["answerable"]) for q in qs)
            lines.append(f"  {len(qs):4}  {p}")
        lines.append(f"  {total:4}  questions, {answerable} answerable | qrels: {self.qrels_path}")
        return "\n".join(lines)


def load(name: str | None = None, config: str | Path | None = None) -> Experiment:
    """The experiment ``name`` from ``experiments.toml``.

    Without a name: ``CSRAG_EXPERIMENT`` if set, else the file's ``active`` key.
    """
    path = Path(config) if config else CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found (the working directory must be the repo root: {Path.cwd()})")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    table = cfg.get("experiments", {})
    name = name or os.environ.get(ENV) or cfg.get("active")
    if not name:
        raise KeyError(f"no experiment selected: set `active` in {path} or the {ENV} environment variable")
    if name not in table:
        raise KeyError(f"experiment {name!r} is not defined in {path}; defined: {sorted(table)}")
    e = table[name]
    unknown = set(e) - {"description", "questions", "qrels"}
    if unknown:
        raise KeyError(f"experiment {name!r} in {path}: unknown keys {sorted(unknown)}")
    questions = tuple(Path(p) for p in e.get("questions", []))
    missing = [str(p) for p in questions if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"experiment {name!r}: question files not found: {missing}")
    return Experiment(name=name, questions=questions, description=e.get("description", ""),
                      qrels=Path(e["qrels"]) if e.get("qrels") else None)


_active: Experiment | None = None


def active() -> Experiment:
    """The experiment of this process: resolved on first use, then fixed."""
    global _active
    if _active is None:
        _active = load()
    return _active


def use(experiment: str | Experiment) -> Experiment:
    """Switch this process to another experiment (ad-hoc analysis, tests)."""
    global _active
    _active = experiment if isinstance(experiment, Experiment) else load(experiment)
    return _active
