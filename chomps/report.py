"""Paper tables from the result stores: CSV + LaTeX fragments in ../results/tables/.

    python -m chomps.report                                   # every table, both languages, all splits
    python -m chomps.report --tables factuality_main significance --lang english --split test
    python -m chomps.report --allow-missing                   # while runs are in progress

Inputs. Votes come from `judge.votes_path` / `judge.load_votes`, gold labels
from `data.load_labels`, a meta-learner's members and CV folds from meta.py's
summary CSV. Anything wrong inside an input raises: an example count that
differs from the split, mismatched repeats, invalid votes, a file name in
the votes store that is not judge.py's layout.

Completeness: a table never silently loses a row.
* Only stores that can exist are asked for: a meta-learner on its protocol's
  targets (original: validation and test; cv: test), approach 3 of the
  competition files on validation and test, no fine-tuning-format store for
  the context-free factuality condition. per_judge and meta_cv list the
  stores found on disk instead of a grid of every conceivable run.
* A system is in play once one of its defining stores exists: for a council,
  a member other than its lead (gpt-5-mini, which is also `single`); else its
  one store. Judge runs are made per split, so a judge-run system is expected
  in every language on each split where it has a store in some language (an
  ablation run only on test is fine; a council missing a member, or a
  language missing a run, is not). meta.py writes all of its target splits
  in one run, as did the competition scripts, so those are expected on all
  of their splits once one exists. A meta-learner row also needs meta.py's
  summary rows for its fmt/temp.
* An expected row with a missing input is a gap. By default a table with a
  gap is not written, and every missing input is printed with the rows that
  need it; after the complete tables are written, IncompleteInputs is raised
  (exit status 1). `--allow-missing` writes such a table anyway, with an
  INCOMPLETE header in its .tex and the list in {name}.missing.txt. Systems
  with no store at all are listed as not in the table; they are not gaps.

Protocol decisions:
* Systems (main tables). `single` is gpt-5-mini. `council` is the majority,
  at each repeat index, of gpt-5-mini and the two factuality fine-tunes on
  the original evaluator prompt: the paper's council, on both tracks (the
  original fluency scripts, approach_2/3, used the factuality fine-tunes as
  fluency judges). Factuality adds `council_ftfmt`, the same council with the
  fine-tuned members queried in the format they were fine-tuned on. Fluency
  adds `council_fluft`, gpt-5-mini plus the fluency fine-tunes, and
  `council_fluft_ftfmt`, the same in the fluency fine-tuning format.
  `meta-{protocol}-{cw}[-logits]` is meta.py's predictions at the canonical
  judge/default fmt/temp; its members are the judges named in meta.py's
  summary. Every row's composition is in the CSV `members` column and in a
  comment legend at the top of the .tex. Everything is at temp `default`,
  the only temperature gpt-5-mini accepts and, for GPT-4.1, the API default
  (1.0) that the original scripts set explicitly. Factuality uses
  `--condition` (default pseudo_rag, the competition setting); fluency always
  uses `none`.
* Per system: the mean of every metric over repeats; the macro-F1 std across
  repeats (ddof=1, blank for a single repeat); and three 95 % bootstrap CIs
  of macro-F1 (metrics.summarize_system): of the mean over repeats (the .tex
  CI column, printed next to that mean), of repeat 0 and of the
  majority-over-repeats consensus (CSV).
* Significance: exact McNemar test and paired bootstrap of the macro-F1
  difference (A - B), each on repeat 0 and on the consensus.
* `original_rescored` re-scores the competition-era prediction files
  (../datasets/{lang}/{split}/approach_{1,2,3}_answers_{track}.jsonl) with
  the same metrics.
* Metric values are written with 4 decimals as in the paper (p-values with 4
  significant digits, class percentages with 1 decimal).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import config, data, metrics

TABLES = ("factuality_main", "fluency_main", "retrieval_ablation", "per_judge", "meta_cv",
          "class_distribution", "confusion_matrices", "significance", "original_rescored")
SINGLE_JUDGE = "gpt-5-mini"
BASE_JUDGES = ("gpt-5-mini", "gpt-4.1-mini", "gpt-4.1-nano")
FT_BASES = ("gpt-4.1-mini", "gpt-4.1-nano")
META_JUDGES = tuple(f"meta-{p}-{cw}{lg}" for lg in ("", "-logits") for p in ("original", "cv")
                    for cw in ("none", "balanced"))
# The fluency track has a second council (the dedicated fluency fine-tunes), tagged "fluft".
META_TAGS = {"fluency": ("fluft",)}
META_JUDGES_TAGGED = tuple(f"{v}-{t}" for ts in META_TAGS.values() for t in ts for v in META_JUDGES)
ALL_META_JUDGES = META_JUDGES + META_JUDGES_TAGGED


def _meta_parts(variant: str) -> tuple:
    """(protocol, class_weight, use_logits, tag) of a meta store name."""
    _, protocol, cw, *rest = variant.split("-")
    return protocol, cw, "logits" in rest, "".join(x for x in rest if x != "logits")
META_SPLITS = {"original": ("validation", "test"), "cv": ("test",)}  # meta.run's prediction targets
RETRIEVAL_CONDITIONS = ("pseudo_rag", "pseudo_rag_rerun", "bm25_k1", "bm25_k3", "bm25_k5",
                        "dense_k1", "dense_k3", "dense_k5", "full", "none")
FMTS = ("judge", "ft")
TEMPS = ("default", "0", "1.0")
PREDS = ("repeat0", "consensus")
ORIGINAL_APPROACHES = {1: "single", 2: "council", 3: "meta-original-none"}
ORIGINAL_SPLITS = {1: config.SPLITS, 2: config.SPLITS, 3: ("validation", "test")}  # approach 3 fit on train

# A fine-tuned deployment's language-independent role, so that ft-fact-en-mini and
# ft-fact-es-mini count as the same system across languages.
_FT_ROLE = {dep: (track, base) for (track, _lang, base), dep in config.FT_DEPLOYMENTS.items()}
_ORDER = {r: i for i, r in enumerate([*BASE_JUDGES, *dict.fromkeys(_FT_ROLE.values()), *ALL_META_JUDGES])}


class IncompleteInputs(RuntimeError):
    """Some requested table has an expected row whose inputs are missing (see the module docstring)."""


# --- systems ----------------------------------------------------------------------

def _role(judge: str):
    return _FT_ROLE.get(judge, judge)


def _deployment(role, lang: str) -> str:
    return config.FT_DEPLOYMENTS[(role[0], lang, role[1])] if isinstance(role, tuple) else role


def _role_label(role) -> str:
    return "|".join(_deployment(role, lang) for lang in config.LANGS) if isinstance(role, tuple) else role


@dataclass(frozen=True)
class Member:
    """One votes store (lang/track/split are supplied when it is read)."""
    judge: str
    condition: str
    fmt: str = "judge"
    temp: str = "default"

    def label(self) -> str:
        return f"{self.judge}[{self.condition},{self.fmt},t{self.temp}]"


@dataclass(frozen=True)
class System:
    """A table row: one store, or a council (odd number of stores, majority per repeat).
    `splits`: where its stores can exist. `joint`: one run writes all of them (meta.py),
    so they are expected together; judge runs are per split."""
    name: str
    members: tuple
    splits: tuple = config.SPLITS
    joint: bool = False

    def label(self) -> str:
        return " + ".join(m.label() for m in self.members)

    @property
    def key(self) -> tuple:
        """Language-independent identity: the members, fine-tuned deployments by role."""
        return tuple((_role(m.judge), m.condition, m.fmt, m.temp) for m in self.members)

    @property
    def defining(self) -> tuple:
        """Members whose store puts the system in play: a council's members other than
        its lead (the lead's store is also `single`), else the one store."""
        return self.members[1:] if len(self.members) > 1 else self.members


def _one(name: str, judge: str, condition: str, fmt: str = "judge", temp: str = "default") -> System:
    return System(name, (Member(judge, condition, fmt, temp),))


def _meta(variant: str, condition: str, fmt: str = "judge", temp: str = "default") -> System:
    return System(variant, (Member(variant, condition, fmt, temp),), META_SPLITS[variant.split("-")[1]], True)


def _ft_judges(track: str, lang: str) -> list:
    return [config.FT_DEPLOYMENTS[(track, lang, b)] for b in FT_BASES]


def _ft_possible(track: str, condition: str) -> bool:
    """The fine-tuning format has no context-free variant (judge.py refuses it)."""
    return not (track == "factuality" and condition == "none")


def main_systems(lang: str, track: str, condition: str) -> list:
    """Rows of the main tables. `council` has the same members on both tracks (the paper's
    council); the fluency fine-tunes form the separately named council_fluft*."""
    lead = Member(SINGLE_JUDGE, condition)

    def council(name, ft_track, fmt="judge"):
        return System(name, (lead, *(Member(j, condition, fmt) for j in _ft_judges(ft_track, lang))))

    out = [System("single", (lead,)), council("council", "factuality")]
    if track == "factuality" and _ft_possible(track, condition):
        out.append(council("council_ftfmt", "factuality", "ft"))
    if track == "fluency":
        out += [council("council_fluft", "fluency"), council("council_fluft_ftfmt", "fluency", "ft")]
    tagged = [f"{v}-{t}" for t in META_TAGS.get(track, ()) for v in META_JUDGES]
    return out + [_meta(v, condition) for v in (*META_JUDGES, *tagged)]


def comparisons(lang: str, track: str, condition: str) -> list:
    """(family, A, B) pairs of the significance table:
    main = single vs every other main system, and council (the paper's) vs every other
    council and meta variant; retrieval = single judge on pseudo_rag (the competition
    context) vs each other condition; fmt = each of the track's fine-tuned judges on the
    evaluator prompt vs its fine-tuning format; temp = each GPT-4.1 judge (on fluency also
    the paper council's factuality fine-tunes) at default vs 0 and vs 1.0. No
    multiple-comparison correction is applied."""
    main = {s.name: s for s in main_systems(lang, track, condition)}
    out = [("main", main["single"], s) for name, s in main.items() if name != "single"]
    out += [("main", main["council"], s) for name, s in main.items() if name not in ("single", "council")]
    if track == "factuality":
        ref = _one("single@pseudo_rag", SINGLE_JUDGE, "pseudo_rag")
        out += [("retrieval", ref, _one(f"single@{c}", SINGLE_JUDGE, c))
                for c in RETRIEVAL_CONDITIONS if c != "pseudo_rag"]
    if _ft_possible(track, condition):
        for j in _ft_judges(track, lang):
            out.append(("fmt", _one(f"{j}@judge", j, condition), _one(f"{j}@ft", j, condition, "ft")))
    council_ft = _ft_judges("factuality", lang) if track == "fluency" else []
    for j in dict.fromkeys((*BASE_JUDGES, *_ft_judges(track, lang), *council_ft)):
        if j not in config.REASONING_DEPLOYMENTS:
            out += [("temp", _one(f"{j}@tdefault", j, condition), _one(f"{j}@t{t}", j, condition, "judge", t))
                    for t in ("0", "1.0")]
    return out


# --- inputs -------------------------------------------------------------------------

def _store_api() -> tuple:
    """(votes_path, load_votes) from judge.py. Imported lazily because judge.py
    needs the LLM stack. Tests replace this function."""
    from . import judge
    return judge.votes_path, judge.load_votes


def _meta_summary_path(lang, track, condition, protocol, class_weight, use_logits, tag="") -> Path:
    """meta.py's summary CSV, imported lazily (sklearn stack); tests replace this function."""
    from . import meta
    return Path(meta.summary_path(lang, track, condition, protocol, class_weight, use_logits, tag))


def _read_meta_summary(path: Path, protocol: str, fmt: str, temp: str) -> tuple:
    """({"judges": [...], "cv": (mean, std) | None}, None) from the rows of `path` for this
    fmt/temp, or (None, what is missing). meta.py names the file without fmt/temp, so a
    run at another fmt/temp replaces the rows of this one."""
    if not path.exists():
        return None, str(path)
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r["fmt"], r["temp"]) == (fmt, temp)]
    if not rows:
        return None, f"{path} [no rows for fmt={fmt} temp={temp}]"
    judges = {r["judges"] for r in rows}
    if len(judges) != 1:
        raise ValueError(f"{path}: rows for fmt={fmt} temp={temp} name different judges {sorted(judges)}")
    cv = None
    if protocol == "cv":
        got = {r["term"]: float(r["estimate"]) for r in rows
               if (r["kind"], r["repeat"]) == ("cv_macro_f1", "0") and r["term"] in ("mean", "std")}
        if set(got) != {"mean", "std"}:
            return None, f"{path} [no cv_macro_f1 mean/std for fmt={fmt} temp={temp} repeat 0]"
        cv = (got["mean"], got["std"])
    return {"judges": judges.pop().split(";"), "cv": cv}, None


def _original_predictions(path: Path, track: str) -> dict:
    """{index: vote} of a competition-era approach file."""
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["index"] in out:
                raise ValueError(f"{path}: duplicate index {r['index']}")
            out[r["index"]] = r[config.LABEL_KEY[track]]
    return out


@dataclass(frozen=True)
class Need:
    """A row a table asked for. `keys`: what puts each system it depends on in play (with
    the split for per-split judge runs); `present`: those of the keys whose defining input
    exists in this row; `missing`: its inputs that do not exist."""
    label: str
    group: str
    split: str
    keys: tuple
    present: tuple
    missing: tuple


class Context:
    """One report run: memoised store, label and meta-summary access, and the record of
    every row the current table asked for (`needs`, reset per table by `build`). It is a
    snapshot (what exists is read once): make a new one after the stores change."""

    def __init__(self, langs, splits, condition="pseudo_rag", n_boot=2000, seed=0):
        self.langs, self.splits, self.condition = list(langs), list(splits), condition
        self.n_boot, self.seed = n_boot, seed
        self.needs = []
        self._api = None
        self._votes, self._labels, self._preds, self._summaries, self._meta = {}, {}, {}, {}, {}

    def condition_for(self, track: str) -> str:
        return self.condition if track == "factuality" else "none"

    def gold(self, lang: str, split: str, track: str) -> dict:
        if (lang, split) not in self._labels:
            self._labels[(lang, split)] = data.load_labels(lang, split)
        return {i: lab[track] for i, lab in self._labels[(lang, split)].items()}

    def path(self, lang, track, split, m: Member) -> Path:
        if self._api is None:
            self._api = _store_api()
        return Path(self._api[0](lang, track, split, m.condition, m.judge, m.fmt, m.temp))

    def _store(self, lang, track, split, m: Member):
        """The member's {repeat: {index: vote}}, or None if its store does not exist."""
        path = self.path(lang, track, split, m)
        if path not in self._votes:
            self._votes[path] = self._api[1](path) if path.exists() else None
        return self._votes[path]

    def discover(self, lang, track, split, condition) -> list:
        """Members of every votes store of (lang, track, split) for `condition`, parsed back
        from judge.votes_path's file names ({condition}__{judge}__{fmt}__t{temp}.jsonl)."""
        folder = self.path(lang, track, split, Member(SINGLE_JUDGE, condition)).parent
        out = []
        for p in sorted(folder.glob(f"{condition}__*.jsonl")):
            parts = p.name[:-len(".jsonl")].split("__")
            ok = len(parts) == 4 and parts[2] in FMTS and parts[3][:1] == "t" and parts[3][1:] in TEMPS
            m = Member(parts[1], parts[0], parts[2], parts[3][1:]) if ok else None
            if m is None or self.path(lang, track, split, m) != p:
                raise ValueError(f"{p}: not a votes store name of judge.votes_path")
            out.append(m)
        return out

    def meta_info(self, lang, track, m: Member) -> tuple:
        """meta.py's summary for a meta-learner store: see _read_meta_summary."""
        protocol, cw, use_logits, tag = _meta_parts(m.judge)
        path = _meta_summary_path(lang, track, m.condition, protocol, cw, use_logits, tag)
        if (path, m.fmt, m.temp) not in self._meta:
            self._meta[(path, m.fmt, m.temp)] = _read_meta_summary(path, protocol, m.fmt, m.temp)
        return self._meta[(path, m.fmt, m.temp)]

    def record(self, need: Need) -> bool:
        self.needs.append(need)
        return not need.missing

    def need(self, lang, track, split, *systems, group: str) -> bool:
        """Record a row that needs `systems` at (lang, track, split); True iff all of its
        inputs exist (member stores, and meta.py's summary for a meta-learner)."""
        keys, present, missing = [], [], []
        for s in systems:
            key = (track, s.key) if s.joint else (track, s.key, split)
            keys.append(key)
            for m in s.members:
                if self._store(lang, track, split, m) is None:
                    missing.append(str(self.path(lang, track, split, m)))
                elif m in s.defining:
                    present.append(key)
                why = self.meta_info(lang, track, m)[1] if m.judge in ALL_META_JUDGES else None
                if why:
                    missing.append(why)
        label = f"{lang}/{track}/{split} " + " vs ".join(s.name for s in systems)
        return self.record(Need(label, group, split, tuple(keys), tuple(present), tuple(dict.fromkeys(missing))))

    def gaps(self) -> tuple:
        """(gaps, not_run, some_splits) of the rows recorded for the current table: expected
        rows with a missing input; groups with no store at all; groups expected on only some
        of the requested splits, as {group: those splits}."""
        live = {k for n in self.needs for k in n.present}
        expected = [all(k in live for k in n.keys) for n in self.needs]
        asked, kept = defaultdict(set), defaultdict(set)
        for n, e in zip(self.needs, expected):
            asked[n.group].add(n.split)
            if e:
                kept[n.group].add(n.split)
        order = {s: i for i, s in enumerate(config.SPLITS)}
        return ([n for n, e in zip(self.needs, expected) if e and n.missing],
                sorted(g for g in asked if not kept[g]),
                {g: sorted(kept[g], key=order.get) for g in asked if kept[g] and kept[g] != asked[g]})

    def predictions(self, lang, track, split, system: System):
        """{repeat: {index: vote}} of a system whose stores exist (check with `need` first)."""
        key = (lang, track, split, system.members)
        if key not in self._preds:
            stores = [self._store(lang, track, split, m) for m in system.members]
            if any(s is None for s in stores):
                raise RuntimeError(f"{lang}/{track}/{split} {system.label()}: a member store is missing")
            try:
                self._preds[key] = stores[0] if len(stores) == 1 else metrics.council_by_repeat(stores)
            except ValueError as e:
                raise ValueError(f"{lang}/{track}/{split} {system.label()}: {e}") from e
        return self._preds[key]

    def view(self, lang, track, split, system: System, which: str) -> tuple:
        """Aligned (gold, pred) lists of repeat 0 or of the consensus over repeats."""
        preds = self.predictions(lang, track, split, system)
        try:
            if which == "repeat0" and 0 not in preds:
                raise ValueError(f"no repeat 0 among repeats {sorted(preds)}")
            p = preds[0] if which == "repeat0" else metrics.consensus(preds)
            return metrics.align(self.gold(lang, split, track), p)
        except ValueError as e:
            raise ValueError(f"{lang}/{track}/{split} {system.label()}: {e}") from e

    def summary(self, lang, track, split, system: System) -> dict:
        key = (lang, track, split, system.members)
        if key not in self._summaries:
            preds = self.predictions(lang, track, split, system)
            try:
                self._summaries[key] = metrics.summarize_system(
                    self.gold(lang, split, track), preds, "macro_f1", self.n_boot, self.seed)
            except ValueError as e:
                raise ValueError(f"{lang}/{track}/{split} {system.label()}: {e}") from e
        return self._summaries[key]

    def members(self, lang, track, system: System) -> str:
        """A row's composition: its member stores; a meta-learner's are the judges it was fitted on."""
        parts = []
        for m in system.members:
            if m.judge in ALL_META_JUDGES:
                judges = self.meta_info(lang, track, m)[0]["judges"]
                parts.append(f"{m.judge}(" + " + ".join(Member(j, m.condition, m.fmt, m.temp).label()
                                                         for j in judges) + ")")
            else:
                parts.append(m.label())
        return " + ".join(parts)


# --- tables ---------------------------------------------------------------------------

def _cells(s: dict) -> dict:
    """Columns shared by every per-system table (means over repeats + macro-F1 spread/CIs)."""
    m = s["mean"]
    return {"n": s["n"], "n_repeats": s["n_repeats"],
            **{k: m[k] for k in ("accuracy", "balanced_accuracy", "precision_y", "recall_y", "f1_y")},
            "macro_f1": m["macro_f1"], "macro_f1_std": s["std"]["macro_f1"],
            "ci_mean_lo": s["ci_mean"][0], "ci_mean_hi": s["ci_mean"][1],
            "macro_f1_r0": s["repeat0"]["macro_f1"], "ci_r0_lo": s["ci_repeat0"][0], "ci_r0_hi": s["ci_repeat0"][1],
            "macro_f1_consensus": s["consensus"]["macro_f1"],
            "ci_consensus_lo": s["ci_consensus"][0], "ci_consensus_hi": s["ci_consensus"][1],
            "mcc": m["mcc"]}


def _main(ctx: Context, track: str) -> list:
    rows = []
    for lang in ctx.langs:
        for split in ctx.splits:
            for system in main_systems(lang, track, ctx.condition_for(track)):
                if split in system.splits and ctx.need(lang, track, split, system, group=f"{track} {system.name}"):
                    rows.append({"lang": lang, "split": split, "system": system.name,
                                 **_cells(ctx.summary(lang, track, split, system)),
                                 "members": ctx.members(lang, track, system)})
    return rows


def factuality_main(ctx: Context) -> list:
    return _main(ctx, "factuality")


def fluency_main(ctx: Context) -> list:
    return _main(ctx, "fluency")


def retrieval_ablation(ctx: Context) -> list:
    rows = []
    for lang in ctx.langs:
        for split in ctx.splits:
            for cond in RETRIEVAL_CONDITIONS:
                system = _one("single", SINGLE_JUDGE, cond)
                if ctx.need(lang, "factuality", split, system, group=f"factuality single@{cond}"):
                    rows.append({"lang": lang, "split": split, "condition": cond,
                                 **_cells(ctx.summary(lang, "factuality", split, system))})
    return rows


def _found(ctx: Context, track: str, meta: bool) -> list:
    """(role, fmt, temp) of the stores of the track's condition in any requested
    language/split: individual judges, or (meta=True) meta-learners, in table order."""
    out = set()
    for lang in ctx.langs:
        for split in ctx.splits:
            for m in ctx.discover(lang, track, split, ctx.condition_for(track)):
                if m.judge.startswith("meta-") != meta:
                    continue
                if meta and (m.judge not in ALL_META_JUDGES or split not in META_SPLITS[m.judge.split("-")[1]]):
                    raise ValueError(f"{ctx.path(lang, track, split, m)}: meta.py writes no such store")
                out.add((_role(m.judge), m.fmt, m.temp))
    return sorted(out, key=lambda x: (_ORDER.get(x[0], len(_ORDER)), str(x[0]), FMTS.index(x[1]), TEMPS.index(x[2])))


def per_judge(ctx: Context) -> list:
    """Every individual judge store found for the track's condition (meta-learners are in
    meta_cv). A fine-tuned judge is one row group across languages (by role)."""
    found = {track: _found(ctx, track, meta=False) for track in config.TRACKS}
    rows = []
    for lang in ctx.langs:
        for track in config.TRACKS:
            condition = ctx.condition_for(track)
            for split in ctx.splits:
                for role, fmt, temp in found[track]:
                    j = _deployment(role, lang)
                    system = _one(j, j, condition, fmt, temp)
                    if ctx.need(lang, track, split, system, group=f"{track} {_role_label(role)}@{fmt}/t{temp}"):
                        rows.append({"lang": lang, "track": track, "split": split, "judge": j, "fmt": fmt,
                                     "temp": temp, "condition": condition,
                                     **_cells(ctx.summary(lang, track, split, system))})
    return rows


def meta_cv(ctx: Context) -> list:
    """Every meta-learner store found (any fmt/temp), scored like every other system, with
    its judges and, for the `cv` protocol, the CV fold macro-F1 of meta.py's summary.
    Coefficients stay in meta.py's summary CSVs."""
    found = {track: _found(ctx, track, meta=True) for track in config.TRACKS}
    rows = []
    for lang in ctx.langs:
        for track in config.TRACKS:
            condition = ctx.condition_for(track)
            for split in ctx.splits:
                for variant, fmt, temp in found[track]:
                    system = _meta(variant, condition, fmt, temp)
                    if split not in system.splits or not ctx.need(lang, track, split, system,
                                                                  group=f"{track} {variant}@{fmt}/t{temp}"):
                        continue
                    protocol, cw, logits, tag = _meta_parts(variant)
                    info = ctx.meta_info(lang, track, system.members[0])[0]
                    cv_mean, cv_std = info["cv"] or (None, None)
                    rows.append({"lang": lang, "track": track, "split": split, "variant": variant,
                                 "protocol": protocol, "class_weight": cw, "council": tag or "competition",
                                 "features": "votes+logits" if logits else "votes", "fmt": fmt,
                                 "temp": temp, "judges": " + ".join(info["judges"]),
                                 "cv_macro_f1_mean": cv_mean, "cv_macro_f1_std": cv_std,
                                 **_cells(ctx.summary(lang, track, split, system))})
    return rows


def class_distribution(ctx: Context) -> list:
    rows = []
    for lang in ctx.langs:
        for split in ctx.splits:
            for track in config.TRACKS:
                gold = ctx.gold(lang, split, track)
                n, y = len(gold), sum(gold.values())
                rows.append({"lang": lang, "split": split, "track": track, "n": n, "n_y": y, "n_n": n - y,
                             "pct_y": 100 * y / n, "pct_n": 100 * (n - y) / n})
    return rows


def confusion_matrices(ctx: Context) -> list:
    rows = []
    for lang in ctx.langs:
        for track in config.TRACKS:
            for split in ctx.splits:
                for system in main_systems(lang, track, ctx.condition_for(track)):
                    if split not in system.splits or not ctx.need(lang, track, split, system,
                                                                  group=f"{track} {system.name}"):
                        continue
                    for which in PREDS:
                        c = metrics.confusion(*ctx.view(lang, track, split, system, which))
                        rows.append({"lang": lang, "track": track, "split": split, "system": system.name,
                                     "pred": which, **c, "n": sum(c.values())})
    return rows


def significance(ctx: Context) -> list:
    rows = []
    for lang in ctx.langs:
        for track in config.TRACKS:
            for split in ctx.splits:
                for family, a, b in comparisons(lang, track, ctx.condition_for(track)):
                    if split not in a.splits or split not in b.splits or not ctx.need(
                            lang, track, split, a, b, group=f"{track} {family}: {a.name} vs {b.name}"):
                        continue
                    for which in PREDS:
                        gold, xa = ctx.view(lang, track, split, a, which)
                        _, xb = ctx.view(lang, track, split, b, which)
                        fa, fb = metrics.scores(gold, xa)["macro_f1"], metrics.scores(gold, xb)["macro_f1"]
                        mean_d, lo, hi = metrics.paired_bootstrap_diff(gold, xa, xb, "macro_f1", ctx.n_boot, ctx.seed)
                        n_b, n_c, p = metrics.mcnemar(gold, xa, xb)
                        rows.append({"lang": lang, "track": track, "split": split, "family": family,
                                     "system_a": a.name, "system_b": b.name, "pred": which, "n": len(gold),
                                     "macro_f1_a": fa, "macro_f1_b": fb, "diff": fa - fb,
                                     "diff_boot_mean": mean_d, "ci_lo": lo, "ci_hi": hi,
                                     "mcnemar_b": n_b, "mcnemar_c": n_c, "p_mcnemar": p})
    return rows


def original_rescored(ctx: Context) -> list:
    rows = []
    for lang in ctx.langs:
        for track in config.TRACKS:
            for split in ctx.splits:
                for k, name in ORIGINAL_APPROACHES.items():
                    if split not in ORIGINAL_SPLITS[k]:
                        continue
                    path = config.DATASETS / lang / split / f"approach_{k}_answers_{track}.jsonl"
                    key, exists = ("original", track, k), path.exists()
                    if not ctx.record(Need(f"{lang}/{track}/{split} approach_{k}", f"{track} approach_{k}", split,
                                           (key,), (key,) if exists else (), () if exists else (str(path),))):
                        continue
                    try:
                        s = metrics.summarize_system(ctx.gold(lang, split, track),
                                                     {0: _original_predictions(path, track)},
                                                     "macro_f1", ctx.n_boot, ctx.seed)
                    except ValueError as e:
                        raise ValueError(f"original approach_{k} {lang}/{track}/{split}: {e}") from e
                    rows.append({"lang": lang, "track": track, "split": split, "approach": f"approach_{k}",
                                 "system": name, **_cells(s)})
    return rows


# --- output ---------------------------------------------------------------------------

CSV_FORMATS = {"p_mcnemar": "{:.4g}", "pct_y": "{:.1f}", "pct_n": "{:.1f}"}


def _csv_value(key: str, v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return CSV_FORMATS.get(key, "{:.4f}").format(v)
    return str(v)


def _esc(s: str) -> str:
    for ch in ("_", "%", "&", "#", "$"):
        s = s.replace(ch, "\\" + ch)
    return s


_CAPITALIZED = ("lang", "split", "track")


def _t(key: str):
    """LaTeX cell of one column: 4 decimals for floats, escaped text otherwise."""
    def cell(r):
        v = r[key]
        if v is None:
            return "--"
        return f"{v:.4f}" if isinstance(v, float) else _esc(str(v).capitalize() if key in _CAPITALIZED else str(v))
    return cell


def _pm(r) -> str:
    sd = r["macro_f1_std"]
    return f"{r['macro_f1']:.4f}" + ("" if sd is None else f" $\\pm$ {sd:.4f}")


def _ci(lo: str, hi: str):
    return lambda r: f"[{r[lo]:.4f}, {r[hi]:.4f}]"


def _p(r) -> str:
    return "$<$0.0001" if r["p_mcnemar"] < 1e-4 else f"{r['p_mcnemar']:.4f}"


_SCORES = [("Acc.", "c", _t("accuracy")), ("Bal.\\ acc.", "c", _t("balanced_accuracy")),
           ("P(y)", "c", _t("precision_y")), ("R(y)", "c", _t("recall_y")), ("F1(y)", "c", _t("f1_y")),
           ("Macro-F1", "c", _pm), ("95\\% CI", "c", _ci("ci_mean_lo", "ci_mean_hi"))]
_SCORES_NOTE = ("% Acc. to Macro-F1: means over repeats; Macro-F1 +- its sample std across repeats;"
                " 95% CI: percentile bootstrap over examples of that mean macro-F1.")
_LANG_SPLIT = [("Language", "l", _t("lang")), ("Split", "l", _t("split"))]
_LTS = [("Language", "l", _t("lang")), ("Track", "l", _t("track")), ("Split", "l", _t("split"))]

LATEX = {
    "factuality_main": _LANG_SPLIT + [("System", "l", _t("system"))] + _SCORES,
    "fluency_main": _LANG_SPLIT + [("System", "l", _t("system"))] + _SCORES,
    "retrieval_ablation": _LANG_SPLIT + [("Condition", "l", _t("condition"))] + _SCORES,
    "per_judge": _LTS + [("Judge", "l", _t("judge")), ("Format", "l", _t("fmt")),
                         ("Temp.", "l", _t("temp"))] + _SCORES,
    "meta_cv": _LTS + [("Variant", "l", _t("variant")), ("Format", "l", _t("fmt")), ("Temp.", "l", _t("temp")),
                       ("CV macro-F1", "c", lambda r: "--" if r["cv_macro_f1_mean"] is None else
                        f"{r['cv_macro_f1_mean']:.4f} $\\pm$ {r['cv_macro_f1_std']:.4f}")] + _SCORES,
    "class_distribution": _LANG_SPLIT + [("Track", "l", _t("track")), ("Examples", "c", _t("n")),
                                         ("\\# y", "c", _t("n_y")), ("\\# n", "c", _t("n_n")),
                                         ("\\% y", "c", lambda r: f"{r['pct_y']:.1f}")],
    "confusion_matrices": _LTS + [("System", "l", _t("system")), ("Pred.", "l", _t("pred")), ("TP", "c", _t("tp")),
                                  ("FP", "c", _t("fp")), ("TN", "c", _t("tn")), ("FN", "c", _t("fn"))],
    "significance": _LTS + [("Family", "l", _t("family")), ("A", "l", _t("system_a")), ("B", "l", _t("system_b")),
                            ("Pred.", "l", _t("pred")), ("Macro-F1 A", "c", _t("macro_f1_a")),
                            ("Macro-F1 B", "c", _t("macro_f1_b")), ("$\\Delta$", "c", _t("diff")),
                            ("95\\% CI of $\\Delta$", "c", _ci("ci_lo", "ci_hi")), ("$b$", "c", _t("mcnemar_b")),
                            ("$c$", "c", _t("mcnemar_c")), ("$p$", "c", _p)],
    "original_rescored": _LTS + [("Approach", "l", _t("system"))] + _SCORES,
}


def _gap_lines(gaps: list) -> list:
    """One line per missing input, with the rows that need it."""
    by_input = defaultdict(list)
    for n in gaps:
        for x in n.missing:
            by_input[x].append(n.label)
    return [f"{x}  <- {', '.join(rows)}" for x, rows in sorted(by_input.items())]


def _latex(rows: list, cols: list, notes: list) -> str:
    lines = ["% generated by chomps.report; rerun it instead of editing", *notes,
             "\\begin{tabular}{" + "".join(a for _, a, _ in cols) + "}", "\\hline",
             " & ".join(f"\\textbf{{{h}}}" for h, _, _ in cols) + " \\\\", "\\hline"]
    lines += [" & ".join(fn(r) for _, _, fn in cols) + " \\\\" for r in rows]
    return "\n".join(lines + ["\\hline", "\\end{tabular}", ""])


def write_table(name: str, rows: list, gaps: list = ()) -> Path:
    """CSV + .tex of a table. A table with gaps (--allow-missing) gets an INCOMPLETE header
    in its .tex and {name}.missing.txt; writing it complete removes that file."""
    out = config.RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(rows[0]))
        w.writerows([[_csv_value(k, v) for k, v in r.items()] for r in rows])
    notes, missing = [], out / f"{name}.missing.txt"
    if gaps:
        lines = _gap_lines(gaps)
        notes += [f"% INCOMPLETE (built with --allow-missing): {len(gaps)} expected row(s) absent; "
                  f"missing inputs <- rows:", *(f"%   {x}" for x in lines)]
        missing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        missing.unlink(missing_ok=True)
    if "members" in rows[0]:
        legend = {(r["lang"], r["system"]): r["members"] for r in rows}
        notes += ["% Systems (votes stores as judge[condition,fmt,temp]; a council is the per-repeat majority):",
                  *(f"%   {lang} {system}: {m}" for (lang, system), m in legend.items())]
    if "ci_mean_lo" in rows[0]:
        notes.append(_SCORES_NOTE)
    (out / f"{name}.tex").write_text(_latex(rows, LATEX[name], notes), encoding="utf-8")
    return out / f"{name}.csv"


BUILDERS = {"factuality_main": factuality_main, "fluency_main": fluency_main,
            "retrieval_ablation": retrieval_ablation, "per_judge": per_judge, "meta_cv": meta_cv,
            "class_distribution": class_distribution, "confusion_matrices": confusion_matrices,
            "significance": significance, "original_rescored": original_rescored}


def build(names, ctx: Context, allow_missing: bool = False) -> dict:
    """Build and write the named tables; returns {name: rows} of those written. A table
    with a gap is written only with `allow_missing`; otherwise IncompleteInputs is raised
    once every other table has been written."""
    written, refused = {}, []
    for name in names:
        ctx.needs = []
        rows = BUILDERS[name](ctx)
        gaps, not_run, some_splits = ctx.gaps()
        if not rows and not gaps:
            print(f"[report] {name}: SKIPPED (no inputs yet"
                  + (f"; none of its {len(not_run)} row groups has a store)" if not_run else ")"))
            continue
        if not_run:
            print(f"[report] {name}: not in the table, no input at all ({len(not_run)}): {', '.join(not_run)}")
        if some_splits:
            print(f"[report] {name}: only on the splits that have inputs: "
                  + "; ".join(f"{g} ({', '.join(s)})" for g, s in some_splits.items()))
        if gaps:
            lines = _gap_lines(gaps)
            print(f"[report] {name}: {len(gaps)} expected row(s) lack inputs; {len(lines)} missing input(s) <- rows:")
            print("\n".join(f"    {x}" for x in lines))
            if not allow_missing:
                print(f"[report] {name}: NOT WRITTEN (an existing {name}.csv/.tex is left as it was)")
                refused.append(name)
                continue
            if not rows:
                print(f"[report] {name}: SKIPPED (no complete row)")
                continue
        path = write_table(name, rows, gaps)
        print(f"[report] {name}: {len(rows)} rows -> {path} (+ .tex)" + (" INCOMPLETE" if gaps else ""))
        written[name] = rows
    if refused:
        raise IncompleteInputs(f"not written, expected inputs missing: {', '.join(refused)}. Produce the inputs "
                               f"listed above, restrict --lang/--split, or pass --allow-missing.")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the paper tables from ../results stores.")
    ap.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    ap.add_argument("--lang", nargs="+", choices=config.LANGS, default=list(config.LANGS))
    ap.add_argument("--split", nargs="+", choices=config.SPLITS, default=list(config.SPLITS))
    ap.add_argument("--condition", choices=RETRIEVAL_CONDITIONS, default="pseudo_rag",
                    help="factuality condition of the main, per-judge, meta and significance tables "
                         "(fluency always uses none)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing", action="store_true",
                    help="write a table even if expected rows lack inputs (marked INCOMPLETE in its .tex and "
                         "listed in <table>.missing.txt); by default it is not written and the exit status is 1")
    a = ap.parse_args(argv)
    try:
        build(a.tables, Context(a.lang, a.split, a.condition, a.n_boot, a.seed), a.allow_missing)
    except IncompleteInputs as e:
        sys.exit(f"[report] {e}")


if __name__ == "__main__":
    main()
