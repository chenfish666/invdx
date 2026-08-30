#!/usr/bin/env python3
"""Mechanical drift check for the English/Traditional-Chinese doc pairs.

The repo keeps every document under docs/, under tutorials/, and at the repo
root in two languages. Every one of those is the same fact written down twice,
and this project's own docs name "two copies, one updated" as the failure shape
that produces no error signal. This script is that error signal. It only checks
what a machine can check without pretending to read meaning:

  1. pairing        every X.zh-TW.md has an X.md next to it
  2. header links   each half's first line points at the other half
  3. headings       the two sides have the same sequence of heading levels
  4. code blocks    the two sides carry byte-identical commands and code
  5. link sets      neither side has a relative link the other lacks
  6. dead links     every relative link resolves to a file that exists
  7. anchors        every `file.md` and every <section> anchor resolves
  8. banned forms   no ruled-out Chinese rendering survives anywhere, in
                    either language tree
  9. join keys      a ruled Chinese form appears only where its English
                    counterpart term is recoverable
 10. ruling table   the two halves of the glossary hold the same rulings,
                    row for row and column for column

Checks 7-10 exist because docs/glossary.zh-TW.md used to carry the facts they
recompute. That page is the terminology ruling table; it causes the other six
Chinese documents to be revised, and a revision is exactly what invalidates a
sentence like "optimize.zh-TW.md currently writes X". Three revision rounds
each fixed the rot found at the time and each grew new rot, so the rot rate is
the page's own revision rate. The fix is a split: the page keeps the rulings
(decisions, which do not rot) and this script recomputes the observations
(claims about other files' current state) on every run.

Checks 8-10 are only as strong as the counts they are fed, so every one of
those counts is itself guarded: COUNT_FLOORS holds a floor per count, checked
in one loop after the counters are populated, and a ruling table is identified
by its header rather than by being four columns wide. The floors exist because
the counts were shrinkable in silence -- a count printed next to an OK is a
description of the input, not a check on it. What they do not cover, on
purpose, is someone lowering a floor or padding a table deliberately; see
COUNT_FLOORS.

What it deliberately does NOT check: prose. Translation quality, added or
dropped sentences, whether a symbol is actually defined at its first use --
none of that is mechanical, and a check that claims to cover it would be worse
than no check, because it would be believed. Checks 8 and 9 are string
matching over a hand-written ruling table, not comprehension.

Two normalisations are applied before code blocks are compared, both of them
required by conventions the docs themselves state:

  - Comments are stripped. docs/env.zh-TW.md states the rule the Chinese side
    follows: "commands, flags, paths, config keys, versions and error messages
    are left exactly as they are; only comments are translated." Comparing
    comments verbatim would therefore fire on every block that has one. The
    count of excluded comment lines is printed, so the loss of coverage is
    visible rather than silent. A "#" only starts a comment when it follows
    whitespace and is outside quotes -- a blind split() would silently behead
    `mpich/1.0#hash` and `env.md#anchor`, i.e. drop exactly the kind of string
    this check exists to compare.
  - Angle-bracket placeholders are wildcarded, but only when the text inside
    them contains a space or a non-ASCII character: those are prose stand-ins
    ("<your GPU model>" / "<你的顯示卡型號>"). `<run-dir>`, `<dir>`, `<br/>`
    and friends stay literal and are compared, because they are identical on
    both sides and a difference in one would be real.

mermaid blocks are compared on structure, not text: node labels are prose and
are translated, so label contents are blanked and the remaining skeleton (node
ids, edges, arrow directions, subgraphs) is compared. A diagram that gains an
edge on one side only is drift, and skipping mermaid outright would miss it.

Usage:  python3 scripts/check_bilingual.py     (or: make bilingual)
Exit:   0 clean, 1 problems found, 2 the check itself could not run.
"""

from __future__ import annotations

import bisect
import difflib
import re
import sys
from pathlib import Path

# Directories scanned recursively for bilingual pairs, relative to the repo root.
SCAN_DIRS = ("docs", "tutorials")

# The repo root is scanned too, but **only its top level**. README.md is the
# most-read page in the tree and for a long time it was the one page no check
# here ever looked at: it was appended to `pages` for the dead-link count
# alone, its Chinese half was never collected, so no pair was ever formed and
# the banned-form, join-key, code-block, link-set, heading and anchor checks
# all skipped it. The summary still printed "0 English-only pages", which
# reads as "everything is paired" -- the hole was hidden by a denominator that
# never covered the place the hole was in.
#
# Non-recursive on purpose: rglob from the root walks .venv/, spack/env/,
# runs/ and every third-party README underneath them, none of which this repo
# writes or translates.
SCAN_ROOT = True

ZH_SUFFIX = ".zh-TW.md"

# Languages whose comment marker is "#". Everything a fence in this repo is
# tagged with is in here; an unknown tag falls back to no comment stripping,
# which is the conservative direction (it compares more, not less).
HASH_COMMENT = {"bash", "sh", "shell", "console", "yaml", "yml", "toml",
                "python", "py", "make", "makefile", "ini", "cfg", ""}

# Fence languages compared on structure rather than text: the words inside a
# diagram's labels are prose and get translated, the graph itself does not.
DIAGRAM_FENCES = {"mermaid"}

LINK_RE = re.compile(r"!?\[[^\]]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
# A placeholder, not just any angle brackets: prose stand-ins contain a space
# or non-ASCII text. Keeping <run-dir> and <br/> literal preserves coverage.
PLACEHOLDER_RE = re.compile(r"<(?=[^<>]{0,80}>)[^<>]*?(?:\s|[^\x00-\x7f])[^<>]*?>")
# Used to blank mermaid label text. Quotes and edge-label pipes are handled
# separately from brackets on purpose: a single character class holding both
# openers and closers matches `{"` as if it were a pair.
MERMAID_QUOTE_RE = re.compile(r'"[^"]*"')
MERMAID_PIPE_RE = re.compile(r"\|[^|]*\|")
MERMAID_BRACKET_RE = re.compile(r"[\[({][^\[\](){}]*[\])}]")
EXTERNAL_RE = re.compile(r"^(https?:|mailto:|ftp:|#)")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def split_blocks(lines: list[str]) -> tuple[list[tuple[str, int, list[str]]], list[tuple[int, str]]]:
    """Return (fenced blocks, numbered prose lines).

    A block is (fence-language, 1-based line number of the opening fence,
    content lines). An unterminated fence is returned as a block so that a
    truncated file is reported as a content difference rather than silently
    dropping half the document.

    Prose lines keep their 1-based line numbers: the anchor, banned-form and
    join-key checks all have to say *where*, and a finding a reader cannot
    locate is a finding they will not act on.
    """
    blocks: list[tuple[str, int, list[str]]] = []
    prose: list[tuple[int, str]] = []
    lang: str | None = None
    start = 0
    body: list[str] = []
    for n, line in enumerate(lines, 1):
        if line.startswith("```"):
            if lang is None:
                lang, start, body = line[3:].strip().lower(), n, []
            else:
                blocks.append((lang, start, body))
                lang = None
            continue
        if lang is None:
            prose.append((n, line))
        else:
            body.append(line)
    if lang is not None:
        blocks.append((lang, start, body))
    return blocks, prose


def strip_trailing_comment(line: str) -> tuple[str, bool]:
    """Cut a "#" comment off the end of a line, respecting quotes.

    A "#" only opens a comment at the start of the line or after whitespace,
    and never inside a quoted string. Without this, `mpich/1.0#hash`,
    `env.md#anchor` and `sed 's/#.*//'` all lose their tail -- and the tail is
    precisely what a drift check is supposed to compare.
    """
    quote: str | None = None
    for i, ch in enumerate(line):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip(), True
    return line, False


def normalise_code(lang: str, lines: list[str]) -> tuple[list[str], int]:
    """Drop translated comments and wildcard prose placeholders.

    Returns (comparable lines, number of lines whose comment was removed) so
    the caller can print how much was excluded.
    """
    strip_comments = lang in HASH_COMMENT
    out: list[str] = []
    excluded = 0
    for line in lines:
        kept = line
        if strip_comments:
            kept, cut = strip_trailing_comment(line)
            excluded += cut
        kept = PLACEHOLDER_RE.sub("<PLACEHOLDER>", kept)
        if kept.strip():
            out.append(kept)
    return out, excluded


def normalise_diagram(lines: list[str]) -> list[str]:
    """Blank the label text in a mermaid block, keep the graph skeleton.

    Node ids, edges, arrow directions and subgraph structure survive; the words
    a reader sees inside a box do not, because those are translated.
    """
    out = []
    for line in lines:
        line = MERMAID_QUOTE_RE.sub('""', line)
        line = MERMAID_PIPE_RE.sub("||", line)
        prev = None
        while prev != line:
            prev = line
            line = MERMAID_BRACKET_RE.sub(
                lambda m: m.group(0)[0] + m.group(0)[-1], line)
        line = line.strip()
        if line:
            out.append(line)
    return out


def headings(prose: list[tuple[int, str]]) -> list[int]:
    return [len(m.group(1)) for _, line in prose
            for m in [HEADING_RE.match(line)] if m]


def heading_texts(prose: list[tuple[int, str]]) -> list[str]:
    """Heading text with the leading #s removed, in document order."""
    return [line.lstrip("#").strip() for _, line in prose
            if HEADING_RE.match(line)]


def relative_links(prose: list[tuple[int, str]]) -> list[str]:
    """Relative link targets, in order, from prose only (not code blocks)."""
    found = []
    for _, line in prose:
        for m in LINK_RE.finditer(line):
            target = m.group(1)
            if EXTERNAL_RE.match(target):
                continue
            found.append(target)
    return found


def canonical(target: str) -> str:
    """Link target with the language suffix folded away.

    A page may point at either half of a pair; what must match between the two
    sides is *which document* is referenced, not which translation of it.
    """
    return target.replace(ZH_SUFFIX, ".md")


# --------------------------------------------------------------------------
# the ruling table: parsing, and the three checks that recompute what it used
# to state by hand
# --------------------------------------------------------------------------

# The page whose rulings drive checks 7-9. Only this one file is parsed for
# rulings: a second ruling table would be a second source of truth.
RULING_PAGE = "docs/glossary.zh-TW.md"

# A ruling table is identified by its header: four columns wide AND carrying
# the three operative column names. Width alone was the whole test until an
# audit showed what that buys: a four-column table of any content, anywhere on
# the page, was counted as rulings, so deleting a real ruling and adding a
# four-column "maintenance log" took the count from 64 to 66 and printed OK.
# The floor below was measuring table area, not rulings.
#
# The header names are the identity check the floor needs. They are also the
# three columns this page declares operative -- the fourth column is a comment
# and is deliberately excluded from the signature, because its heading differs
# per section ("為什麼(機制與碰撞)", "中文既有義", "為什麼不譯"...) and its
# content is not maintained.
#
# Tables of any other shape on that page (the two-column division-of-labour
# table, the three-column symbol registry) are not ruling tables and are not
# parsed. Inside a ruling table a row that is not four columns wide is an
# error, never a skip -- see parse_rulings.
RULING_COLUMNS = 4
RULING_HEADER = ("概念", "English original", "規定的中文寫法")

# Every count the checks below depend on gets a floor, and they all get it
# here. Two floors were added one at a time, each after the same defect was
# found one level down; a third (join keys) was then argued against on the
# grounds that each new floor is one more constant to keep in sync. That is an
# argument for a table, not against the guard -- a hand-copied guard is a
# promise that the next person adding a count will forget it.
#
# The defect all three share: the counter was a description of the input, not a
# check on it. A row widened to five columns, or a 不要用「X」 declaration
# deleted, quietly removed work from a later scan, and the line that scan
# printed still showed a plausible number followed by OK.
#
# What a floor catches: a count that shrank by accident.
#
# What a floor does NOT catch, by construction: someone keeping the number up
# on purpose -- lowering the floor in the same change, or padding a ruling
# table with a row that parses. Both are deliberate, both are plain in
# `git diff`, and neither is what this guard is for. Closing that would take an
# identity ledger of every ruling, i.e. one more hand-maintained list that
# rots; this page has already paid for that lesson six times over. The page's
# own maintenance section calls the list a ratchet: entries go in and are not
# removed on a hunch, so a genuine retirement lowers the number here in the
# same change.
COUNT_FLOORS = {
    # stats key      floor  what the count is, for the failure message
    "rulings":      (64, "rulings on " + "docs/glossary.zh-TW.md"),
    "banned_terms": (30, "renderings those rulings ban (不要用「X」)"),
    "bindings":     (9,  "Chinese-form/English-term bindings (「…」 in column 3)"),
}

# "不要用「X」" -- the one spelling a ruling uses to ban a rendering. One form
# only, so that a ban is greppable by a human and unambiguous to this script.
BAN_RE = re.compile(r"不要用「([^」]+)」")
# The same token, matched whole. This -- and nothing wider -- is what the ban
# scan deletes from a line of the ruling page before searching it.
BAN_DECL_RE = re.compile(r"不要用「[^」]*」")

# Anchors. A bare 〈section〉 points inside the same page; a `file.md` in
# backticks immediately before it (optionally through "的" or whitespace)
# points into that file. Both must resolve; neither may be a sentence.
ANCHOR_RE = re.compile(
    r"(?:`(?P<file>[^`\s]+\.md)`[的\s]*)?〈(?P<sec>[^〈〉]+)〉")
MD_REF_RE = re.compile(r"`((?:\.\.?/)*[A-Za-z0-9_][A-Za-z0-9_./-]*\.md)`")

# Not anchors: both README pages use `X.md` / `X.zh-TW.md` to name the pairing
# convention itself. They are placeholders in the same sense as the prose
# stand-ins inside angle brackets above, and the count skipped is printed.
MD_PLACEHOLDERS = {"X.md", "X.zh-TW.md"}


def _squash(s: str) -> str:
    """Drop all whitespace. Anchors wrap across lines; headings do not."""
    return re.sub(r"\s+", "", s)


SEP_CELL_RE = re.compile(r":?-{2,}:?")


def _tables(prose: list[tuple[int, str]]) -> list[list[tuple[int, list[str]]]]:
    """Contiguous markdown tables, each as a list of (line number, cells).

    Separator rows are kept: whichever row follows one is what identifies the
    header, so a table cannot be misread just because a section was renamed.
    """
    out: list[list[tuple[int, list[str]]]] = []
    cur: list[tuple[int, list[str]]] | None = None
    for n, line in prose:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            if cur is None:
                cur = []
            cur.append((n, [c.strip() for c in s[1:-1].split("|")]))
        elif cur is not None:
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def _is_sep(cells: list[str]) -> bool:
    return bool(cells) and all(SEP_CELL_RE.fullmatch(c) for c in cells)


def parse_rulings(path: Path, rep: Report | None = None,
                  where: str | None = None,
                  stats: dict | None = None) -> list[list[str]]:
    """Data rows of every ruling table on the ruling page.

    A table is a ruling table when its header is RULING_COLUMNS wide **and**
    its first three cells are RULING_HEADER. Both halves of that test earn
    their place:

    - The width test alone made the ruling count a count of four-column rows
      anywhere on the page, so a four-column table of unrelated content
      inflated it. That is how a deleted ruling was hidden: 64 - 1 + 3 = 66,
      above the floor, printed OK. A floor over a dilutable count is not a
      floor, and this is the same defect the floor itself was added to fix.
    - The header test alone would not survive a row being widened, which is
      the other way a ruling has been deleted in silence.

    A four-column table on this page whose header is *not* RULING_HEADER is
    reported rather than skipped. Silently ignoring it would make the
    dilution attempt harmless but invisible, and would also mean a genuinely
    new ruling table with a mistyped header stops being enforced without a
    word. Nothing on this page legitimately needs that shape: the two
    structural tables are two and three columns wide.

    Inside a ruling table **a row that is not RULING_COLUMNS wide is reported,
    not skipped**. When rows were filtered individually, giving one ruling a
    fifth column made that ruling vanish from the ban list and the join-key
    list without a word -- every counter shrank by one and the run still
    printed OK. A ruling the checker cannot parse is a ruling the checker is
    no longer enforcing, and that has to be loud.
    """
    out: list[list[str]] = []
    tables = 0
    for table in _tables(split_blocks(read_lines(path))[1]):
        seps = {i for i, (_, cells) in enumerate(table) if _is_sep(cells)}
        if not seps:
            continue                      # no header rule: not a real table
        head = min(seps) - 1
        if head < 0 or len(table[head][1]) != RULING_COLUMNS:
            continue                      # a table of some other shape
        header = tuple(c.strip("* ") for c in table[head][1][:3])
        if header != RULING_HEADER:
            if rep is not None:
                rep.fail(f"{where or path}:{table[head][0]}",
                         "four-column table on the ruling page whose header is "
                         "not a ruling header",
                         f"header:   {' | '.join(header)}"
                         f"\nexpected: {' | '.join(RULING_HEADER)}"
                         "\na four-column table that is not a ruling table "
                         "inflates the ruling count, and the floor below is "
                         "what a deleted ruling has to get past"
                         "\nuse the standard header, or give the table a "
                         "different width")
            continue
        tables += 1
        for i, (n, cells) in enumerate(table):
            if i == head or i in seps:
                continue
            if len(cells) != RULING_COLUMNS:
                if rep is not None:
                    rep.fail(f"{where or path}:{n}",
                             f"row of a ruling table has {len(cells)} columns, "
                             f"not {RULING_COLUMNS}",
                             "a row this checker cannot parse as a ruling is a "
                             "ruling it stops enforcing -- the ban and join-key "
                             "lists would both silently lose it"
                             f"\nrow: {'|'.join(cells)[:160]}")
                continue
            out.append([str(n)] + cells)
    if stats is not None:
        stats["ruling_tables_parsed"] = tables
    return out


def table_shapes(prose: list[tuple[int, str]]) -> list[tuple[int, ...]]:
    """Per-row column counts for each markdown table, in document order.

    Not (rows, columns): that pair was read off the *first* row only, so a
    single widened row anywhere below it left the shape unchanged and the two
    halves compared equal. One entry per data row, so a difference on any row
    is a difference.
    """
    out: list[tuple[int, ...]] = []
    cur: list[int] | None = None
    for _, line in prose:
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            if _is_sep(cells):
                continue                      # |---|---| separator
            if cur is None:
                cur = []
            cur.append(len(cells))
        elif cur is not None:
            out.append(tuple(cur))
            cur = None
    if cur is not None:
        out.append(tuple(cur))
    return out


def check_ruling_table_shape(rep: Report, root: Path, stats: dict) -> None:
    """The two halves of the ruling page must enumerate the same rulings.

    Only this pair. Elsewhere a table may legitimately exist on the Chinese
    side alone (the per-file term boxes are Chinese-only additions, ruled on in
    the glossary's own backfill section), so a tree-wide version of this check
    would fire on five pairs of correct documents.

    Here it is different: both halves are the same enumeration of rulings, and
    a row added to one half is a ruling the other half's readers never see --
    the exact failure this repo names, two copies with one updated.

    Compared row by row, not table by table: the earlier version compared
    (row count, column count) and so could not see a single row that had grown
    a column, which is precisely the edit that silently drops a ruling.
    """
    zh = root / RULING_PAGE
    en = zh.with_name(zh.name[: -len(ZH_SUFFIX)] + ".md")
    if not en.is_file():
        return
    a = table_shapes(split_blocks(read_lines(en))[1])
    b = table_shapes(split_blocks(read_lines(zh))[1])
    stats["ruling_tables"] = len(b)
    stats["ruling_rows"] = sum(len(t) for t in b)
    if a != b:
        i = _first_diff(a, b)
        detail = [f"first difference at table #{i + 1}",
                  f"EN: {_at(a, i)}", f"ZH: {_at(b, i)}"]
        ta, tb = _at2(a, i), _at2(b, i)
        if ta is not None and tb is not None and len(ta) == len(tb):
            j = _first_diff(list(ta), list(tb))
            detail.append(f"  row #{j + 1} of that table: "
                          f"{ta[j]} columns on the English side, "
                          f"{tb[j]} on the Chinese side")
        rep.fail(f"{en.relative_to(root)} / {zh.relative_to(root)}",
                 "the two halves of the ruling page do not hold the same "
                 "tables (per-row column counts, in order)",
                 "\n".join(detail))


def check_anchors(rep: Report, root: Path, pages: list[Path], stats: dict) -> None:
    """Every `file.md` reference exists; every 〈section〉 anchor is a heading.

    This is the check the anchoring rule always needed. The rule ("cite a
    filename plus a section name, never a quoted sentence") was introduced in
    one revision round and was already broken in that same round: a ruling
    cited a section of dependencies.zh-TW.md that had never existed. A rule
    about anchors with nothing resolving anchors is a rule about nothing.
    """
    headings_of: dict[Path, list[str]] = {}
    for p in pages:
        headings_of[p] = heading_texts(split_blocks(read_lines(p))[1])

    for page in pages:
        prose = split_blocks(read_lines(page))[1]
        # Joined, because an anchor may wrap across a line break.
        text = "\n".join(line for _, line in prose)
        line_of = _line_indexer(prose)

        for m in MD_REF_RE.finditer(text):
            ref = m.group(1)
            if ref in MD_PLACEHOLDERS:
                stats["anchor_placeholders"] += 1
                continue
            stats["anchor_files"] += 1
            if _resolve(root, page, ref) is None:
                rep.fail(f"{page.relative_to(root)}:{line_of(m.start())}",
                         f"`{ref}` does not resolve to a file")

        for m in ANCHOR_RE.finditer(text):
            ref, sec = m.group("file"), m.group("sec")
            target = page if ref is None else _resolve(root, page, ref)
            stats["anchor_sections"] += 1
            where = f"{page.relative_to(root)}:{line_of(m.start())}"
            if target is None:
                rep.fail(where, f"anchor 〈{sec}〉 names a file that does not "
                                f"resolve: `{ref}`")
                continue
            heads = headings_of.get(target)
            if heads is None:
                heads = heading_texts(split_blocks(read_lines(target))[1])
                headings_of[target] = heads
            if _squash(sec) in {_squash(h) for h in heads}:
                continue
            # Ranked by similarity rather than filtered by a cutoff: a cutoff
            # tuned on English words rejects everything for Chinese headings,
            # which have no spaces, and "<none similar>" helps nobody.
            near = [h for _, h in sorted(
                ((difflib.SequenceMatcher(None, _squash(sec), _squash(h)).ratio(), h)
                 for h in heads), key=lambda r: -r[0])][:3]
            rep.fail(where,
                     f"anchor 〈{sec}〉 is not a heading of "
                     f"{target.relative_to(root)}",
                     "closest headings there: "
                     + ("; ".join(near) if near else "<none similar>"))


def check_banned_forms(rep: Report, root: Path, pages: list[Path],
                       rulings: list[list[str]], stats: dict) -> None:
    """A ruled-out rendering must not survive anywhere in the doc tree.

    Exemption, stated exactly once so it cannot quietly widen: on
    docs/glossary.zh-TW.md, **the literal token `不要用「X」` is cut out of the
    line before the line is searched, and nothing else is.** Not the rest of
    that line, not the reason column beside it, not another line of the same
    page, and not the same words quoted in another file. The exemption used to
    be a whole line wide, which meant a banned rendering could sit in the
    reason column of its own ruling and this check said nothing -- an escape
    hatch exactly where the rulings are written.

    Scope is both languages, which needs an argument, because the English half
    legitimately contains Chinese: the prescribed-form column, and section
    anchors. Neither is a place a *banned* form may appear -- the
    prescribed-form column holds the approved rendering by construction, so a
    ban surfacing there is the page contradicting itself, which is worth an
    error rather than an exemption. Scanning English pages costs nothing on a
    clean tree and closes the one side nobody was looking at.
    """
    bans: dict[str, str] = {}          # term -> the exempting declaration
    for row in rulings:
        for term in BAN_RE.findall(" | ".join(row)):
            bans[term] = f"不要用「{term}」"
    stats["banned_terms"] = len(bans)
    stats["banned_pages"] = len(pages)
    ruling_page = root / RULING_PAGE
    for page in pages:
        is_ruling_page = page == ruling_page
        for n, line in split_blocks(read_lines(page))[1]:
            # The declaration token, and only the token, is removed.
            searched = BAN_DECL_RE.sub("", line) if is_ruling_page else line
            for term, decl in bans.items():
                if term not in searched:
                    if is_ruling_page and term in line:
                        stats["banned_exempt"] += 1
                    continue
                rep.fail(f"{page.relative_to(root)}:{n}",
                         f"ruled-out rendering 「{term}」 is still here",
                         f"the ruling that bans it: {decl}"
                         f"\n(only that token is exempt, not the line it is on)"
                         f"\nline: {line.strip()[:160]}")


def check_join_keys(rep: Report, root: Path, pairs: list[tuple[Path, Path]],
                    rulings: list[list[str]], stats: dict) -> None:
    """Where a ruled Chinese form appears, its English join key must be near.

    A ruling binds a Chinese rendering C to one English term E. The English
    term is the join key: rename it and the Chinese ruling stops matching
    silently. The test is per *section*, not per file, because a file-wide test
    is too weak to see the real defect -- journal.md does contain
    "production-scale", just not in the section whose Chinese heading claims
    it.

    A section passes if E appears in the English counterpart section, or if the
    Chinese section glosses E itself (the convention these docs already follow:
    print the English original at first use). Without the second clause the
    check would fire on every Chinese-only addition, e.g. journal.zh-TW.md's
    term table, which has no English counterpart section at all.
    """
    bound = []
    for row in rulings:
        form = row[3]                       # 規定的中文寫法
        if not form.startswith("「"):
            continue                        # no join key declared on this row
        c = form[1:].split("」", 1)[0].strip()
        e = row[2].strip().strip("`* ")     # English original
        if not c or not e or "/" in e:
            continue
        bound.append((c, e))
    stats["bindings"] = len(bound)

    for en, zh in pairs:
        en_sec = _sections(split_blocks(read_lines(en))[1])
        zh_sec = _sections(split_blocks(read_lines(zh))[1])
        if len(en_sec) != len(zh_sec):
            continue                        # already reported by the heading check
        for i, (zh_start, zh_body) in enumerate(zh_sec):
            zh_text = "\n".join(line for _, line in zh_body)
            en_text = "\n".join(line for _, line in en_sec[i][1])
            for c, e in bound:
                if c not in zh_text:
                    continue
                stats["binding_hits"] += 1
                pat = _term_re(e)
                if pat.search(en_text) or pat.search(zh_text):
                    continue
                n = next(n for n, line in zh_body if c in line)
                rep.fail(f"{zh.relative_to(root)}:{n}",
                         f"ruled form 「{c}」 is bound to the English term "
                         f"\"{e}\", which is absent here",
                         f"English counterpart section: "
                         f"{en.relative_to(root)}:{en_sec[i][0]} "
                         f"{_head_of(en_sec[i][1])}"
                         f"\nfix one side or the other -- the two are the same"
                         f" claim, and the binding is what makes them findable")


def _term_re(term: str) -> re.Pattern:
    """Match an English term, tolerating hyphen/space and case variation.

    "from-first-principles" and "from first principles" are the same join key;
    so are "production scale" and "production-scale". What must NOT be folded
    away is an extra word: "production run at full scale" does not contain the
    term, which is the whole point -- that phrasing is how the binding to
    `production-scale` got lost in the first place.
    """
    parts = [re.escape(w) for w in re.split(r"[\s-]+", term.strip()) if w]
    return re.compile(r"(?<![A-Za-z])" + r"[\s-]+".join(parts)
                      + r"(?![A-Za-z])", re.I)


def _sections(prose: list[tuple[int, str]]) -> list[tuple[int, list[tuple[int, str]]]]:
    """Split numbered prose at heading lines. The heading joins its section."""
    out: list[tuple[int, list[tuple[int, str]]]] = []
    for n, line in prose:
        if HEADING_RE.match(line) or not out:
            out.append((n, []))
        out[-1][1].append((n, line))
    return out


def _head_of(body: list[tuple[int, str]]) -> str:
    return body[0][1].strip()[:80] if body else ""


def _line_indexer(prose: list[tuple[int, str]]):
    """Map an offset in the joined prose text back to a real line number."""
    starts, numbers, pos = [], [], 0
    for n, line in prose:
        starts.append(pos)
        numbers.append(n)
        pos += len(line) + 1
    def at(offset: int) -> int:
        i = bisect.bisect_right(starts, offset) - 1
        return numbers[i] if 0 <= i < len(numbers) else 0
    return at


def _resolve(root: Path, page: Path, ref: str) -> Path | None:
    for cand in (page.parent / ref, root / ref):
        if cand.is_file():
            return cand
    return None


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []

    def fail(self, where: str, what: str, detail: str = "") -> None:
        msg = f"  {where}\n      {what}"
        if detail:
            msg += "\n" + "\n".join(f"      {d}" for d in detail.splitlines())
        self.problems.append(msg)


def check_header(rep: Report, path: Path, counterpart: Path, is_zh: bool,
                 lines: list[str]) -> None:
    """First line must be the language switcher pointing at the counterpart."""
    first = lines[0].strip() if lines else ""
    want_link = counterpart.name
    want_bold = "**繁體中文**" if is_zh else "**English**"
    targets = [m.group(1) for m in LINK_RE.finditer(first)]
    if not first.startswith(">") or want_link not in targets or want_bold not in first:
        expect = (f"> [English]({counterpart.name}) · **繁體中文**" if is_zh
                  else f"> **English** · [繁體中文]({counterpart.name})")
        rep.fail(f"{path}:1",
                 "missing or wrong language-switcher header line",
                 f"found:    {first or '<empty>'}\nexpected: {expect}")


def check_pair(rep: Report, en: Path, zh: Path, stats: dict) -> None:
    en_lines, zh_lines = read_lines(en), read_lines(zh)
    en_blocks, en_prose = split_blocks(en_lines)
    zh_blocks, zh_prose = split_blocks(zh_lines)

    check_header(rep, en, zh, is_zh=False, lines=en_lines)
    check_header(rep, zh, en, is_zh=True, lines=zh_lines)
    stats["headers"] += 2

    # -- headings -----------------------------------------------------------
    he, hz = headings(en_prose), headings(zh_prose)
    stats["headings"] += len(he) + len(hz)
    if he != hz:
        rep.fail(f"{en.name} / {zh.name}",
                 f"heading structure differs ({len(he)} on the English side, "
                 f"{len(hz)} on the Chinese side)",
                 f"EN levels: {he}\nZH levels: {hz}\n"
                 f"first divergence at heading #{_first_diff(he, hz) + 1}")

    # -- code blocks --------------------------------------------------------
    if len(en_blocks) != len(zh_blocks):
        rep.fail(f"{en.name} / {zh.name}",
                 f"different number of code blocks "
                 f"({len(en_blocks)} English, {len(zh_blocks)} Chinese)")
    for i, (be, bz) in enumerate(zip(en_blocks, zh_blocks)):
        lang_e, line_e, body_e = be
        lang_z, line_z, body_z = bz
        if lang_e != lang_z:
            rep.fail(f"{en.name}:{line_e} / {zh.name}:{line_z}",
                     f"code block #{i} has a different fence language "
                     f"({lang_e or '<none>'} vs {lang_z or '<none>'})")
            continue
        if lang_e in DIAGRAM_FENCES:
            stats["diagrams"] += 1
            norm_e, norm_z = normalise_diagram(body_e), normalise_diagram(body_z)
            stats["diagram_lines"] += len(norm_e)
            what = (f"diagram block #{i} ({lang_e}) has a different structure "
                    f"-- labels may be translated, the graph may not")
        else:
            stats["blocks"] += 1
            norm_e, cut_e = normalise_code(lang_e, body_e)
            norm_z, cut_z = normalise_code(lang_z, body_z)
            stats["code_lines"] += len(norm_e)
            stats["comment_lines"] += cut_e + cut_z
            what = (f"code block #{i} ({lang_e or 'no language'}) differs "
                    f"outside comments -- commands must be identical")
        if norm_e != norm_z:
            j = _first_diff(norm_e, norm_z)
            rep.fail(f"{en.name}:{line_e} / {zh.name}:{line_z}", what,
                     f"EN: {_at(norm_e, j)}\nZH: {_at(norm_z, j)}")

    # -- link sets ----------------------------------------------------------
    # The language-switcher link points at the counterpart, i.e. at this same
    # document in the other language; after canonical() that is a link to
    # self. Dropping self-links removes it from both sides without having to
    # assume it is on any particular line -- assuming it was on line 1 gave a
    # false positive on every English page that has no switcher line yet.
    own = canonical(zh.name)
    le = {c for t in relative_links(en_prose) if (c := canonical(t)) != own}
    lz = {c for t in relative_links(zh_prose) if (c := canonical(t)) != own}
    stats["links_compared"] += len(le) + len(lz)
    if le != lz:
        detail = []
        if le - lz:
            detail.append(f"only on the English side: {sorted(le - lz)}")
        if lz - le:
            detail.append(f"only on the Chinese side: {sorted(lz - le)}")
        rep.fail(f"{en.name} / {zh.name}",
                 "the two sides do not link to the same set of documents",
                 "\n".join(detail))


def check_dead_links(rep: Report, path: Path, stats: dict) -> None:
    _, prose = split_blocks(read_lines(path))
    for target in relative_links(prose):
        file_part = target.split("#", 1)[0]
        if not file_part:
            continue
        stats["links_resolved"] += 1
        if not (path.parent / file_part).exists():
            rep.fail(str(path), f"dead relative link: {target}")


def _first_diff(a: list, b: list) -> int:
    for i in range(max(len(a), len(b))):
        if i >= len(a) or i >= len(b) or a[i] != b[i]:
            return i
    return -1


def _at(seq: list, i: int) -> str:
    return repr(seq[i]) if 0 <= i < len(seq) else "<end of block>"


def _at2(seq: list, i: int):
    """The element itself, or None past the end. Used where the caller wants
    to look inside the differing element instead of just printing it."""
    return seq[i] if 0 <= i < len(seq) else None


def check_count_floors(rep: Report, stats: dict) -> None:
    """Every count a later check depends on, against its floor. See COUNT_FLOORS.

    Runs last, after every counter is populated, so one loop covers all of
    them and a count added later gets a floor by appearing in that dict rather
    than by someone remembering to write a fourth `if`.
    """
    for key, (floor, what) in COUNT_FLOORS.items():
        got = stats.get(key)
        if got is None:
            rep.fail(RULING_PAGE,
                     f"COUNT_FLOORS names a count this run never produced: "
                     f"{key!r}",
                     "either the counter was renamed and this floor now guards "
                     "nothing, or the check that sets it did not run"
                     "\nan unguarded count is the defect this table exists to "
                     "prevent, so this is an error rather than a skip")
            continue
        if got < floor:
            rep.fail(RULING_PAGE,
                     f"only {got} {what}, fewer than the {floor} this page is "
                     f"known to carry",
                     f"the checks built on that count just got weaker by "
                     f"{floor - got}, and the line printing it would still have "
                     "shown a plausible number followed by OK"
                     f"\nif one was genuinely retired, lower the {key!r} floor "
                     "in COUNT_FLOORS (scripts/check_bilingual.py) in the same "
                     "change")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    pages: list[Path] = []
    # What the summary line has to name. Built here rather than written out
    # below, so the printed denominator cannot drift from the scan.
    scanned_where: list[str] = []
    if SCAN_ROOT:
        pages += sorted(p for p in root.glob("*.md") if p.is_file())
        scanned_where.append("the repo root (top level only)")
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            print(f"error: {base} does not exist -- wrong root?", file=sys.stderr)
            return 2
        pages += sorted(p for p in base.rglob("*.md") if p.is_file())
        scanned_where.append(d + "/")

    zh_pages = [p for p in pages if p.name.endswith(ZH_SUFFIX)]
    en_pages = [p for p in pages if not p.name.endswith(ZH_SUFFIX)]

    rep = Report()
    stats = dict(headers=0, headings=0, blocks=0, diagrams=0,
                 code_lines=0, comment_lines=0, diagram_lines=0,
                 links_compared=0, links_resolved=0,
                 anchor_files=0, anchor_sections=0, anchor_placeholders=0,
                 rulings=0, ruling_tables=0, ruling_rows=0,
                 ruling_tables_parsed=0,
                 banned_terms=0, banned_pages=0, banned_exempt=0,
                 bindings=0, binding_hits=0)

    pairs: list[tuple[Path, Path]] = []
    for zh in zh_pages:
        en = zh.with_name(zh.name[: -len(ZH_SUFFIX)] + ".md")
        if en.exists():
            pairs.append((en, zh))
        else:
            rep.fail(str(zh.relative_to(root)),
                     f"orphan: no English counterpart at {en.name}")

    for en, zh in pairs:
        check_pair(rep, en, zh, stats)
    for page in pages:
        check_dead_links(rep, page, stats)

    ruling_page = root / RULING_PAGE
    if not ruling_page.is_file():
        print(f"error: {ruling_page} is missing -- the ruling table is the "
              f"input to checks 7-9, and its absence is not 'clean'",
              file=sys.stderr)
        return 2
    rulings = parse_rulings(ruling_page, rep, RULING_PAGE, stats)
    stats["rulings"] = len(rulings)
    if not rulings:
        rep.fail(RULING_PAGE,
                 "no rulings parsed -- checks 8 and 9 would pass vacuously",
                 f"expected tables whose header is {RULING_COLUMNS} columns "
                 f"wide; found none")
    check_ruling_table_shape(rep, root, stats)
    check_anchors(rep, root, pages, stats)
    check_banned_forms(rep, root, pages, rulings, stats)
    check_join_keys(rep, root, pairs, rulings, stats)
    check_count_floors(rep, stats)

    # Membership in the scan, not existence on disk. Those two came apart on
    # the root pair: README.zh-TW.md existed, so README.md was not listed as
    # English-only, but it was never collected, so no pair was formed and
    # nothing was compared. "0 English-only pages" then read as "everything is
    # paired" while naming a set the scan had never entered. This list is a
    # denominator, not an error -- but it has to be a denominator over what was
    # actually scanned.
    scanned = set(pages)
    monolingual = [p for p in en_pages
                   if p.with_name(p.name[:-3] + ZH_SUFFIX) not in scanned]

    rel = lambda p: p.relative_to(root)  # noqa: E731
    print(f"bilingual drift check -- {root}")
    print(f"scanned {len(pages)} markdown pages under "
          f"{', '.join(scanned_where)}")
    print(f"  {len(pairs)} bilingual pairs:")
    for en, zh in pairs:
        print(f"      {rel(en)}  <->  {rel(zh)}")
    print(f"  {len(monolingual)} English-only pages (not an error, listed so "
          f"a missing pair is visible):")
    for p in monolingual:
        print(f"      {rel(p)}")
    print()
    print("checked:")
    print(f"  header links     {stats['headers']} page headers")
    print(f"  headings         {stats['headings']} headings across {len(pairs)} pairs")
    print(f"  code blocks      {stats['blocks']} blocks, "
          f"{stats['code_lines']} code lines compared, "
          f"{stats['comment_lines']} comments excluded "
          f"(translated by the docs' own rule)")
    print(f"  diagrams         {stats['diagrams']} mermaid blocks, "
          f"{stats['diagram_lines']} skeleton lines compared "
          f"(labels blanked: they are prose)")
    print(f"  link sets        {stats['links_compared']} relative link targets "
          f"across {len(pairs)} pairs")
    print(f"  dead links       {stats['links_resolved']} relative targets resolved "
          f"across {len(pages)} pages")
    print(f"  anchors          {stats['anchor_files']} `*.md` references and "
          f"{stats['anchor_sections']} 〈section〉 anchors resolved "
          f"({stats['anchor_placeholders']} placeholders skipped)")
    print(f"  banned forms     {stats['banned_terms']} ruled-out renderings "
          f"(floor {COUNT_FLOORS['banned_terms'][0]}) searched across all "
          f"{stats['banned_pages']} "
          f"pages, both languages "
          f"({stats['banned_exempt']} 不要用「X」 tokens exempt -- the token, "
          f"not the line)")
    print(f"  join keys        {stats['bindings']} Chinese-form/English-term "
          f"bindings (floor {COUNT_FLOORS['bindings'][0]}), "
          f"{stats['binding_hits']} section occurrences verified")
    print(f"  ruling table     {stats['ruling_tables']} tables / "
          f"{stats['ruling_rows']} rows compared column-count-for-column-count "
          f"between the two halves of the ruling page")
    print(f"  rulings read     {stats['rulings']} rows from "
          f"{stats['ruling_tables_parsed']} tables headed "
          f"{'/'.join(RULING_HEADER)} in {RULING_PAGE} "
          f"(floor {COUNT_FLOORS['rulings'][0]}; the source of the four "
          f"checks above)")
    print()

    if not pairs:
        print("FAIL: no bilingual pairs found at all -- the check scanned "
              "nothing, which is not the same as clean.")
        return 1
    if rep.problems:
        print(f"FAIL: {len(rep.problems)} problem(s)")
        print()
        for p in rep.problems:
            print(p)
            print()
        return 1
    print("OK: no drift found in the checks listed above. "
          "Prose is NOT checked -- that still needs a reader.")
    return 0


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def self_test() -> int:
    """Pin down the rules that would otherwise widen without anyone noticing.

    The exemption in check_banned_forms is the dangerous one: it is the single
    place where this script is told to ignore a match. An exemption that grows
    by one plausible-sounding step at a time ends up swallowing the real hits,
    and nothing about a passing run would show it. So it is nailed down here,
    on synthetic input, and `make bilingual` runs this before it runs anything
    else.

    Pinned alongside it: that a ruling row which is not four columns wide is
    an error rather than a skip. Widening one row by a column was a way to
    delete a ruling in silence -- the ban list shrank by one, the shape check
    compared only headers, and the run still said OK.
    """
    import tempfile
    failures: list[str] = []

    def expect(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}\n      got:  {got}\n      want: {want}")

    ruling_rows = [
        "| 概念 | production | 「正式運轉」 | **不要用「產線」**——那兩個字在中文第一義是製造流水線 |",
        "| 概念 | rasterize | `rasterize` | **不要用「光柵化」**;**不要用「重新離散成幾何」** |",
    ]
    header = "| " + " | ".join(RULING_HEADER) + " | 為什麼 |"
    page = ["> header", "", "# T", "", header, "|---|---|---|---|",
            *ruling_rows, "", "| three | column | table |", "|---|---|---|",
            "| x | y | z |", "", "| two | col |", "|---|---|", "| p | q |"]
    ROW0 = 6                              # index of ruling_rows[0] in `page`
    # The audit's dilution attack, verbatim: a four-column table of unrelated
    # content elsewhere on the ruling page. It used to be counted as rulings,
    # which is how deleting a real one stayed above the rulings floor.
    decoy = ["", "| 維護紀錄 | 日期 | 誰 | 備註 |", "|---|---|---|---|",
             "| 修了理由欄 | 2026-08-30 | someone | 沒有裁定 |",
             "| 又修了一次 | 2026-08-30 | someone | 沒有裁定 |"]

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "docs").mkdir()
        ruling = root / RULING_PAGE
        ruling.write_text("\n".join(page) + "\n", encoding="utf-8")

        rep0 = Report()
        rulings = parse_rulings(ruling, rep0, RULING_PAGE)
        # Only rows of a table whose header is 4 columns wide are rulings;
        # header and separator rows are not, and the 3- and 2-column tables on
        # the same page are not.
        expect("parse_rulings picks up exactly the four-column tables' rows",
               [r[1:] for r in rulings],
               [[c.strip() for c in r.strip("|").split("|")] for r in ruling_rows])
        expect("a well-formed ruling page reports nothing", rep0.problems, [])

        # A row with a fifth column: reported, and not silently dropped into
        # the "not a ruling" bin. This is the audit's own planted positive.
        widened = list(page)
        widened[ROW0] = ruling_rows[0] + " 多出來的一欄 |"
        ruling.write_text("\n".join(widened) + "\n", encoding="utf-8")
        rep1 = Report()
        wide_rulings = parse_rulings(ruling, rep1, RULING_PAGE)
        expect("a 5-column ruling row is reported, not skipped",
               len(rep1.problems), 1)
        expect("...and the widened row is no longer counted as a ruling",
               len(wide_rulings), len(rulings) - 1)
        ruling.write_text("\n".join(page) + "\n", encoding="utf-8")

        # A four-column table that is not a ruling table: reported, and it
        # does NOT raise the ruling count. Without both halves, deleting a
        # ruling and adding a table like this stays above the rulings floor --
        # the count was of four-column rows, not of rulings.
        ruling.write_text("\n".join(page + decoy) + "\n", encoding="utf-8")
        rep2 = Report()
        diluted = parse_rulings(ruling, rep2, RULING_PAGE)
        expect("a four-column non-ruling table does not inflate the count",
               len(diluted), len(rulings))
        expect("...and it is reported rather than silently ignored",
               len(rep2.problems), 1)
        ruling.write_text("\n".join(page) + "\n", encoding="utf-8")

        # The header is the identity: rename a ruling table's columns and its
        # rows stop being rulings, loudly. The floor then sees the drop.
        renamed = list(page)
        renamed[4] = "| 項目 | English original | 規定的中文寫法 | 為什麼 |"
        ruling.write_text("\n".join(renamed) + "\n", encoding="utf-8")
        rep3 = Report()
        expect("a ruling table with a renamed header stops being parsed",
               len(parse_rulings(ruling, rep3, RULING_PAGE)), 0)
        expect("...and that is reported too", len(rep3.problems), 1)
        ruling.write_text("\n".join(page) + "\n", encoding="utf-8")

        # parse_rulings reports how many ruling tables it identified, so the
        # printed line names its own input instead of just its output size.
        st: dict = {}
        parse_rulings(ruling, None, RULING_PAGE, st)
        expect("the number of ruling tables is reported",
               st.get("ruling_tables_parsed"), 1)

        # Per-row shapes, not (rows, columns): the widened row must change the
        # shape of its table, or the two halves compare equal.
        expect("table_shapes records one column count per row",
               table_shapes(split_blocks(page)[1]),
               [(4, 4, 4), (3, 3), (2, 2)])
        # (header, first data row, second data row) -- the widened row is the
        # middle entry, which the old (rows, columns) pair could not express.
        expect("a single widened row changes that table's shape",
               table_shapes(split_blocks(widened)[1])[0], (4, 5, 4))

        def hits(rel: str, body: str) -> list[str]:
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("> header\n\n# T\n\n" + body + "\n", encoding="utf-8")
            rep = Report()
            check_banned_forms(rep, root, [p], rulings,
                               dict(banned_terms=0, banned_pages=0, banned_exempt=0))
            p.unlink()
            return rep.problems

        # 1. The declaring token, on the ruling page, is exempt -- for that term.
        expect("declaring line is exempt",
               len(hits(RULING_PAGE, ruling_rows[0])), 0)
        # 2. ...and only for that term. A different banned term on the same
        #    exempt line is still a hit.
        expect("exemption does not spill to a second term on the same line",
               len(hits(RULING_PAGE, ruling_rows[0] + " 光柵化")), 1)
        # 3. The exemption is the token, not the line: a banned term repeated
        #    in the reason column of its own ruling is a real hit. A line-wide
        #    exemption is where the audit hid 稜線 in the ridge ruling.
        expect("a second mention on the declaring line itself is a hit",
               len(hits(RULING_PAGE,
                        "| 概念 | production | 「正式運轉」 | **不要用「產線」**"
                        "——「產線」在中文第一義是製造流水線 |")), 1)
        # 4. Any other line of the ruling page is checked like everywhere else.
        expect("second mention elsewhere on the ruling page is a hit",
               len(hits(RULING_PAGE, "說明:「產線」那一條的由來")), 1)
        # 5. The declaration text does not exempt anything in another file --
        #    otherwise quoting the ruling would be a way to keep using the word.
        expect("declaration text in another file does not exempt",
               len(hits("docs/other.zh-TW.md", "不要用「產線」,但這裡還是寫了產線")), 1)
        # 6. English pages are in scope too: the prescribed-form column is the
        #    approved rendering, so a ban surfacing on the English half is the
        #    page contradicting itself, not a false positive to exempt.
        expect("the English half is scanned as well",
               len(hits("docs/other.md", "the banned rendering 產線 sits here")), 1)
        # 7. A clean page is clean.
        expect("no false positive on unrelated text",
               len(hits("docs/other.zh-TW.md", "這一段沒有任何被裁定禁用的寫法")), 0)

    # Join-key term matching: hyphen and case fold, an inserted word does not.
    expect("hyphen folds", bool(_term_re("from first principles")
                                .search("a from-first-principles count")), True)
    expect("case folds", bool(_term_re("production-scale")
                              .search("at Production Scale")), True)
    expect("an inserted word is not the term",
           bool(_term_re("production-scale").search("production run at full scale")),
           False)
    expect("no match inside a longer word",
           bool(_term_re("rank").search("frankly")), False)

    # A floor only bites if it is a real number. Setting one to 0 restores the
    # old "only zero is a problem" behaviour -- "30 renderings searched" and
    # "29 renderings searched" print identically to a reader who is not
    # counting -- so that edit has to fail here rather than pass quietly. One
    # assertion over the whole table, so a count added later is covered by
    # appearing in COUNT_FLOORS rather than by someone adding a fourth expect.
    expect("every floor in COUNT_FLOORS is a real floor, not a disabled one",
           sorted(k for k, (f, _) in COUNT_FLOORS.items() if f <= 0), [])
    # And that the table is not empty, which would disable all of them at once.
    expect("COUNT_FLOORS is not empty", bool(COUNT_FLOORS), True)

    if failures:
        print(f"SELF-TEST FAIL: {len(failures)} assertion(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("self-test OK: ruling parsing, ban exemption and join-key matching "
          "behave as documented")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main())
