"""Nothing this project prints to a console may contain a non-ASCII character.

This is `AGENTS.md` rule I3, and it is earned rather than stylistic: a `->` written as an arrow in a
progress print crashed `_download_file()` with `UnicodeEncodeError` **after a 189 MB file had already fully
downloaded**, and a `--help` text crashed the same way for the same reason. Windows' default console codepage
is cp1252, so the exception lands wherever the string does -- usually at the end of the expensive part.

**It is a latent fault, not a dormant style issue.** Whether it fires depends on the codepage of whatever
console the script happens to run on, which is not a property of this repository. A string that is safe on
one machine raises on the next.

**Why an AST walk and not a grep.** The question is what a string is *for*. A non-ASCII character in a
docstring, a comment or a data literal is fine -- `docs/` and `memory/` are full of em dashes and this file
does not care -- and the same character inside a `print()` argument or an argparse `help=` is a crash
waiting for the wrong terminal. Only the use distinguishes them, so the test reads the syntax tree.

Sources are read with `utf-8-sig`, because four Python files here start with a byte-order mark and
`ast.parse` on text read as plain `utf-8` dies with `SyntaxError: invalid non-printable character U+FEFF`
(`memory/gotchas.md`). The gitignored `include/` tree is skipped: it holds the upstream Li2Former sources,
which are GPL-3.0, are not ours to edit, and are scored as released (`memory/dos-and-donts.md`).
"""
import ast
import sys
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

if str(REPO / "utils") not in sys.path:
    sys.path.insert(0, str(REPO / "utils"))
if str(REPO / "library") not in sys.path:
    sys.path.insert(0, str(REPO / "library"))

#: This project's own Python. Named rather than globbed from the repository root so that a new directory has
#: to be added here deliberately, and so the count below means something.
ROOTS = ("library/takheliped", "utils", "tests", "deploy", "memory/scripts")

#: `include/` is the downloaded dataset-and-weights tree, which carries upstream sources we do not edit.
SKIP = "include"

#: argparse keywords whose text `--help` prints.
HELP_KEYWORDS = frozenset({"help", "description", "epilog", "metavar", "usage", "prog", "title"})

#: Calls whose string arguments reach a stream. `write` and `warn` are matched on the attribute name, so
#: `sys.stdout.write` and `warnings.warn` are both caught without importing anything.
WRITERS = frozenset({"print", "write", "warn", "showwarning"})

#: What to write instead. Every one of these is a character that has appeared in this repository's own
#: console output; the replacement is what `AGENTS.md` I3 asks for.
ASCII_FOR = {"—": "--", "–": "-", "…": "...", "×": "x", "§": "section ",
             "→": "->", "°": " deg", "±": "+-", "≈": "~", "≤": "<=", "≥": ">=",
             "•": "*", "∝": "proportional to", "−": "-", "·": "*", "│": "|",
             "─": "-", "├": "|-", "└": "`-"}


def sources():
    """Every one of this project's own Python files, as (repo-relative name, text) pairs."""
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if SKIP not in path.relative_to(REPO).parts:
                yield path.relative_to(REPO).as_posix(), path.read_text(encoding="utf-8-sig")


def strings_in(node):
    """Every string constant inside `node`, which covers an f-string's literal parts and implicit
    concatenation -- the two forms a line-by-line search misses."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def called_name(call):
    if isinstance(call.func, ast.Name):
        return call.func.id
    return call.func.attr if isinstance(call.func, ast.Attribute) else ""


def prints_its_docstring(tree):
    """Whether this module hands `__doc__` to something that prints it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in HELP_KEYWORDS and isinstance(keyword.value, ast.Name) and keyword.value.id == "__doc__":
                    return True
            if called_name(node) in WRITERS:
                if any(isinstance(arg, ast.Name) and arg.id == "__doc__" for arg in node.args):
                    return True
    return False


def console_strings(text, filename="<test>"):
    """Every (line, what it is, the string) this file hands to a console.

    **A module docstring counts when the module prints it**, which almost every script here does:
    `ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)` puts the whole docstring
    in `--help` verbatim. That is the form the original `--help` crash took, and it is invisible to a search
    for string literals, because the argument is the name `__doc__` rather than any text."""
    tree = ast.parse(text, filename=filename)
    if prints_its_docstring(tree) and (doc := ast.get_docstring(tree, clean=False)):
        yield 1, "description=__doc__", doc
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if called_name(node) in WRITERS:
                for arg in node.args:
                    for value in strings_in(arg):
                        yield node.lineno, called_name(node) + "()", value
            for keyword in node.keywords:
                if keyword.arg in HELP_KEYWORDS:
                    for value in strings_in(keyword.value):
                        yield node.lineno, keyword.arg + "=", value
        # A traceback is printed with the same codepage as stdout, so an exception message is console output
        # too -- and the one place a crash message can itself crash.
        elif isinstance(node, ast.Raise) and node.exc is not None:
            for value in strings_in(node.exc):
                yield node.lineno, "raise", value


def offenders(text, filename="<test>"):
    for line, kind, value in console_strings(text, filename):
        bad = sorted({c for c in value if ord(c) > 126})
        if bad:
            yield line, kind, bad, value


class TestNothingThisProjectPrintsIsNonAscii:
    def test_no_console_string_anywhere_holds_a_non_ascii_character(self):
        found = []
        for name, text in sources():
            for line, kind, bad, value in offenders(text, name):
                codes = " ".join(f"U+{ord(c):04X} {ASCII_FOR.get(c, '?')!r}" for c in bad)
                found.append(f"  {name}:{line} in {kind}: {codes}\n"
                             f"      {value[:110].encode('ascii', 'backslashreplace').decode()}")
        assert not found, ("non-ASCII characters reach a console from %d place(s) (AGENTS.md I3):\n%s"
                           % (len(found), "\n".join(found)))

    def test_the_walk_actually_reaches_this_project(self):
        """A scan that matches nothing passes. These two numbers are what stops that being mistaken for a
        clean result; raise them when the project grows, never lower them to make the suite green."""
        names = [name for name, _ in sources()]

        assert len(names) > 140, f"only {len(names)} files scanned, so the roots are wrong"
        assert all(any(name.startswith(root) for root in ROOTS) for name in names)
        for root in ROOTS:
            assert any(name.startswith(root) for name in names), f"{root} contributed nothing"

    def test_the_upstream_tree_is_not_scanned(self):
        """It is GPL-3.0, it is not ours to edit, and it is scored as released."""
        assert not [name for name, _ in sources() if "include" in name]


class TestTheScannerSeesWhatItClaimsTo:
    """Each of these is a form the fault has actually taken here, or a form a line-based search would miss."""

    @pytest.mark.parametrize("source, kind", [
        ('print("No GPU found — using CPU.")', "print()"),
        ('ap.add_argument("--dpi", help="Figure DPI — controls resolution")', "help="),
        ('ap = ArgumentParser(description="a sweep — over radii")', "description="),
        ('sys.stdout.write("done …")', "write()"),
        ('warnings.warn("deprecated — use the other one")', "warn()"),
        ('raise SystemExit("cannot continue — no odometry")', "raise"),
        ('print(f"{n} trials × {m} epochs")', "print()"),
        ('print("a long message "\n      "continued — here")', "print()"),
    ])
    def test_it_flags(self, source, kind):
        found = list(offenders(source))

        assert len(found) == 1, found
        assert found[0][1] == kind

    @pytest.mark.parametrize("source", [
        '"""A docstring with an em dash — which is fine."""',
        '# a comment with an em dash — which is fine\nprint("ascii only")',
        'TITLE = "a data literal — never printed"',
        'print("plain ascii -- fine")',
        'ap.add_argument("--x", help="plain ascii -- fine")',
        'label = "—".join(parts)',
    ])
    def test_it_leaves_alone(self, source):
        assert not list(offenders(source))

    def test_it_parses_a_file_that_starts_with_a_byte_order_mark(self):
        """Four Python files here do, and plain `utf-8` turns that into a SyntaxError rather than a finding."""
        with_bom = '﻿' + 'print("an em dash — here")'

        assert len(list(offenders(with_bom.lstrip("﻿")))) == 1
        with pytest.raises(SyntaxError):
            ast.parse(with_bom)

    def test_every_character_it_reports_has_an_ascii_replacement(self):
        """So a failure tells the next reader what to write, instead of only that something is wrong."""
        for character, replacement in ASCII_FOR.items():
            assert ord(character) > 126 and replacement.isascii()


class TestEveryHelpTextRenders:
    """The other way `--help` fails on this machine, and it was failing at HEAD when this was written.

    `argparse` expands a help string through `help % params`, so a literal per cent sign in it is read as a
    format specifier: `"cost 24% of training time"` raised `TypeError: %o format: an integer is required, not
    dict` and `train_takheliped.py step1 --help` exited 1 after printing most of its text. The ASCII walk above
    cannot see it, because every character involved is ASCII -- what is wrong is the escaping, and the only
    thing that tells you is rendering the text.

    Rendering is also all this needs: `format_help()` is what `--help` calls, so a parser that formats here
    cannot crash there.

    `train.py` and `evaluate.py` are reached only by name: both build their parser inside `main()`, so there is
    nothing to call here, and their help text is covered by the ASCII walk above alone.
    """

    def _parsers(self):
        for script in ("train_takheliped", "empty_scene", "train", "evaluate"):
            module = import_module(script)
            if not hasattr(module, "build_parser"):
                continue
            parser = module.build_parser()
            yield script, parser
            for group in parser._subparsers._group_actions if parser._subparsers else ():
                for name, sub in getattr(group, "choices", {}).items():
                    yield f"{script} {name}", sub

    def test_every_parser_and_subcommand_formats_its_help(self):
        formatted = {}
        for name, parser in self._parsers():
            formatted[name] = parser.format_help()          # raises on an unescaped per cent sign
        assert set(formatted) >= {"train_takheliped", "train_takheliped step1", "train_takheliped step2",
                                  "train_takheliped step3a", "train_takheliped step3b", "train_takheliped step4",
                                  "train_takheliped score"}, sorted(formatted)
        for name, text in formatted.items():
            assert text.isascii(), name

    def test_an_unescaped_per_cent_sign_is_what_this_catches(self):
        """The red half, pinned: the guard has to fail on the exact text that was in the file."""
        parser = ArgumentParser()
        parser.add_argument("--chunk-s", help="the 1 s warm-up tied and cost 24% of training time")
        with pytest.raises(TypeError):
            parser.format_help()
