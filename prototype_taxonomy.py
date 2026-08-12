"""PROTOTYPE — test whether deterministic expression strata are auditable.

Question: can small, dependency-free lexical rules separate direct retrieval,
absolute position, relational reference, and explicit logical operators without
pretending to infer latent reasoning? Run: python3 prototype_taxonomy.py --check
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass


WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")

LEXICONS = {
    "attribute": {
        "beige", "black", "blue", "brown", "colorful", "dark", "gold",
        "gray", "green", "grey", "light", "orange", "pink", "purple",
        "red", "silver", "striped", "white", "wooden", "yellow",
    },
    "absolute": {
        "background", "bottom", "bottommost", "center", "central", "foreground",
        "leftmost", "middle", "rear", "rightmost", "top", "topmost",
    },
    "comparison": {
        "biggest", "closest", "farthest", "fewest", "furthest", "highest",
        "largest", "least", "longest", "lowest", "most", "nearest",
        "same", "shortest", "smaller", "smallest", "tallest",
    },
    "ordinal": {
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
        "eighth", "ninth", "tenth",
    },
    "cardinality": {
        "both", "couple", "double", "eight", "five", "four", "nine", "pair",
        "seven", "six", "three", "triple", "two",
    },
    "negation": {
        "aren't", "cannot", "can't", "doesn't", "don't", "except", "isn't",
        "neither", "no", "nor", "not", "nothing", "without",
    },
}

RELATIONS = (
    "attached to", "behind", "below", "beside", "between", "carrying",
    "close to", "covered by", "crossing", "facing", "far from", "holding",
    "in front of", "inside", "left of", "looking at", "near", "next to",
    "on top of", "outside", "owned by", "ridden by", "riding", "right of",
    "under", "underneath", "wearing", "with", "worn by",
)

ABSOLUTE_PHRASES = (
    "at the left", "at the right", "in the left", "in the right",
    "on the left", "on the right",
)


@dataclass(frozen=True)
class Classification:
    stratum: str
    tags: tuple[str, ...]
    relation_count: int
    token_count: int
    compositional: bool


def classify(expression: str) -> Classification:
    normalized = " ".join(WORDS.findall(expression.lower()))
    tokens = normalized.split()
    token_set = set(tokens)
    tags = {name for name, words in LEXICONS.items() if token_set & words}
    if any(token.isdigit() for token in tokens):
        tags.add("cardinality")
    if any(phrase in normalized for phrase in ABSOLUTE_PHRASES):
        tags.add("absolute")
    relation_count = sum(normalized.count(phrase) for phrase in RELATIONS)
    if relation_count:
        tags.add("relation")

    logical = tags & {"comparison", "ordinal", "cardinality", "negation"}
    if logical:
        stratum = "logical"
    elif "relation" in tags:
        stratum = "relational"
    elif "absolute" in tags:
        stratum = "absolute"
    else:
        stratum = "direct"

    structural_tags = tags - {"attribute"}
    compositional = relation_count >= 2 or len(structural_tags) >= 2
    return Classification(
        stratum=stratum,
        tags=tuple(sorted(tags)),
        relation_count=relation_count,
        token_count=len(tokens),
        compositional=compositional,
    )


CASES = {
    "the dog": ("direct", (), False),
    "the red cup": ("direct", ("attribute",), False),
    "person in the background": ("absolute", ("absolute",), False),
    "the cup next to the laptop": ("relational", ("relation",), False),
    "the woman wearing red beside the child": (
        "relational", ("attribute", "relation"), True,
    ),
    "the second person from the right": ("logical", ("ordinal",), False),
    "the smaller pot in front of the pan": (
        "logical", ("comparison", "relation"), True,
    ),
    "the person not holding anything": (
        "logical", ("negation", "relation"), True,
    ),
    "the elephant ridden by three people": (
        "logical", ("cardinality", "relation"), True,
    ),
    "2 dogs on the left": (
        "logical", ("absolute", "cardinality"), True,
    ),
}


def check() -> None:
    for expression, expected in CASES.items():
        actual = classify(expression)
        got = (actual.stratum, actual.tags, actual.compositional)
        assert got == expected, (expression, got, expected)
    print(f"ok: {len(CASES)} taxonomy boundary cases")


def main() -> None:
    if sys.argv[1:] == ["--check"]:
        check()
        return
    print("Enter expressions; blank line exits.")
    while expression := input("> ").strip():
        print(classify(expression))


if __name__ == "__main__":
    main()
