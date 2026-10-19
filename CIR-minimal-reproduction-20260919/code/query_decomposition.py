from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import re


QUERY_MARKER = ", but "

ATTRIBUTE_LEXICON: "OrderedDict[str, OrderedDict[str, tuple[str, ...]]]" = OrderedDict(
    [
        (
            "color",
            OrderedDict(
                [
                    ("black", ("black",)),
                    ("white", ("white", "ivory")),
                    ("blue", ("blue", "navy", "teal")),
                    ("red", ("red", "burgundy", "maroon")),
                    ("green", ("green", "olive")),
                    ("yellow", ("yellow", "mustard")),
                    ("pink", ("pink", "fuchsia", "rose")),
                    ("purple", ("purple", "lavender")),
                    ("orange", ("orange", "coral")),
                    ("brown", ("brown", "tan", "camel", "khaki")),
                    ("gray", ("gray", "grey", "silver")),
                    ("beige", ("beige", "cream", "nude")),
                    ("gold", ("gold",)),
                ]
            ),
        ),
        (
            "pattern",
            OrderedDict(
                [
                    ("solid", ("solid", "plain")),
                    ("striped", ("striped", "stripes")),
                    ("floral", ("floral", "flowers", "flower")),
                    ("plaid", ("plaid", "tartan")),
                    ("checked", ("checked", "checkered", "checks")),
                    ("polka dot", ("polka dot", "polka dots", "dots", "dotted")),
                    ("animal print", ("animal print", "leopard", "cheetah", "snake print")),
                    ("graphic", ("graphic", "printed", "print")),
                    ("lace pattern", ("lace",)),
                ]
            ),
        ),
        (
            "neckline",
            OrderedDict(
                [
                    ("v-neck", ("v neck", "v-neck")),
                    ("crew neck", ("crew neck", "crewneck")),
                    ("scoop neck", ("scoop neck",)),
                    ("square neck", ("square neck",)),
                    ("turtleneck", ("turtleneck", "high neck")),
                    ("collared", ("collared", "collar")),
                    ("off-shoulder", ("off shoulder", "off-shoulder")),
                    ("strapless", ("strapless",)),
                    ("halter", ("halter",)),
                ]
            ),
        ),
        (
            "sleeve",
            OrderedDict(
                [
                    ("sleeveless", ("sleeveless", "without sleeves", "no sleeves")),
                    ("short sleeve", ("short sleeve", "short sleeves")),
                    ("long sleeve", ("long sleeve", "long sleeves")),
                    ("three-quarter sleeve", ("three quarter sleeve", "three quarter sleeves", "quarter sleeve", "quarter sleeves")),
                    ("cap sleeve", ("cap sleeve", "cap sleeves")),
                    ("spaghetti strap", ("spaghetti strap", "spaghetti straps")),
                ]
            ),
        ),
        (
            "length",
            OrderedDict(
                [
                    ("cropped", ("cropped", "crop")),
                    ("short", ("short", "shorter")),
                    ("knee-length", ("knee length", "knee-length")),
                    ("midi", ("midi",)),
                    ("maxi", ("maxi",)),
                    ("long", ("long", "longer", "floor length", "floor-length")),
                ]
            ),
        ),
        (
            "fit",
            OrderedDict(
                [
                    ("fitted", ("fitted", "tight", "slim")),
                    ("loose", ("loose", "relaxed", "baggy")),
                    ("oversized", ("oversized", "oversize")),
                ]
            ),
        ),
        (
            "silhouette",
            OrderedDict(
                [
                    ("a-line", ("a line", "a-line")),
                    ("bodycon", ("bodycon",)),
                    ("shift", ("shift",)),
                    ("flared", ("flared", "flare")),
                    ("straight", ("straight",)),
                    ("structured", ("structured", "tailored")),
                ]
            ),
        ),
        (
            "material",
            OrderedDict(
                [
                    ("denim", ("denim", "jean")),
                    ("knit", ("knit", "knitted")),
                    ("leather", ("leather", "faux leather")),
                    ("lace", ("lace", "lacy")),
                    ("cotton", ("cotton",)),
                    ("silk", ("silk", "silky", "satin")),
                    ("chiffon", ("chiffon",)),
                    ("wool", ("wool", "woolen")),
                    ("linen", ("linen",)),
                ]
            ),
        ),
        (
            "decoration",
            OrderedDict(
                [
                    ("ruffle", ("ruffle", "ruffles", "ruffled")),
                    ("bow", ("bow", "bows")),
                    ("button", ("button", "buttons", "buttoned")),
                    ("zipper", ("zipper", "zip")),
                    ("embroidery", ("embroidered", "embroidery")),
                    ("sequin", ("sequins", "sequin")),
                    ("pocket", ("pocket", "pockets")),
                    ("belt", ("belt", "belted")),
                    ("pleated", ("pleated", "pleats")),
                ]
            ),
        ),
    ]
)


TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9\s]+")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class StructuredQueryRecord:
    raw_query: str
    reference_text: str
    modification_text: str
    reference_attributes: tuple[tuple[str, tuple[str, ...]], ...]
    modification_attributes: tuple[tuple[str, tuple[str, ...]], ...]
    retain_attributes: tuple[tuple[str, tuple[str, ...]], ...]
    modify_attributes: tuple[tuple[str, tuple[str, ...]], ...]
    structured_text: str
    retain_text: str
    modify_text: str
    reference_attr_text: str
    modification_attr_text: str

    def to_json_dict(self) -> dict:
        return {
            "raw_query": self.raw_query,
            "reference_text": self.reference_text,
            "modification_text": self.modification_text,
            "reference_attributes": {key: list(values) for key, values in self.reference_attributes},
            "modification_attributes": {key: list(values) for key, values in self.modification_attributes},
            "retain_attributes": {key: list(values) for key, values in self.retain_attributes},
            "modify_attributes": {key: list(values) for key, values in self.modify_attributes},
            "structured_text": self.structured_text,
            "retain_text": self.retain_text,
            "modify_text": self.modify_text,
            "reference_attr_text": self.reference_attr_text,
            "modification_attr_text": self.modification_attr_text,
        }


def split_text_query(query: str) -> tuple[str, str]:
    if QUERY_MARKER in query:
        reference_caption, modification = query.split(QUERY_MARKER, 1)
        return reference_caption.strip(), modification.strip()
    stripped = query.strip()
    return stripped, stripped


def normalize_text(text: str) -> str:
    normalized = text.lower().replace("-", " ").replace("/", " ")
    normalized = TOKEN_CLEAN_RE.sub(" ", normalized)
    normalized = SPACE_RE.sub(" ", normalized).strip()
    return normalized


def _match_variants(normalized_text: str, variants: tuple[str, ...]) -> bool:
    haystack = f" {normalized_text} "
    for variant in variants:
        variant_norm = normalize_text(variant)
        if not variant_norm:
            continue
        if f" {variant_norm} " in haystack:
            return True
    return False


def _extract_attributes(text: str) -> OrderedDict[str, tuple[str, ...]]:
    normalized = normalize_text(text)
    results: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for attribute_name, canonical_map in ATTRIBUTE_LEXICON.items():
        matches: list[str] = []
        for canonical, variants in canonical_map.items():
            if _match_variants(normalized, variants):
                matches.append(canonical)
        if matches:
            results[attribute_name] = tuple(dict.fromkeys(matches))
    return results


def extract_attributes(text: str) -> dict[str, list[str]]:
    return {key: list(values) for key, values in _extract_attributes(text).items()}


def _freeze_slots(slots: OrderedDict[str, tuple[str, ...]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((key, tuple(values)) for key, values in slots.items())


def _summarize_slots(slots: tuple[tuple[str, tuple[str, ...]], ...], empty_text: str) -> str:
    if not slots:
        return empty_text
    parts = [f"{key}={', '.join(values)}" for key, values in slots]
    return "; ".join(parts)


@lru_cache(maxsize=50000)
def decompose_query(query: str) -> StructuredQueryRecord:
    reference_text, modification_text = split_text_query(query)
    reference_slots = _extract_attributes(reference_text)
    modification_slots = _extract_attributes(modification_text)

    retain_slots: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    modify_slots: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    for attribute_name in ATTRIBUTE_LEXICON.keys():
        ref_values = reference_slots.get(attribute_name)
        mod_values = modification_slots.get(attribute_name)
        if mod_values:
            modify_slots[attribute_name] = mod_values
        elif ref_values:
            retain_slots[attribute_name] = ref_values

    reference_frozen = _freeze_slots(reference_slots)
    modification_frozen = _freeze_slots(modification_slots)
    retain_frozen = _freeze_slots(retain_slots)
    modify_frozen = _freeze_slots(modify_slots)

    reference_attr_text = "reference attributes: " + _summarize_slots(
        reference_frozen,
        "reference attributes unknown",
    )
    modification_attr_text = "requested change attributes: " + _summarize_slots(
        modification_frozen,
        "requested change attributes not explicitly detected",
    )
    retain_text = "keep these reference attributes: " + _summarize_slots(
        retain_frozen,
        "keep unspecified core reference attributes",
    )
    modify_text = "modify these attributes: " + _summarize_slots(
        modify_frozen,
        "modify according to raw request",
    )
    structured_text = (
        f"reference description: {reference_text}. "
        f"{reference_attr_text}. "
        f"{retain_text}. "
        f"raw modification request: {modification_text}. "
        f"{modification_attr_text}. "
        f"{modify_text}."
    )

    return StructuredQueryRecord(
        raw_query=query,
        reference_text=reference_text,
        modification_text=modification_text,
        reference_attributes=reference_frozen,
        modification_attributes=modification_frozen,
        retain_attributes=retain_frozen,
        modify_attributes=modify_frozen,
        structured_text=structured_text,
        retain_text=retain_text,
        modify_text=modify_text,
        reference_attr_text=reference_attr_text,
        modification_attr_text=modification_attr_text,
    )
