from __future__ import annotations

from dataclasses import dataclass

from query_decomposition import StructuredQueryRecord, decompose_query


@dataclass
class StructuredPromptBatch:
    raw_queries: list[str]
    reference_texts: list[str]
    modification_texts: list[str]
    structured_texts: list[str]
    retain_texts: list[str]
    modify_texts: list[str]
    reference_attr_texts: list[str]
    modification_attr_texts: list[str]
    records: list[StructuredQueryRecord]


def prepare_structured_batch(textual_queries: list[str]) -> StructuredPromptBatch:
    records = [decompose_query(query) for query in textual_queries]
    return StructuredPromptBatch(
        raw_queries=list(textual_queries),
        reference_texts=[record.reference_text for record in records],
        modification_texts=[record.modification_text for record in records],
        structured_texts=[record.structured_text for record in records],
        retain_texts=[record.retain_text for record in records],
        modify_texts=[record.modify_text for record in records],
        reference_attr_texts=[record.reference_attr_text for record in records],
        modification_attr_texts=[record.modification_attr_text for record in records],
        records=records,
    )
