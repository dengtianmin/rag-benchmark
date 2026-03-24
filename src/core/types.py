from __future__ import annotations

from enum import Enum
from typing import Literal, TypeAlias


class QuestionType(str, Enum):
    FACT = "fact"
    RELATION = "relation"
    MULTI_EVIDENCE = "multi_evidence"
    EXPLANATION = "explanation"


class SourceScope(str, Enum):
    SINGLE_SECTION = "single_section"
    SINGLE_DOC = "single_doc"
    CROSS_DOC = "cross_doc"


class RetrievalMode(str, Enum):
    DOCUMENT = "document"
    TRIPLE = "triple"
    HYBRID = "hybrid"


QuestionId: TypeAlias = str
DocumentId: TypeAlias = str
SectionId: TypeAlias = str
SplitName: TypeAlias = Literal["train", "dev", "test", "full"]

