from __future__ import annotations

from core.schema import (
    QuestionSkeletonLabel,
    SkeletonRewritePayload,
    StructuredSkeletonConstraint,
    StructuredSkeletonEntity,
    StructuredSkeletonRelation,
)
from modules.question_type_classifier import QuestionTypeClassifier
from modules.skeleton_extractor import SkeletonExtractionResult


def _build_skeleton(
    *,
    entities: list[StructuredSkeletonEntity] | None = None,
    relations: list[StructuredSkeletonRelation] | None = None,
    constraints: list[StructuredSkeletonConstraint] | None = None,
) -> SkeletonExtractionResult:
    entities = entities or []
    relations = relations or []
    constraints = constraints or []
    return SkeletonExtractionResult(
        entities=[item.name for item in entities],
        relations=[item.name for item in relations],
        constraints=[item.value for item in constraints],
        skeleton=QuestionSkeletonLabel(
            entities=[item.name for item in entities],
            relations=[item.name for item in relations],
            constraints=[item.value for item in constraints],
        ),
        mode="stub_predicted",
        structured_entities=entities,
        structured_relations=relations,
        structured_constraints=constraints,
        rewrite_payload=SkeletonRewritePayload(original_query="q", retrieval_query="q", compensation_query="q"),
        details={},
    )


def test_question_type_classifier_rules_cover_all_target_buckets() -> None:
    classifier = QuestionTypeClassifier()

    procedure = classifier.classify("如何配置 Alpha 服务？", _build_skeleton())
    cause = classifier.classify(
        "为什么 Alpha 延迟变得严重？",
        _build_skeleton(relations=[StructuredSkeletonRelation(name="影响原因", normalized_name="影响原因", relation_type="cause")]),
    )
    fact = classifier.classify("Alpha 当前版本是什么？", _build_skeleton())
    relation = classifier.classify(
        "Alpha 和 Beta 的区别是什么？",
        _build_skeleton(
            entities=[
                StructuredSkeletonEntity(name="Alpha", normalized_name="alpha", role="anchor"),
                StructuredSkeletonEntity(name="Beta", normalized_name="beta", role="context"),
            ],
            relations=[StructuredSkeletonRelation(name="依赖关系", normalized_name="依赖关系", relation_type="relation")],
        ),
    )
    fallback = classifier.classify("请介绍一下 Alpha", _build_skeleton())

    assert procedure.question_type == "procedure_method"
    assert cause.question_type == "cause_explanation"
    assert fact.question_type == "fact_attribute"
    assert relation.question_type == "relation_compare"
    assert fallback.question_type == "fallback_balanced"


def test_question_type_classifier_applies_skeleton_correction() -> None:
    classifier = QuestionTypeClassifier()
    skeleton = _build_skeleton(
        entities=[
            StructuredSkeletonEntity(name="Alpha", normalized_name="alpha", role="anchor"),
            StructuredSkeletonEntity(name="Beta", normalized_name="beta", role="context"),
        ],
        relations=[StructuredSkeletonRelation(name="依赖", normalized_name="依赖", relation_type="relation")],
    )

    result = classifier.classify("Alpha 和 Beta 有什么关系？", skeleton)

    assert result.question_type == "relation_compare"
    assert result.question_type_confidence in {"high", "medium"}


def test_question_type_classifier_low_confidence_falls_back() -> None:
    classifier = QuestionTypeClassifier()

    result = classifier.classify("介绍 Alpha", _build_skeleton())

    assert result.question_type == "fallback_balanced"
    assert result.question_type_confidence == "low"
