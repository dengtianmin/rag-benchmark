from __future__ import annotations

from dataclasses import dataclass

from core.schema import RetrievedDocument


@dataclass(slots=True)
class NumberedEvidence:
    index: int
    text: str
    document: RetrievedDocument


class RAGPromptBuilder:
    """Builds Chinese prompts with numbered evidence snippets."""

    def build_numbered_evidence(self, retrieved_documents: list[RetrievedDocument]) -> list[NumberedEvidence]:
        numbered: list[NumberedEvidence] = []
        for index, document in enumerate(retrieved_documents, start=1):
            title = str(document.metadata.get("doc_title", "")).strip()
            path = " / ".join(str(item).strip() for item in document.metadata.get("section_path", []) if str(item).strip())
            header_parts = [part for part in [title, path] if part]
            header = f"（{' | '.join(header_parts)}）" if header_parts else ""
            numbered.append(NumberedEvidence(index=index, text=f"[{index}] {header}\n{document.content}".strip(), document=document))
        return numbered

    def build_context(self, retrieved_documents: list[RetrievedDocument]) -> str:
        return "\n\n".join(item.text for item in self.build_numbered_evidence(retrieved_documents))

    def build_prompt(self, *, question: str, retrieved_documents: list[RetrievedDocument]) -> str:
        context = self.build_context(retrieved_documents)
        return (
            "你是一个面向企业产品文档问答的模型。\n"
            "请仅依据给定的编号证据回答问题，不要使用外部知识，不要猜测，不要补充未在证据中明确出现的信息。\n\n"
            "你的输出必须满足以下要求：\n"
            "1. 只能输出合法 JSON。\n"
            "2. 不要输出 markdown，不要输出代码块，不要输出解释、分析过程或额外文字。\n"
            "3. JSON 必须严格满足以下格式：\n"
            '输出格式：{"answer":"...","supporting_evidence":[1,2]}\n\n'
            "其中：\n"
            '- "answer"：最终答案，必须尽可能短，适合严格评测。\n'
            '- "supporting_evidence"：你实际使用到的证据编号列表，只能填写下方证据中出现过的编号。\n'
            "如果证据不足以回答问题，请输出：\n"
            '{"answer":"无法根据已检索到的证据确定答案","supporting_evidence":[]}\n\n'
            "额外约束：\n"
            "- answer 中不要重复问题。\n"
            "- answer 中不要包含推理过程。\n"
            "- answer 尽量复用证据中的原始表述。\n"
            "- 保留关键实体、数字、单位、型号、版本号的原始写法。\n"
            "- 如果是列表题，用最简洁形式作答。\n"
            "- 如果是是非题，优先输出“是”或“否”；只有在确有必要时再补充极短说明。\n"
            "- supporting_evidence 只保留真正支撑答案的编号，不要滥选。\n\n"
            f"问题：\n{question}\n\n"
            f"证据：\n{context}\n"
        )
