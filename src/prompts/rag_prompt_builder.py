from __future__ import annotations

from core.schema import RetrievedDocument


class RAGPromptBuilder:
    """Builds Chinese prompts with numbered evidence snippets."""

    def build_context(self, retrieved_documents: list[RetrievedDocument]) -> str:
        blocks = []
        for index, document in enumerate(retrieved_documents, start=1):
            title = str(document.metadata.get("doc_title", "")).strip()
            path = " / ".join(str(item).strip() for item in document.metadata.get("section_path", []) if str(item).strip())
            header_parts = [part for part in [title, path] if part]
            header = f"（{' | '.join(header_parts)}）" if header_parts else ""
            blocks.append(f"[{index}] {header}\n{document.content}".strip())
        return "\n\n".join(blocks)

    def build_prompt(self, *, question: str, retrieved_documents: list[RetrievedDocument]) -> str:
        context = self.build_context(retrieved_documents)
        return (
            "你是一个面向企业产品文档问答的模型。\n"
            "你只能根据给定的编号证据片段回答问题。\n\n"
            "只返回合法 JSON。\n"
            "不要输出 Markdown、代码块、解释、说明或任何额外文本。\n\n"
            "规则：\n"
            '1. "answer" 必须是尽量短、适合严格评测的最终答案。\n'
            '2. "supporting_evidence" 必须是证据编号数组，只能使用上下文中实际出现过的编号。\n'
            '3. 如果证据不足，"answer" 必须固定为 "无法根据已检索到的证据确定答案"。\n'
            '4. 如果证据不足，"supporting_evidence" 必须为 []。\n\n'
            '输出格式：{"answer":"...","supporting_evidence":[1,2]}\n\n'
            f"问题：\n{question}\n\n"
            f"编号证据：\n{context}\n"
        )
