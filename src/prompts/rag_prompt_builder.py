from __future__ import annotations

from core.schema import RetrievedDocument


class RAGPromptBuilder:
    """Minimal prompt builder for mock and future real generators."""

    def build_context(self, retrieved_documents: list[RetrievedDocument]) -> str:
        blocks = []
        for document in retrieved_documents:
            title = document.metadata.get("doc_title", "")
            path = " / ".join(document.metadata.get("section_path", []))
            blocks.append(
                "\n".join(
                    [
                        f"[DOC] {document.source_id}",
                        f"[SECTION] {document.section_id}",
                        f"[TITLE] {title}",
                        f"[PATH] {path}",
                        f"[CONTENT] {document.content}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def build_prompt(self, *, question: str, retrieved_documents: list[RetrievedDocument]) -> str:
        context = self.build_context(retrieved_documents)
        return (
            "You are a QA model for enterprise product documents.\n"
            "Answer strictly based on the retrieved context.\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )
