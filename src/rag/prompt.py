"""Grounded RAG prompt formatting engineered specifically for 1-2B micro LLMs."""

from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a precise, factual local knowledge assistant.
Your task is to answer the user's question using ONLY the retrieved document snippets provided inside <context>.

Rules for your answer:
1. Ground your statements strictly in the facts provided in <context>.
2. If the context does not contain the answer, explicitly state: "I cannot find the answer in the provided documents."
3. Do not extrapolate, assume, or hallucinate facts beyond what is stated.
4. When stating facts, cite the source using [Source N] corresponding to the document number.
5. Be concise, direct, and structured."""


def format_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved snippets and query into an explicit delimiter structure."""
    if not retrieved_chunks:
        context_str = "No relevant documents found in the database."
    else:
        doc_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            source_tag = f"[Source {idx}: {chunk.get('relative_path', 'unknown')}]"
            section_info = f" (Section: {chunk.get('section', 'General')})" if chunk.get('section') else ""
            block = (
                f'<document index="{idx}">\n'
                f"{source_tag}{section_info}\n"
                f"{chunk.get('text', '').strip()}\n"
                f"</document>"
            )
            doc_blocks.append(block)
        context_str = "\n\n".join(doc_blocks)

    user_message = f"""<context>
{context_str}
</context>

<question>
{query}
</question>

Answer:"""
    return user_message


def build_chat_messages(query: str, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build messages payload for OpenAI-compatible /v1/chat/completions."""
    user_content = format_rag_prompt(query, retrieved_chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
