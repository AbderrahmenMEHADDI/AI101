"""
Prompt Builder
==============
Constructs the final prompt passed to the LLM.

Responsibilities
----------------
- Combine system instructions, retrieved context, and the user question.
- Keep the structure consistent so the LLM knows exactly what to do.
- NOT responsible for token counting or generation.

Prompt structure
----------------

SYSTEM
──────
<grounding instructions>

CONTEXT
───────
[Source 1]
...

[Source 2]
...

QUESTION
────────
<user question>

ANSWER
──────
(LLM fills this in)
"""

import tiktoken
from dataclasses import dataclass
from typing import Optional

_tokenizer = tiktoken.get_encoding("cl100k_base")

_SYSTEM_INSTRUCTIONS = """\
You are a precise research assistant answering questions about documents.

Rules you MUST follow:
1. Answer using ONLY information from the provided sources.
2. Cite sources inline using their number: [Source 1], [Source 2], etc.
3. If the answer spans multiple sources, cite all relevant ones.
4. Use table data accurately — do not paraphrase numbers incorrectly.
5. When a figure description is a source, mention that it comes from a figure.
6. If the provided sources do not contain enough information to answer the question,
   say exactly: "The provided documents do not contain enough information to answer this question."
7. Do not fabricate facts, numbers, or citations.
8. Be concise but thorough. Use markdown formatting (headers, bullet points, bold) where it aids clarity.
"""


@dataclass
class BuiltPrompt:
    """The assembled prompt ready to send to the LLM."""
    system: str
    context: str
    question: str
    full_prompt: str   # Combined text for single-string LLM APIs
    total_tokens: int = 0


class PromptBuilder:
    """
    Builds a structured RAG prompt from context and a user question.

    Parameters
    ----------
    system_instructions : str, optional
        Override the default grounding instructions.

    Examples
    --------
    >>> builder = PromptBuilder()
    >>> prompt = builder.build(context_text, "What is Model A's accuracy?")
    >>> print(prompt.full_prompt)
    """

    def __init__(self, system_instructions: Optional[str] = None):
        self.system = system_instructions or _SYSTEM_INSTRUCTIONS

    def build(self, context_text: str, question: str) -> BuiltPrompt:
        """
        Assemble the full prompt.

        Parameters
        ----------
        context_text : str
            Pre-formatted context block from ``ContextBuilder.build()``.
        question : str
            The user's natural-language question.

        Returns
        -------
        BuiltPrompt
        """
        context_text = context_text.strip() or "(No relevant sources were retrieved.)"
        question = question.strip()

        full_prompt = (
            f"SYSTEM:\n{self.system.strip()}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:"
        )

        tokens = len(_tokenizer.encode(full_prompt, disallowed_special=()))

        return BuiltPrompt(
            system=self.system,
            context=context_text,
            question=question,
            full_prompt=full_prompt,
            total_tokens=tokens,
        )
