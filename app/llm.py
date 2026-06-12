from __future__ import annotations

import logging
import re
import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import Settings
from app.text_utils import extract_options, strip_options


LOGGER = logging.getLogger(__name__)
ANSWER_PATTERN = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
MAX_CONTEXT_CHARS = 8000


class TeacherProxyTimeoutError(RuntimeError):
    """Teacher proxy timed out."""


class TeacherProxyRequestError(RuntimeError):
    """Teacher proxy request failed."""


class LlmService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.teacher_proxy_base_url,
            api_key=settings.student_id,
            timeout=settings.llm_timeout_seconds,
        )

    def answer_question(self, question: str, context_chunks: list[str]) -> str:
        started_at = time.perf_counter()
        context = "\n\n".join(
            f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(context_chunks)
        )
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0].strip()
        prompt = self._build_prompt(question, context)

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer Vietnamese legal multiple-choice questions using only the "
                            "provided legal text. Identify the best legal basis internally, then "
                            "return exactly one uppercase letter: A, B, C, or D."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
        except APITimeoutError as exc:
            LOGGER.exception("Teacher proxy request timed out")
            raise TeacherProxyTimeoutError("Teacher proxy request timed out") from exc
        except (APIConnectionError, APIStatusError) as exc:
            LOGGER.exception("Teacher proxy request failed")
            raise TeacherProxyRequestError("Teacher proxy request failed") from exc

        content = response.choices[0].message.content or ""
        answer = self._normalize_answer(content)
        if answer is None:
            raise RuntimeError(f"Could not parse answer from model output: {content!r}")
        LOGGER.info(
            "Teacher proxy returned answer=%s in %.3fs",
            answer,
            time.perf_counter() - started_at,
        )
        return answer

    @staticmethod
    def _build_prompt(question: str, context: str) -> str:
        options = extract_options(question)
        if options:
            formatted_options = "\n".join(
                f"{label}. {content}" for label, content in options.items()
            )
            return (
                "Use the legal text below to answer the multiple-choice question.\n"
                "Prefer the option that is most directly supported by the legal text.\n"
                "If two options are similar, choose the one with the strongest explicit basis.\n"
                "Return only one letter: A, B, C, or D.\n\n"
                f"Legal text:\n{context}\n\n"
                f"Question:\n{strip_options(question)}\n\n"
                f"Options:\n{formatted_options}"
            )

        return (
            "Use the legal text below to answer the Vietnamese multiple-choice question.\n"
            "Return only one letter: A, B, C, or D.\n\n"
            f"Legal text:\n{context}\n\n"
            f"Question:\n{question}"
        )

    @staticmethod
    def _normalize_answer(raw_text: str) -> str | None:
        text = raw_text.strip().upper()
        if text in {"A", "B", "C", "D"}:
            return text

        match = ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).upper()
        return None
