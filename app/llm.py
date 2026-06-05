from __future__ import annotations

import logging
import re

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import Settings


LOGGER = logging.getLogger(__name__)
ANSWER_PATTERN = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


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
        context = "\n\n".join(
            f"[Chunk {index + 1}]\n{chunk}" for index, chunk in enumerate(context_chunks)
        )
        prompt = (
            "Bạn là trợ lý làm bài trắc nghiệm dựa trên tài liệu được cung cấp.\n"
            "Chỉ chọn đúng một đáp án A, B, C hoặc D.\n"
            "Không giải thích, không thêm từ nào khác ngoài một ký tự duy nhất.\n\n"
            f"Tài liệu tham chiếu:\n{context}\n\n"
            f"Câu hỏi:\n{question}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Chỉ trả về đúng một ký tự A, B, C hoặc D. "
                            "Không thêm dấu câu hay giải thích."
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
        return answer

    @staticmethod
    def _normalize_answer(raw_text: str) -> str | None:
        text = raw_text.strip().upper()
        if text in {"A", "B", "C", "D"}:
            return text

        match = ANSWER_PATTERN.search(text)
        if match:
            return match.group(1).upper()
        return None
