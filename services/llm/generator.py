import logging
import os

from google import genai
from google.genai.errors import ClientError

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

LLM_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"


def generate_answer(prompt: str, request_id: str = "-") -> str:
    """Generate a grounded answer from the Gemini LLM."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        text = (response.text or "").strip()

        logger.debug(
            "llm.response",
            extra={"preview": text[:120], "request_id": request_id},
        )

        return text.split("\n")[0]

    except ClientError as e:
        status = getattr(e, "status_code", None) or getattr(e, "code", None)
        if status in (429, 503):
            logger.warning(
                "llm.quota_exceeded",
                extra={"request_id": request_id, "status": status},
            )
        else:
            logger.error(
                "llm.client_error",
                extra={"request_id": request_id, "status": status},
            )
        return LLM_UNAVAILABLE

    except Exception:
        logger.exception(
            "llm.error",
            extra={"request_id": request_id, "model": MODEL_NAME},
        )
        return LLM_UNAVAILABLE