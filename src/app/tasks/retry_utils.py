from requests.exceptions import ConnectionError, Timeout
from huggingface_hub.utils import HfHubHTTPError


RETRYABLE_HTTP_STATUS_CODES = {
    408,  # request timeout
    429,  # rate limit
    500,
    502,
    503,
    504,
}


def is_retryable_error(exc: Exception) -> bool:
    """
    Return True only for transient errors that may succeed on retry.

    Examples:
    - network connection dropped
    - request timeout
    - Hugging Face temporary server/rate-limit errors

    Non-retryable examples:
    - bad input
    - missing columns
    - target repo already exists
    - invalid upload_mode
    - source repo update refused
    """

    if isinstance(exc, (ConnectionError, Timeout)):
        return True

    if isinstance(exc, HfHubHTTPError):
        status_code = exc.response.status_code if exc.response is not None else None
        return status_code in RETRYABLE_HTTP_STATUS_CODES

    return False