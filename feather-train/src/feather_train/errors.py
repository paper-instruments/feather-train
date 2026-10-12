"""Failures that retrying a rollout cannot repair."""

from feather_harness.core.token_budget import PromptTokenBudgetError
from feather_harness.tools.web.exa import ExaConfigurationError
from verifiers.v1 import HarnessError

from feather_train.sandbox.retries import PERMANENT_E2B_ERRORS


class NonRetryableHarnessError(HarnessError):
    """Invalid configuration or a contract that retrying the rollout cannot repair."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code


NON_RETRYABLE_CAUSES = (
    NonRetryableHarnessError,
    PromptTokenBudgetError,
    ExaConfigurationError,
    *PERMANENT_E2B_ERRORS,
)
