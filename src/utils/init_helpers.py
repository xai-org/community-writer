import os
from typing import List

from requests_oauthlib import OAuth1Session  # type: ignore
import dotenv
from data_models.environment_variables import KeySet, EnvironmentVariables


def load_environment_variables(x_key_sets: List[str]) -> EnvironmentVariables:
    dotenv.load_dotenv(override=True)
    missing_vars = []
    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        missing_vars.append("XAI_API_KEY")

    if not os.getenv("POST_HASH_SALT"):
        missing_vars.append("POST_HASH_SALT")

    for path_var in (
        "CONFIG_DIR",
        "FEED_LOGS_DIR",
        "SUBMISSION_LOGS_DIR",
        "SCREENSHOT_DIR",
    ):
        if not os.getenv(path_var):
            missing_vars.append(path_var)
    x_api_keys = {}
    for suffix in x_key_sets:
        upper_suffix = suffix.upper().replace("-", "_")
        api_key = os.getenv(f"X_API_KEY_{upper_suffix}")
        api_secret_key = os.getenv(f"X_API_KEY_SECRET_{upper_suffix}")
        access_token = os.getenv(f"X_ACCESS_TOKEN_{upper_suffix}")
        access_token_secret = os.getenv(f"X_ACCESS_TOKEN_SECRET_{upper_suffix}")

        if not api_key:
            missing_vars.append(f"X_API_KEY_{upper_suffix}")
        if not api_secret_key:
            missing_vars.append(f"X_API_KEY_SECRET_{upper_suffix}")
        if not access_token:
            missing_vars.append(f"X_ACCESS_TOKEN_{upper_suffix}")
        if not access_token_secret:
            missing_vars.append(f"X_ACCESS_TOKEN_SECRET_{upper_suffix}")
        x_api_keys[suffix] = KeySet(
            x_api_key=api_key,
            x_api_secret_key=api_secret_key,
            x_access_token=access_token,
            x_access_token_secret=access_token_secret,
        )

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in your .env file or environment."
        )

    return EnvironmentVariables(
        xai_api_key=xai_api_key,
        x_api_keys=x_api_keys,
    )


def create_oauth_sessions(x_api_keys: dict[str, KeySet]) -> dict[str, OAuth1Session]:
    return {
        suffix: OAuth1Session(
            x_api_key.x_api_key,
            client_secret=x_api_key.x_api_secret_key,
            resource_owner_key=x_api_key.x_access_token,
            resource_owner_secret=x_api_key.x_access_token_secret,
        )
        for suffix, x_api_key in x_api_keys.items()
    }
