from pydantic import BaseModel


class KeySet(BaseModel):
    x_api_key: str
    x_api_secret_key: str
    x_access_token: str
    x_access_token_secret: str


class EnvironmentVariables(BaseModel):
    xai_api_key: str
    x_api_keys: dict[str, KeySet]
