import re
import httpx


URL_PATTERN = re.compile(r'https?://[^\s<>"\'\)]+[^\s<>"\'\)\.,;:!?]')


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub("", text).strip()


async def extract_and_validate_urls(
    text: str, citations: list[str], timeout: int = 10
) -> list[tuple[str, int | None]]:
    urls = URL_PATTERN.findall(text)

    if not urls:
        return []

    failed_urls = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            if url in citations:
                continue
            is_valid, status_code = await validate_url(
                url, client=client, timeout=timeout
            )
            if not is_valid:
                failed_urls.append((url, status_code))

    return failed_urls


async def validate_url(
    url: str, client: httpx.AsyncClient | None = None, timeout: int = 10
) -> tuple[bool, int | None]:
    try:
        if client is None:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=timeout, follow_redirects=True)
                status_code = response.status_code
                is_valid = 200 <= status_code < 400
                return is_valid, status_code
        else:
            response = await client.get(url, timeout=timeout, follow_redirects=True)
            status_code = response.status_code
            is_valid = 200 <= status_code < 400
            return is_valid, status_code
    except httpx.HTTPError:
        return False, None
