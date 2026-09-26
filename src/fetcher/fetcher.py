import asyncio
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from playwright.async_api import TimeoutError as PlaywrightTimeout

from browser.session import open_url_in_browser, wait_for_idle
from data_models.writer_data_models import ScreenshotResult

from utils.log_setup import get_logger

logger = get_logger("fetcher")


def compute_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:8]


_SECOND_LEVEL_DOMAINS = frozenset(
    {
        "co",
        "com",
        "org",
        "net",
        "gov",
        "edu",
        "ac",
        "or",
        "ne",
        "go",
        "gob",
        "nic",
        "mil",
    }
)


def extract_base_domain(url: str) -> str:
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return "unknown"

    if not hostname:
        return "unknown"

    parts = hostname.lower().split(".")

    if len(parts) <= 1:
        domain = parts[0] if parts else "unknown"
    elif len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _SECOND_LEVEL_DOMAINS:
        domain = parts[-3]
    else:
        domain = parts[-2]

    domain = re.sub(r"[^a-z0-9-]", "", domain)
    return domain or "unknown"


async def _screenshot_with_max_height(page, path: str, max_height: int = 5000) -> None:
    dimensions = await page.evaluate("""() => {
        return {
            width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
            height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
        }
    }""")

    clip_height = min(dimensions["height"], max_height)
    clip_width = dimensions["width"]

    await page.set_viewport_size({"width": clip_width, "height": clip_height})

    await page.wait_for_timeout(100)

    await page.screenshot(path=path, full_page=True)

    with Image.open(path) as img:
        width, height = img.size
        if height > max_height:
            cropped = img.crop((0, 0, width, max_height))
            cropped.save(path)


async def capture_screenshots_for_url(
    file_prefix: str,
    url: str,
    output_dir: Path,
    browser,
    page_timeout: int,
    post_load_delay: int,
    max_retries: int,
    max_screenshot_height: int = 5000,
) -> ScreenshotResult:
    url_hash = compute_url_hash(url)
    screenshots_saved: list[str] = []
    last_error: str | None = None

    for attempt in range(max_retries):
        try:
            context, page = await open_url_in_browser(url, browser, page_timeout)

            try:
                load_path = output_dir / f"{file_prefix}_{url_hash}_load.png"
                await _screenshot_with_max_height(
                    page, str(load_path), max_screenshot_height
                )
                screenshots_saved.append(str(load_path))

                await wait_for_idle(page)

                idle_path = output_dir / f"{file_prefix}_{url_hash}_idle.png"
                await _screenshot_with_max_height(
                    page, str(idle_path), max_screenshot_height
                )
                screenshots_saved.append(str(idle_path))

                await asyncio.sleep(post_load_delay / 1000.0)
                delay_path = output_dir / f"{file_prefix}_{url_hash}_delay.png"
                await _screenshot_with_max_height(
                    page, str(delay_path), max_screenshot_height
                )
                screenshots_saved.append(str(delay_path))

                return ScreenshotResult(
                    file_prefix=file_prefix,
                    url=url,
                    success=True,
                    screenshots_saved=screenshots_saved,
                )

            finally:
                await context.close()

        except PlaywrightTimeout as e:
            last_error = f"Timeout: {e}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < max_retries - 1:
            backoff_seconds = 2**attempt
            await asyncio.sleep(backoff_seconds)

    logger.error(
        f"Screenshot error after {max_retries} attempts on {url} with file prefix {file_prefix}"
    )
    return ScreenshotResult(
        file_prefix=file_prefix,
        url=url,
        success=False,
        screenshots_saved=screenshots_saved,
        error_message=last_error,
    )
