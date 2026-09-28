from playwright.async_api import TimeoutError as PlaywrightTimeout


def preflight() -> None:
    return


async def launch_browser(playwright):
    return await playwright.chromium.launch(headless=True)


async def open_url_in_browser(
    url: str,
    browser,
    page_timeout: int,
):
    context = await browser.new_context()
    page = await context.new_page()
    page.set_default_timeout(page_timeout)
    await page.goto(url, wait_until="domcontentloaded")
    return context, page


async def wait_for_idle(page) -> None:
    try:
        await page.wait_for_load_state("networkidle", timeout=10000)
    except PlaywrightTimeout:
        pass
