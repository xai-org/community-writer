import logging
import sys


LOG_NAME_WIDTH = 12
LOG_FORMAT = (
    f"%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-{LOG_NAME_WIDTH}s    %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


_QUIET_LIBRARIES = (
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "asyncio",
    "filelock",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "PIL",
    "matplotlib",
    "playwright",
    "websockets",
    "xai_sdk",
)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
