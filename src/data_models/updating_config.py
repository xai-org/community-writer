import asyncio
import difflib
import os
from dataclasses import dataclass, field

from data_models.arena_config import ArenaConfig

from utils.log_setup import get_logger

logger = get_logger("cfg_state")


@dataclass
class UpdatingConfig:
    config: ArenaConfig
    config_dir: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update(self, new_config: ArenaConfig) -> None:
        async with self.lock:
            old_config = self.config
            self.config = new_config

        self._log_config_change(old_config, new_config)

    async def get(self) -> ArenaConfig:
        async with self.lock:
            return self.config

    def _log_config_change(
        self, old_config: ArenaConfig, new_config: ArenaConfig
    ) -> None:
        old_timestamp = old_config.config_timestamp
        new_timestamp = new_config.config_timestamp

        old_filepath = os.path.join(self.config_dir, f"{old_timestamp}.toml")
        new_filepath = os.path.join(self.config_dir, f"{new_timestamp}.toml")

        try:
            with open(old_filepath, "r") as f:
                old_content = f.read().splitlines(keepends=True)
        except FileNotFoundError:
            old_content = ["(old config file not found)\n"]

        try:
            with open(new_filepath, "r") as f:
                new_content = f.read().splitlines(keepends=True)
        except FileNotFoundError:
            new_content = ["(new config file not found)\n"]

        diff_lines = list(
            difflib.unified_diff(
                old_content,
                new_content,
                fromfile=f"{old_timestamp}.toml",
                tofile=f"{new_timestamp}.toml",
            )
        )
        diff_text = "".join(diff_lines) if diff_lines else "(no differences found)"

        logger.info(f"Config updated from {old_timestamp} to {new_timestamp}")
        logger.info(f"Diff:\n{diff_text}")
