import re
import unicodedata

import pandas as pd

from utils.log_setup import get_logger

logger = get_logger("results")


def _is_zero_width(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat in ("Mn", "Me", "Cf")


def _display_width(s: str) -> int:
    w = 0
    for ch in s:
        if _is_zero_width(ch):
            continue
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ("W", "F") else 1
    return w


def _truncate_to_width(s: str, width: int) -> str:
    w = 0
    for i, ch in enumerate(s):
        if _is_zero_width(ch):
            continue
        eaw = unicodedata.east_asian_width(ch)
        cw = 2 if eaw in ("W", "F") else 1
        if w + cw > width - 3:
            truncated = s[:i] + "..."

            pad = width - (w + 3)
            return truncated + " " * pad
        w += cw
    return s


def _rjust_to_width(s: str, width: int) -> str:
    pad = width - _display_width(s)
    return (" " * max(pad, 0)) + s


def _format_value(val, width: int) -> str:
    if pd.isna(val):
        s = "<NA>"
    elif isinstance(val, float):
        s = f"{val:.3f}"
    else:
        s = str(val)

    s = re.sub(
        r"[\U0001F300-\U0001F9FF"
        r"\U0001FA00-\U0001FAFF"
        r"\U00002600-\U000027BF"
        r"\U0000203C"
        r"\U00002049"
        r"\U0000FE00-\U0000FE0F"
        r"\U0000200D"
        r"\U000020E3"
        r"]",
        "",
        s,
    )
    s = s.replace("\n", "\\n").replace("\r", "\\r")
    if _display_width(s) > width:
        return _truncate_to_width(s, width)
    return _rjust_to_width(s, width)


OVER_LENGTH_MARKER = "over_280"


def print_writing_results(
    writing_results_df: pd.DataFrame, column_specs: list[tuple]
) -> None:
    columns = [spec[0] for spec in column_specs]
    formatted_df = writing_results_df[columns].copy()
    formatted_df["note_id"] = formatted_df["note_id"].astype(str)
    if (
        "error" in formatted_df.columns
        and "over_length_note" in writing_results_df.columns
    ):
        over_length = writing_results_df["over_length_note"].notna()

        formatted_df.loc[over_length, "error"] = [
            OVER_LENGTH_MARKER if pd.isna(err) else f"{OVER_LENGTH_MARKER}; {err}"
            for err in formatted_df.loc[over_length, "error"]
        ]
    for spec in column_specs:
        col, width = spec[0], spec[1]
        formatted_df[col] = formatted_df[col].apply(
            lambda val, w=width: _format_value(val, w)
        )

    idx_width = max(len(str(i)) for i in formatted_df.index)

    header_parts = [" " * idx_width]
    for spec in column_specs:
        col, width = spec[0], spec[1]
        display_name = spec[2] if len(spec) > 2 else col
        header_parts.append(display_name.rjust(width))
    lines = ["  ".join(header_parts)]

    for idx, row in formatted_df.iterrows():
        row_parts = [str(idx).rjust(idx_width)]
        for spec in column_specs:
            row_parts.append(row[spec[0]])
        lines.append("  ".join(row_parts))

    logger.info("Writing results:\n%s", "\n".join(lines))
