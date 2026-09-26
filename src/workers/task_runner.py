import argparse
import asyncio
import os
import pickle
import sys


_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.append(_src_dir)

from note_writer.write_note import _draft_and_query_rejector  # noqa: E402

from utils.log_setup import configure_logging, get_logger

logger = get_logger("task_runner")


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run a single writing task in an isolated process"
    )
    parser.add_argument(
        "--input", required=True, help="Path to pickled input arguments"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write pickled result DataFrame"
    )
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        task_kwargs = pickle.load(f)

    try:
        result_df = asyncio.run(_draft_and_query_rejector(**task_kwargs))
        with open(args.output, "wb") as f:
            pickle.dump(result_df, f)
    except Exception:
        logger.exception("Writing task failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
