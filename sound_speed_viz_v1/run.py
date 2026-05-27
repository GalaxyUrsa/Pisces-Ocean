"""
Entry point — mirrors original app.py CLI interface.

Usage:
    python run.py --nc_dir ../20260521_112842 --date 20251220 --port 8000
"""

import argparse
import os
import sys

import uvicorn

# Make sure the package is importable when run from this directory
sys.path.insert(0, os.path.dirname(__file__))

from backend.main import app, init_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "20260521_112842"))
    parser.add_argument("--date",  type=str, default="20251220")
    parser.add_argument("--port",  type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pred_path = os.path.join(args.nc_dir, f"prediction_{args.date}.nc")
    if os.path.exists(pred_path):
        print(f"Loading data from {args.nc_dir} ...")
        init_data(args.nc_dir, args.date)
    else:
        print(f"No data file found at {pred_path}, starting without preloaded data.")

    print(f"\nOpen http://127.0.0.1:{args.port}/")
    if args.debug:
        uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port,
                    reload=True, log_level="info")
    else:
        uvicorn.run(app, host="127.0.0.1", port=args.port,
                    reload=False, log_level="info")


if __name__ == "__main__":
    main()
