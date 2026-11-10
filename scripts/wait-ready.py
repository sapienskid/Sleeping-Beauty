"""Poll an HTTP health or router /v1/models endpoint until ready."""

import argparse
import json
import sys
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", choices=("http", "router"), default="http")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=2.0) as resp:
                if resp.status == 200:
                    if args.mode == "http":
                        return 0
                    data = json.loads(resp.read().decode("utf-8"))
                    models = data.get("data", [])
                    if any(
                        isinstance(m, dict)
                        and m.get("status", {}).get("value") == "loaded"
                        for m in models
                    ):
                        return 0
        except Exception:
            pass
        time.sleep(0.35)
    return 1


if __name__ == "__main__":
    sys.exit(main())
