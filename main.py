"""Entry point shim — the implementation lives in `essaywriter.cli`."""

from essaywriter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
