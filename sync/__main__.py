"""
    python -m sync                                  preview only, port 8080
    python -m sync --display ws://HOST:9000/tetris-17x9
    python -m sync --display tcp://HOST:9000
    python -m sync --display py://utilities.dummy:DummyDisplay
    python -m sync --no-appraisal                   no model calls at all
"""

from __future__ import annotations

import argparse
import logging
import socket

from aiohttp import web

from .server import create_app


def _lan_ip() -> str:
    """The address to put on the QR code. Phones cannot reach localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    p = argparse.ArgumentParser(prog="sync", description="SYNC - the building takes the crowd's pulse")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--display", default="none", help="ws:// | tcp:// | py://module:attr | none")
    p.add_argument("--no-appraisal", action="store_true", help="skip all model calls")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ip = _lan_ip()
    print()
    print("  SYNC")
    print(f"  phones    http://{ip}:{args.port}/")
    print(f"  facade    http://{ip}:{args.port}/preview")
    print(f"  display   {args.display}")
    print()

    web.run_app(
        create_app(display=args.display, appraisal=not args.no_appraisal),
        host=args.host,
        port=args.port,
        print=lambda *a: None,
    )


if __name__ == "__main__":
    main()
