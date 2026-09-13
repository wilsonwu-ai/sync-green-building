"""
    python -m sync                                  preview only, port 8080
    python -m sync --display gbsim://curious-cat    the Green Building simulator
    python -m sync --display gbsim+https://HOST/curious-cat
    python -m sync --display ws://HOST:9000/tetris-17x9
    python -m sync --display tcp://HOST:9000
    python -m sync --display py://gbsim:WebDisplay?args=curious-cat,http://HOST/api
    python -m sync --no-appraisal                   no model calls at all
"""

from __future__ import annotations

import argparse
import logging
import socket

from aiohttp import web

from .server import create_app

# --display is the only argument anyone has to think about, and the scheme it
# takes is the whole difference between a preview and a facade. A one-line
# help= string cannot carry that, so the schemes go in an epilog that --help
# prints verbatim. The gbsim ones came late and lived only in producer.py's
# docstring, which meant the delivery path we actually have was unreachable
# from the command line - an install-day trap, not a documentation nicety.
DISPLAY_SCHEMES = """\
display schemes (--display):
  gbsim://INSTANCE             the Green Building simulator; one POST a frame to
                               http://sundai.willsarg.com/api/i/INSTANCE/frame
  gbsim+https://HOST/INSTANCE  the same simulator, hosted somewhere else
  ws://HOST:PORT/tetris-17x9   remote protocol v1 over WebSocket
  tcp://HOST:PORT              remote protocol v1, newline-delimited
  py://MODULE:ATTR?args=a,b    a legacy Display subclass; args= are passed to
                               its constructor, which gbsim's WebDisplay needs:
                               py://gbsim:WebDisplay?args=curious-cat,http://HOST/api
  none                         preview only (default)

An http:// or https:// URL is taken as a gbsim endpoint too, with or without
the trailing /frame, because that is what copying it out of the simulator's
address bar gives you.

Measured against the live simulator: its viewer saturates near 15 frames a
second and delivers FEWER above that - 20 sent arrived as 15.2, 28 sent as
13.4. Sending harder makes the facade worse, not smoother.
"""


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
    p = argparse.ArgumentParser(
        prog="sync",
        description="SYNC - the building takes the crowd's pulse",
        epilog=DISPLAY_SCHEMES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--display",
        default="none",
        help="gbsim://instance | ws:// | tcp:// | py://module:attr?args=a,b | none (see below)",
    )
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
