#!/usr/bin/env python3
"""Dump IC Markets MT5 demo XAUUSD M1 bid closes → GALOAP1 .bin + debug CSV.

Linux (this repo's cloud path): needs a running mt5linux Docker bridge
(`lprett/mt5linux`) with RPyC on localhost:18812, then:

  python3 scripts/dump_mt5_xauusd_m1.py --days 5

Windows (official MetaTrader5 package + terminal installed):

  pip install MetaTrader5 python-dotenv
  python scripts/dump_mt5_xauusd_m1.py --days 5 --native

Credentials: repo-root `.env` (gitignored) with MT5_LOGIN / MT5_PASSWORD /
MT5_SERVER / MT5_SYMBOL. Never commit `.env`. Never print the password.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGIC = b"GALOAP1\0"


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)


def write_outputs(rows: list[tuple[int, float]], symbol: str, server: str, stem: str) -> None:
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{stem}.csv"
    bin_path = out / f"{stem}.bin"
    meta_path = out / f"{stem}.meta.txt"

    with csv_path.open("w") as f:
        f.write("ts_ns,close\n")
        for t, c in rows:
            f.write(f"{t * 1_000_000_000},{c}\n")

    with bin_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(rows)))
        for t, _ in rows:
            f.write(struct.pack("<q", t * 1_000_000_000))
        for _, c in rows:
            f.write(struct.pack("<d", c))

    first_t, last_t = rows[0][0], rows[-1][0]
    meta_path.write_text(
        "\n".join(
            [
                f"symbol={symbol}",
                "timeframe=M1",
                "close_convention=mt5_bar_close_bid",
                f"bars={len(rows)}",
                f"ts_start_utc={datetime.fromtimestamp(first_t, timezone.utc).isoformat()}",
                f"ts_end_utc={datetime.fromtimestamp(last_t, timezone.utc).isoformat()}",
                f"server={server}",
                f"canonical=data/{stem}.bin",
                f"debug_csv=data/{stem}.csv",
                "",
            ]
        )
    )
    print(
        f"wrote {bin_path} + {csv_path} bars={len(rows)} "
        f"range={datetime.fromtimestamp(first_t, timezone.utc)} -> "
        f"{datetime.fromtimestamp(last_t, timezone.utc)}"
    )


def dump_native(login: int, password: str, server: str, symbol: str, count: int, days: int, stem: str) -> None:
    import MetaTrader5 as mt5  # type: ignore

    if not mt5.initialize(login=login, password=password, server=server):
        print("initialize failed:", mt5.last_error(), file=sys.stderr)
        sys.exit(2)
    if not mt5.symbol_select(symbol, True):
        print("symbol_select failed:", symbol, mt5.last_error(), file=sys.stderr)
        sys.exit(3)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
    if rates is None or len(rates) == 0:
        print("no rates:", mt5.last_error(), file=sys.stderr)
        sys.exit(4)
    rows = [(int(r["time"]), float(r["close"])) for r in rates]
    rows.sort(key=lambda x: x[0])
    end = rows[-1][0]
    start = end - days * 86400
    filtered = [(t, c) for t, c in rows if t >= start] or rows
    write_outputs(filtered, symbol, server, stem)
    mt5.shutdown()


def dump_mt5linux(login: int, password: str, server: str, symbol: str, count: int, days: int, stem: str, port: int) -> None:
    import rpyc

    conn = rpyc.classic.connect("localhost", port)
    conn._config["sync_request_timeout"] = 180
    conn.execute(r"import sys; sys.path.append('C:\\mt5libs')")
    conn.execute("import MetaTrader5 as mt5")
    conn.execute(
        f'ok=mt5.initialize(r"C:\\MT5\\terminal64.exe", login={login}, '
        f'server="{server}", password={password!r}, timeout=120000, portable=True)'
    )
    if not conn.eval("ok"):
        print("initialize failed:", conn.eval("mt5.last_error()"), file=sys.stderr)
        sys.exit(2)
    conn.execute(f'mt5.symbol_select("{symbol}", True)')
    conn.execute(f'rates=mt5.copy_rates_from_pos("{symbol}", mt5.TIMEFRAME_M1, 0, {count})')
    n = conn.eval("None if rates is None else len(rates)")
    if not n:
        print("no rates:", conn.eval("mt5.last_error()"), file=sys.stderr)
        sys.exit(4)
    conn.execute(
        f"""
rows=[(int(r["time"]), float(r["close"])) for r in rates]
rows.sort(key=lambda x: x[0])
end=rows[-1][0]
start=end-{days}*86400
day=[(t,c) for t,c in rows if t>=start] or rows
"""
    )
    n = conn.eval("len(day)")
    rows: list[tuple[int, float]] = []
    for i in range(0, n, 500):
        part = conn.eval(f"day[{i}:{i+500}]")
        rows.extend((int(t), float(c)) for t, c in part)
    write_outputs(rows, symbol, server, stem)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=5, help="calendar days ending at last bar")
    p.add_argument("--count", type=int, default=8000, help="M1 bars to request from MT5")
    p.add_argument("--stem", default=None, help="output stem under data/ (default xauusd_m1_{days}d)")
    p.add_argument("--native", action="store_true", help="use Windows MetaTrader5 package")
    p.add_argument("--rpyc-port", type=int, default=18812, help="mt5linux RPyC port")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    args = p.parse_args()

    load_dotenv(args.env)
    try:
        login = int(os.environ["MT5_LOGIN"])
        password = os.environ["MT5_PASSWORD"]
        server = os.environ["MT5_SERVER"]
        symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")
    except KeyError as e:
        print(f"missing {e} in {args.env} (copy from secrets; never commit)", file=sys.stderr)
        sys.exit(1)

    stem = args.stem or f"xauusd_m1_{args.days}d"
    print(f"dump login={login} server={server} symbol={symbol} days={args.days} (password redacted)")

    if args.native:
        dump_native(login, password, server, symbol, args.count, args.days, stem)
    else:
        dump_mt5linux(login, password, server, symbol, args.count, args.days, stem, args.rpyc_port)


if __name__ == "__main__":
    main()
