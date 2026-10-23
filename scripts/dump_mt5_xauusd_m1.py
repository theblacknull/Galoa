#!/usr/bin/env python3
"""Dump IC Markets MT5 XAUUSD M1 bid closes → GALOAP1 .bin (+ optional CSV).

Windows (official MetaTrader5 package + terminal installed):

  pip install MetaTrader5 python-dotenv
  python scripts/dump_mt5_xauusd_m1.py --days 5 --native
  # or attach to an already-logged-in terminal:
  python scripts/dump_mt5_xauusd_m1.py --days 5 --native --attach

Linux (mt5linux Docker bridge on localhost:18812):

  python3 scripts/dump_mt5_xauusd_m1.py --days 5

Credentials: repo-root `.env` (gitignored) with MT5_LOGIN / MT5_PASSWORD /
MT5_SERVER / MT5_SYMBOL. Never commit `.env`. Never print the password.
With `--attach`, only MT5_SERVER / MT5_SYMBOL are required from env.
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
        os.environ.setdefault(k.strip(), v.strip())


def write_outputs(
    rows: list[tuple[int, float]],
    symbol: str,
    server: str,
    stem: str,
    *,
    write_csv: bool,
) -> None:
    if not rows:
        raise ValueError("no rows to write")
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / f"{stem}.bin"
    meta_path = out / f"{stem}.meta.txt"

    with bin_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(rows)))
        for t, _ in rows:
            f.write(struct.pack("<q", t * 1_000_000_000))
        for _, c in rows:
            f.write(struct.pack("<d", c))

    csv_note = "none"
    if write_csv:
        csv_path = out / f"{stem}.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("ts_ns,close\n")
            for t, c in rows:
                f.write(f"{t * 1_000_000_000},{c}\n")
        csv_note = f"data/{stem}.csv"

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
                f"debug_csv={csv_note}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {bin_path} bars={len(rows)} "
        f"range={datetime.fromtimestamp(first_t, timezone.utc)} -> "
        f"{datetime.fromtimestamp(last_t, timezone.utc)}"
        + (f" csv={csv_note}" if write_csv else "")
    )


def _filter_days(rows: list[tuple[int, float]], days: int) -> list[tuple[int, float]]:
    rows = sorted(rows, key=lambda x: x[0])
    end = rows[-1][0]
    start = end - days * 86400
    return [(t, c) for t, c in rows if t >= start] or rows


def dump_native(
    login: int | None,
    password: str | None,
    server: str,
    symbol: str,
    count: int,
    days: int,
    stem: str,
    *,
    attach: bool,
    write_csv: bool,
) -> None:
    import MetaTrader5 as mt5  # type: ignore

    if attach:
        ok = mt5.initialize()
    else:
        if login is None or password is None:
            print("login/password required unless --attach", file=sys.stderr)
            sys.exit(1)
        ok = mt5.initialize(login=login, password=password, server=server)
    if not ok:
        print("initialize failed:", mt5.last_error(), file=sys.stderr)
        sys.exit(2)
    try:
        if not mt5.symbol_select(symbol, True):
            print("symbol_select failed:", symbol, mt5.last_error(), file=sys.stderr)
            sys.exit(3)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
        if rates is None or len(rates) == 0:
            print("no rates:", mt5.last_error(), file=sys.stderr)
            sys.exit(4)
        rows = _filter_days([(int(r["time"]), float(r["close"])) for r in rates], days)
        write_outputs(rows, symbol, server, stem, write_csv=write_csv)
    finally:
        mt5.shutdown()


def dump_mt5linux(
    login: int,
    password: str,
    server: str,
    symbol: str,
    count: int,
    days: int,
    stem: str,
    port: int,
    *,
    write_csv: bool,
) -> None:
    import rpyc

    conn = rpyc.classic.connect("localhost", port)
    conn._config["sync_request_timeout"] = 180
    try:
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
            part = conn.eval(f"day[{i}:{i + 500}]")
            rows.extend((int(t), float(c)) for t, c in part)
        write_outputs(rows, symbol, server, stem, write_csv=write_csv)
    finally:
        try:
            conn.execute("mt5.shutdown()")
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=5, help="calendar days ending at last bar")
    p.add_argument("--count", type=int, default=8000, help="M1 bars to request from MT5")
    p.add_argument("--stem", default=None, help="output stem under data/ (default xauusd_m1_{days}d)")
    p.add_argument("--native", action="store_true", help="use Windows MetaTrader5 package")
    p.add_argument(
        "--attach",
        action="store_true",
        help="with --native: attach to already-logged-in terminal (no password)",
    )
    p.add_argument("--csv", action="store_true", help="also write debug CSV (off by default)")
    p.add_argument("--rpyc-port", type=int, default=18812, help="mt5linux RPyC port")
    p.add_argument("--env", type=Path, default=ROOT / ".env")
    args = p.parse_args()

    if args.attach and not args.native:
        print("--attach requires --native", file=sys.stderr)
        sys.exit(1)

    load_dotenv(args.env)
    server = os.environ.get("MT5_SERVER")
    symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")
    if not server:
        print(f"missing MT5_SERVER in {args.env}", file=sys.stderr)
        sys.exit(1)

    login: int | None = None
    password: str | None = None
    if not args.attach:
        try:
            login = int(os.environ["MT5_LOGIN"])
            password = os.environ["MT5_PASSWORD"]
        except KeyError as e:
            print(f"missing {e} in {args.env} (or use --native --attach)", file=sys.stderr)
            sys.exit(1)

    stem = args.stem or f"xauusd_m1_{args.days}d"
    who = f"attach server={server}" if args.attach else f"login={login} server={server}"
    print(f"dump {who} symbol={symbol} days={args.days} (password redacted)")

    if args.native:
        dump_native(
            login,
            password,
            server,
            symbol,
            args.count,
            args.days,
            stem,
            attach=args.attach,
            write_csv=args.csv,
        )
    else:
        assert login is not None and password is not None
        dump_mt5linux(
            login,
            password,
            server,
            symbol,
            args.count,
            args.days,
            stem,
            args.rpyc_port,
            write_csv=args.csv,
        )


if __name__ == "__main__":
    main()
