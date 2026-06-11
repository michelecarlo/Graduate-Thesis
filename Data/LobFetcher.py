"""Collect high-frequency limit-order-book snapshots from Binance into Data/.

Subscribes to the Binance spot partial-book-depth WebSocket stream, which
pushes the top `LEVELS` bids and asks every 100 ms, and records `TARGET`
snapshots to a flat CSV (one row per snapshot). Snapshots are timestamped on
receipt (UTC). Row count is bounded, so the file stays manageable.

Schema: recv_ts_ms, timestamp, last_update_id,
        bid_px_1..L, bid_sz_1..L, ask_px_1..L, ask_sz_1..L

Usage:
    pip install websocket-client pandas
    python lob_fetcher.py [SYMBOL] [TARGET]   # e.g. python lob_fetcher.py BTCUSDT 3000
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from websocket import create_connection

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 3000  # snapshots to collect
LEVELS = 20            # partial-book depth: 5, 10 or 20
INTERVAL_MS = 100      # stream cadence: 100 or 1000
OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / f"lob_{SYMBOL.lower()}_{LEVELS}lvl_{INTERVAL_MS}ms.csv"

STREAM = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth{LEVELS}@{INTERVAL_MS}ms"


def flatten(msg: dict, recv: float) -> dict:
    row = {
        "recv_ts_ms": int(recv * 1000),
        "timestamp": datetime.fromtimestamp(recv, timezone.utc).isoformat(),
        "last_update_id": msg["lastUpdateId"],
    }
    for i, (px, sz) in enumerate(msg["bids"][:LEVELS], 1):
        row[f"bid_px_{i}"], row[f"bid_sz_{i}"] = float(px), float(sz)
    for i, (px, sz) in enumerate(msg["asks"][:LEVELS], 1):
        row[f"ask_px_{i}"], row[f"ask_sz_{i}"] = float(px), float(sz)
    return row


def main() -> None:
    print(f"Connecting to {STREAM}")
    ws = create_connection(STREAM, timeout=30)
    rows: list[dict] = []
    try:
        while len(rows) < TARGET:
            msg = json.loads(ws.recv())
            rows.append(flatten(msg, time.time()))
            if len(rows) % 250 == 0:
                print(f"  {len(rows)}/{TARGET} snapshots")
    except KeyboardInterrupt:
        print("Interrupted; saving what was collected.")
    finally:
        ws.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    span = (df["recv_ts_ms"].iloc[-1] - df["recv_ts_ms"].iloc[0]) / 1000
    print(
        f"Saved {len(df)} snapshots x {df.shape[1]} cols to {OUT_FILE} "
        f"(~{span:.0f}s span, {LEVELS} levels/side @ {INTERVAL_MS}ms)"
    )


if __name__ == "__main__":
    main()