"""
QSE v135 - Fase 6a: Fetch Data Bybit (Public API, tanpa API key)
====================================================================
Endpoint yang dipakai (semua publik, gratis, tanpa login):
  GET /v5/market/instruments-info?category=linear    -> daftar semua perpetual USDT
  GET /v5/market/kline?category=linear&symbol=X&interval=240&limit=1000  -> candle 4H
"""

import time
import requests
import pandas as pd

BASE_URL = "https://api.bybit.com"


def get_all_perp_symbols(quote: str = "USDT") -> list:
    """Ambil semua simbol perpetual linear (mis. BTCUSDT, ETHUSDT, ...)."""
    url = f"{BASE_URL}/v5/market/instruments-info"
    params = {"category": "linear", "limit": 1000}
    symbols = []
    cursor = ""
    while True:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        r = requests.get(url, params=p, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data.get('retMsg')}")
        result = data["result"]
        for item in result.get("list", []):
            if item.get("quoteCoin") == quote and item.get("status") == "Trading" \
               and item.get("contractType") == "LinearPerpetual":
                symbols.append(item["symbol"])
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
    return sorted(set(symbols))


def get_klines_4h(symbol: str, limit: int = 1000) -> pd.DataFrame:
    """Ambil candle 4H terbaru (maks 1000 candle per simbol, batas Bybit)."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": "240", "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error utk {symbol}: {data.get('retMsg')}")
    rows = data["result"]["list"]  # urutan Bybit: terbaru dulu
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype({"ts": "int64", "open": "float64", "high": "float64", "low": "float64",
                     "close": "float64", "volume": "float64"})
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").set_index("ts")
    return df[["open", "high", "low", "close", "volume"]]


def fetch_all_symbols(symbols: list, pause: float = 0.05, retries: int = 2) -> dict:
    """
    Ambil candle 4H utk banyak simbol sekaligus. `pause` (detik) diberi jeda
    antar-request supaya tidak kena rate limit Bybit (limitnya cukup longgar
    utk endpoint publik, tapi tetap sopan utk 400+ simbol).
    Return: dict {symbol: DataFrame}
    """
    out = {}
    for sym in symbols:
        for attempt in range(retries + 1):
            try:
                df = get_klines_4h(sym)
                if len(df) >= 60:  # minimal data biar indikator bisa dihitung wajar
                    out[sym] = df
                break
            except Exception as e:
                if attempt == retries:
                    print(f"[WARN] gagal ambil {sym}: {e}")
                else:
                    time.sleep(0.5)
        time.sleep(pause)
    return out


if __name__ == "__main__":
    syms = get_all_perp_symbols()
    print(f"Total perpetual USDT ditemukan: {len(syms)}")
    print(syms[:10], "...")
