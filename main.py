"""
QSE v135 Bot - main.py
==========================
Alur:
  1. Ambil semua simbol perpetual USDT di Bybit
  2. Ambil candle 4H BTCUSDT (dipakai gerbang BTC utk semua simbol lain)
  3. Untuk tiap simbol: jalankan Fase 1-5 (indikator -> pola -> sinyal ->
     gerbang -> 76 pola -> backtest -> ranking)
  4. Filter hasil: hanya yang Grade A DAN status EKSEKUSI (bukan TUNGGU)
  5. Kirim ringkasan ke Telegram

Dijalankan otomatis oleh GitHub Actions tiap 4 jam (lihat
.github/workflows/run_bot.yml), atau bisa dites manual:
    python main.py
"""

import sys
import time
import datetime as dt
import traceback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

from bybit_fetch import get_all_perp_symbols, get_klines_4h, fetch_all_symbols
from qse_core import QSECore
from qse_patterns import add_patterns
from qse_signals import add_signals
from qse_gates import add_gates
from qse_pattern76 import build_76
from qse_backtest import QSEBacktest
from qse_ranking import score_and_select
from telegram_notify import send_telegram, build_report

MIN_BARS = 250          # candle minimum supaya indikator (EMA200 dkk) valid
MAX_SYMBOLS = 20      # set angka (mis. 50) utk uji cepat; None = semua pasar


def process_symbol(symbol: str, df, btc_df):
    core = QSECore(df, btc_df=btc_df, symbol=symbol)
    core.run_all()
    add_patterns(core)
    add_signals(core)
    add_gates(core)
    nm, cLa, cSa = build_76(core)
    bt = QSEBacktest(core, cLa, cSa, nm)
    res = bt.run()
    hasil = score_and_select(core, cLa, cSa, nm, bt, res["per_pattern"])
    return [h for h in hasil if h["grade"] == "A" and h["eksekusi"]]


def main():
    t0 = time.time()
    print("Mengambil daftar simbol perpetual Bybit...")
    symbols = get_all_perp_symbols()
    if MAX_SYMBOLS:
        symbols = symbols[:MAX_SYMBOLS]
    print(f"Total simbol: {len(symbols)}")

    print("Mengambil candle BTCUSDT (gerbang BTC)...")
    btc_df = get_klines_4h("BTCUSDT")
    if len(btc_df) < MIN_BARS:
        print("[FATAL] Data BTCUSDT tidak cukup, batalkan run.")
        sys.exit(1)

    all_signals = {}
    ok, fail, skip = 0, 0, 0
    for i, sym in enumerate(symbols, 1):
        try:
            df = get_klines_4h(sym)
            if len(df) < MIN_BARS:
                skip += 1
                continue
            sinyal = process_symbol(sym, df, btc_df)
            if sinyal:
                all_signals[sym] = sinyal
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERROR] {sym}: {e}")
            traceback.print_exc(limit=1)
        if i % 25 == 0:
            print(f"  progres {i}/{len(symbols)} | ok={ok} fail={fail} skip={skip} "
                  f"| waktu {time.time()-t0:.0f}s")

    run_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = build_report(all_signals, run_time)
    print("\n" + "=" * 40)
    print(report)
    print("=" * 40)
    print(f"Selesai. ok={ok} fail={fail} skip={skip} | total waktu {time.time()-t0:.0f}s")

    send_telegram(report)


if __name__ == "__main__":
    main()
