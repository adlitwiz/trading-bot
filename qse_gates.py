"""
QSE v135 - Fase 4a: Gerbang Entry (SL/TP, Konfluensi)
==========================================================
Port baris 623-641 Pine Script. Dijalankan setelah qse_core+patterns+signals.

Nilai default SEMUA diambil persis dari input() di Pine (baris 1-164),
bukan tebakan -- sudah saya cross-check ulang dengan baca ulang source.

APROKSIMASI (belum 100% pasti, ditandai jelas di kode):
  - wkBL/wkBS ("weekly buffer level"): saya definisikan sebagai swing low/high
    mingguan 4 minggu terakhir. Formula asli belum saya temukan definisinya
    di source (kemungkinan ada di bagian yang belum saya baca).
  - crsUp/crsDn (CRS = symbol pembanding, default BTCUSDT): saya samakan
    dengan btcUp/btcDn karena defaultnya sama-sama BTCUSDT.
  - ltUp/ltDn ("long term", dipakai helpL/helpS): saya definisikan sebagai
    1D dan 1W kompak searah (selaras dengan deskripsi input useLT).
"""

import numpy as np
import pandas as pd


def add_gates(core, atr_len=14, sl_buf=0.45, sl_min_a=1.0, sl_max_a=2.5,
              min_conf=4, need_mtf=True, chase_a=1.5, min_help=4):
    df, o = core.df, core.out
    h, l, c = df["high"], df["low"], df["close"]
    atrv = o["atr"]

    sw8L, sw8H = o["sw8L"], o["sw8H"]

    weekly = df.resample("1W").agg({"high": "max", "low": "min"}).dropna()
    wkBL = weekly["low"].rolling(4, min_periods=1).min().reindex(df.index, method="ffill")
    wkBS = weekly["high"].rolling(4, min_periods=1).max().reindex(df.index, method="ffill")
    o["wkBL"], o["wkBS"] = wkBL, wkBS

    slLv = np.maximum(
        np.minimum(np.maximum(np.minimum(sw8L - atrv * sl_buf, c - atrv * sl_min_a), c - atrv * sl_max_a), c - wkBL),
        c - atrv * sl_max_a)
    slSv = np.minimum(
        np.maximum(np.minimum(np.maximum(sw8H + atrv * sl_buf, c + atrv * sl_min_a), c + atrv * sl_max_a), c + wkBS),
        c + atrv * sl_max_a)
    o["slLv"], o["slSv"] = slLv, slSv

    btcOkL, btcOkS = o["btcOkL"], o["btcOkS"]
    o["okL"] = ((c - slLv) >= atrv * sl_min_a * 0.9) & btcOkL
    o["okS"] = ((slSv - c) >= atrv * sl_min_a * 0.9) & btcOkS

    aboveK, belowK = o["aboveK"], o["belowK"]
    tenkan, kijun = o["tenkan"], o["kijun"]
    macdL, macdSg = o["macdL"], o["macdSg"]
    rsiV = o["rsi"]
    diP, diM = o["diP"], o["diM"]
    obvV, obvE = o["obv"], o["obvE"]
    dowUp, dowDn = o["dowUp"], o["dowDn"]
    btcUp, btcDn = o["btcUp"], o["btcDn"]
    mDy, mWk = o["mDy"], o["mWk"]
    crsUp, crsDn = btcUp, btcDn          # aproksimasi (lihat catatan docstring)
    ltUp = (mDy == 1) & (mWk == 1)       # aproksimasi
    ltDn = (mDy == -1) & (mWk == -1)     # aproksimasi

    helpL = (aboveK.astype(int) + (tenkan > kijun).astype(int) + (macdL > macdSg).astype(int)
             + (rsiV > 50).astype(int) + (diP > diM).astype(int) + (obvV > obvE).astype(int)
             + dowUp.astype(int) + crsUp.astype(int) + ltUp.astype(int))
    helpS = (belowK.astype(int) + (tenkan < kijun).astype(int) + (macdL < macdSg).astype(int)
             + (rsiV < 50).astype(int) + (diM > diP).astype(int) + (obvV < obvE).astype(int)
             + dowDn.astype(int) + crsDn.astype(int) + ltDn.astype(int))
    o["helpL"], o["helpS"] = helpL, helpS

    tUp, tDn = o["tUp"], o["tDn"]
    cdlL, cdlS = o["cdlL"], o["cdlS"]
    rvol = o["rvol"]
    cdvB, cdvS = o["cdvB"], o["cdvS"]
    d1 = o["d1"]
    htfBull, htfBear = o["htfBull"], o["htfBear"]

    confL = (htfBull.astype(int) + tUp.astype(int) + cdlL.astype(int) + (rvol >= 1.1).astype(int)
             + cdvB.astype(int) + (d1 > 0).astype(int))
    confS = (htfBear.astype(int) + tDn.astype(int) + cdlS.astype(int) + (rvol >= 1.1).astype(int)
             + cdvS.astype(int) + (d1 < 0).astype(int))
    o["confL"], o["confS"] = confL, confS

    ema20 = o["ema20"]
    ext20 = (c - ema20).abs() / atrv
    o["ext20"] = ext20

    g1L = (confL >= min_conf) & (ext20 <= chase_a) & (helpL >= min_help)
    g1S = (confS >= min_conf) & (ext20 <= chase_a) & (helpS >= min_help)
    o["g1L"], o["g1S"] = g1L, g1S
    o["g0L"] = g1L & ((not need_mtf) | htfBull)
    o["g0S"] = g1S & ((not need_mtf) | htfBear)
    o["vqL"] = (confL >= 5) & htfBull & (rvol >= 1.2)
    o["vqS"] = (confS >= 5) & htfBear & (rvol >= 1.2)

    # gerbang kalkulus f_kOK (kMode default "Skor (lunak)", kMinSc=2)
    kScL, kScS = o["kScL"], o["kScS"]
    o["kOKL"] = kScL >= 2
    o["kOKS"] = kScS >= 2
    return core
