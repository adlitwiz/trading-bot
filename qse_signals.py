"""
QSE v135 - Fase 3a: Sinyal Cross & Divergensi Tambahan
==========================================================
Bahan-bahan tambahan yang belum ada di Fase 1 (qse_core.py) & Fase 2
(qse_patterns.py), dibutuhkan untuk merakit 76 kondisi pola di qse_pattern76.py.

Jalankan SETELAH qse_core.run_all() dan qse_patterns.add_patterns():
    from qse_signals import add_signals
    add_signals(core)   # menambah kolom ke core.out langsung
"""

import numpy as np
import pandas as pd


def add_signals(core, div_len: int = 20, hurst_thr: float = 0.55, z_thr: float = 2.0):
    df, o = core.df, core.out
    h, l, c, op, v = df["high"], df["low"], df["close"], df["open"], df["volume"]
    n = len(df)
    atrv = o["atr"]

    # ---------- Cumulative Delta Volume (cdvB/cdvS) ----------
    cdv = np.where(c >= op, v * 0.6, -v * 0.6)
    cdv_cum = pd.Series(cdv, index=df.index).cumsum()
    cdv_ema = cdv_cum.ewm(span=9, adjust=False).mean()
    o["cdvB"] = cdv_cum > cdv_ema
    o["cdvS"] = cdv_cum < cdv_ema

    # ---------- rvol threshold helper (rv10/rv11/rv12/rv13) ----------
    o["rv10"] = o["rvol"] >= 1.0
    o["rv11"] = o["rvol"] >= 1.1
    o["rv12"] = o["rvol"] >= 1.2
    o["rv13"] = o["rvol"] >= 1.3
    o["a05"] = atrv * 0.5
    o["a04"] = atrv * 0.4

    # ---------- Ichimoku cloud breakout & Tenkan-Kijun cross ----------
    kumoT, kumoB = o["kumoT"], o["kumoB"]
    tenkan, kijun = o["tenkan"], o["kijun"]
    o["kBU"] = (c > kumoT) & (c.shift(1) <= kumoT.shift(1))
    o["kBD"] = (c < kumoB) & (c.shift(1) >= kumoB.shift(1))
    o["tkU"] = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    o["tkD"] = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

    # ---------- MACD cross ----------
    macdL, macdSg = o["macdL"], o["macdSg"]
    o["macdCU"] = (macdL > macdSg) & (macdL.shift(1) <= macdSg.shift(1))
    o["macdCD"] = (macdL < macdSg) & (macdL.shift(1) >= macdSg.shift(1))

    # ---------- Divergensi MACD & RSI (vs low/high 20 bar) ----------
    pLo20, pHi20 = o["pLo20"], o["pHi20"]
    rsiV = o["rsi"]
    macdLo = macdL.rolling(div_len).min()
    macdHi = macdL.rolling(div_len).max()
    rsiLo = rsiV.rolling(div_len).min()
    rsiHi = rsiV.rolling(div_len).max()
    o["divBM"] = (l <= pLo20) & (macdL > macdLo.shift(1))
    o["divSM"] = (h >= pHi20) & (macdL < macdHi.shift(1))
    o["divBR"] = (l <= pLo20) & (rsiV > rsiLo.shift(1)) & (rsiV < 40)
    o["divSR"] = (h >= pHi20) & (rsiV < rsiHi.shift(1)) & (rsiV > 60)

    # ---------- RSI reclaim dari jenuh ----------
    o["rsiRL"] = (rsiV > 30) & (rsiV.shift(1) <= 30) & o["cdlL"]
    o["rsiRS"] = (rsiV < 70) & (rsiV.shift(1) >= 70) & o["cdlS"]

    # ---------- Island reversal (islB/islT) ----------
    gU, gD = o["gU"], o["gD"]
    o["islB"] = gD.shift(1).fillna(False) & gU
    o["islT"] = gU.shift(1).fillna(False) & gD

    # ---------- Bollinger squeeze & reversal ----------
    bbU, bbL, bbM = o["bbU"], o["bbL"], o["bbM"]
    bw = (bbU - bbL) / bbM
    bwPct = bw.rolling(100).apply(
        lambda w: (w < w.iloc[-1]).sum() / (len(w) - 1) * 100 if len(w) > 1 else np.nan, raw=False)
    isSqueeze = bwPct <= 20
    o["bwPct"] = bwPct
    o["sqBU"] = isSqueeze.shift(1).fillna(False) & (c > bbU) & (o["rvol"] >= 1.2)
    o["sqBD"] = isSqueeze.shift(1).fillna(False) & (c < bbL) & (o["rvol"] >= 1.2)
    o["bbRL"] = (l <= bbL) & (c > bbL) & o["cdlL"]
    o["bbRS"] = (h >= bbU) & (c < bbU) & o["cdlS"]

    # ---------- ADX/DMI cross ----------
    adxV, diP, diM = o["adx"], o["diP"], o["diM"]
    adxRise = (adxV > adxV.shift(1)) & (adxV > 20)
    o["dmiCU"] = (diP > diM) & (diP.shift(1) <= diM.shift(1)) & adxRise
    o["dmiCD"] = (diM > diP) & (diM.shift(1) <= diP.shift(1)) & adxRise

    # ---------- OBV divergensi & breakout ----------
    obvV, obvE = o["obv"], o["obvE"]
    obvLo = obvV.rolling(div_len).min()
    obvHi = obvV.rolling(div_len).max()
    o["obvDL"] = (l <= pLo20) & (obvV > obvLo.shift(1))
    o["obvDS"] = (h >= pHi20) & (obvV < obvHi.shift(1))
    obv20Hi_prev = obvV.rolling(20).max().shift(1)
    obv20Lo_prev = obvV.rolling(20).min().shift(1)
    o["obvBU"] = (obvV > obv20Hi_prev) & (c > c.shift(1))
    o["obvBD"] = (obvV < obv20Lo_prev) & (c < c.shift(1))

    # ---------- Infleksi kalkulus (d2) ----------
    d2 = o["d2"]
    o["inflL"] = (d2 > 0) & (d2.shift(1) <= 0)
    o["inflS"] = (d2 < 0) & (d2.shift(1) >= 0)

    # ---------- t-statistic korelasi (dipakai sinyal Hurst Z-score) ----------
    corrR, rSq = o["corrR"], o["rSq"]
    rSq_safe = rSq.clip(upper=0.999)
    o["tSt"] = corrR * np.sqrt(48) / np.sqrt((1 - rSq_safe).clip(lower=1e-6))
    o["hurstT"] = hurst_thr
    o["zThr"] = z_thr

    # ---------- Fraktal 3-bar sekuensial (frU1/frU2/frD1/frD2, breakout & pullback) ----------
    frU, frD = o["frU"].values, o["frD"].values
    frU1_a, frU2_a = np.full(n, np.nan), np.full(n, np.nan)
    frD1_a, frD2_a = np.full(n, np.nan), np.full(n, np.nan)
    frTU = np.zeros(n, bool)
    frTD = np.zeros(n, bool)
    fu1 = fu2 = fd1 = fd2 = np.nan
    for i in range(n):
        if not np.isnan(frU[i]):
            fu2, fu1 = fu1, frU[i]
        if not np.isnan(frD[i]):
            fd2, fd1 = fd1, frD[i]
        frU1_a[i], frU2_a[i] = fu1, fu2
        frD1_a[i], frD2_a[i] = fd1, fd2
        frTU[i] = (not np.isnan(fu1)) and (not np.isnan(fu2)) and fu1 > fu2
        frTD[i] = (not np.isnan(fd1)) and (not np.isnan(fd2)) and fd1 < fd2
    o["frU1"], o["frU2"], o["frD1"], o["frD2"] = frU1_a, frU2_a, frD1_a, frD2_a
    o["frTU"], o["frTD"] = frTU, frTD
    frU1_s, frD1_s = pd.Series(frU1_a, index=df.index), pd.Series(frD1_a, index=df.index)
    o["frBU"] = (c > frU1_s) & (c.shift(1) <= frU1_s.shift(1))
    o["frBD"] = (c < frD1_s) & (c.shift(1) >= frD1_s.shift(1))

    # ---------- Channel (dari garis tren sejajar +/- 2 ATR) & fan principle ----------
    tlHv, tlLv = o["tlHv"], o["tlLv"]
    chTop = tlHv + atrv * 2
    chBot = tlLv - atrv * 2
    o["chTop"], o["chBot"] = chTop, chBot
    chTolP = atrv * 0.35
    o["atChT"] = (~chTop.isna()) & (h >= chTop - chTolP) & (h <= chTop + chTolP * 2) & (c < chTop)
    o["atChB"] = (~chBot.isna()) & (l <= chBot + chTolP) & (l >= chBot - chTolP * 2) & (c > chBot)

    tlBU, tlBD = o["tlBU"], o["tlBD"]
    fanFU = np.zeros(n, bool)
    fanFD = np.zeros(n, bool)
    cntU = cntD = 0
    tlBU_v, tlBD_v = np.asarray(tlBU), np.asarray(tlBD)
    for i in range(n):
        if tlBU_v[i]:
            cntU += 1
            cntD = 0
        elif tlBD_v[i]:
            cntD += 1
            cntU = 0
        fanFU[i] = cntU >= 3   # tembus 3 garis fan (garis 1/2/3) beruntun -> sinyal fan principle
        fanFD[i] = cntD >= 3
    o["fanFU"], o["fanFD"] = fanFU, fanFD

    # ---------- Gap fill tracking (lastGU/lastGD, ala `var` Pine) ----------
    gU_v, gD_v = np.asarray(gU), np.asarray(gD)
    hv, lv = h.values, l.values
    lastGU_a = np.full(n, np.nan)
    lastGD_a = np.full(n, np.nan)
    lastGU = lastGD = np.nan
    for i in range(n):
        if gU_v[i] and i > 0:
            lastGU = lv[i]        # tepi bawah gap naik = low candle saat ini
        if gD_v[i] and i > 0:
            lastGD = hv[i]        # tepi atas gap turun = high candle saat ini
        lastGU_a[i], lastGD_a[i] = lastGU, lastGD
    o["lastGU"], o["lastGD"] = lastGU_a, lastGD_a

    return core
