"""
QSE v135 - Fase 2: Fibonacci/SnR, SMC, Chart Pattern
========================================================
Port dari Pine Script baris ~475-620.

Ditambahkan ke QSECore.out (lihat qse_core.py) via method-method di sini:
  - fibonacci_snr(): Golden Pocket/Zone, Fib retracement, SnR pivot
  - smc(): pivot struktur (ph1-3/pl1-3), BOS/CHoCH, Order Block, FVG,
           liquidity sweep, false breakout, Dow/Elliott dasar
  - chart_patterns(): triangle, double/triple top-bottom, H&S, hound,
           horn, rectangle, ascending/descending triangle, wedge,
           continuation (flag)

Dipakai dengan cara (setelah QSECore.run_all() di qse_core.py):
    from qse_patterns import add_patterns
    core = QSECore(df, btc_df=btc_df, symbol="ETHUSDT")
    out = core.run_all()
    out = add_patterns(core)   # menambah kolom pola ke core.out, return core.out

BELUM di fase ini (menyusul):
  - Sinyal indikator "silang/cross" (MACD cross, Tenkan-Kijun cross, ADX
    cross 20, RSI reclaim 50, dsb), divergensi MACD/RSI/OBV, Bollinger
    squeeze/reversal, fan principle, gap fill tracking, angka bulat breakout.
    Ini "bahan" tambahan yang masih dibutuhkan sebelum 76 kondisi pola
    (cLa/cSa di pine baris 650-651) bisa dirakit lengkap.
  - Perakitan akhir 76 pola (nama, level, arah) + tabel nm/nrA/brkA/lvL/lvS.
"""

import numpy as np
import pandas as pd


def _pivot_mask(values: np.ndarray, left: int, right: int, is_high: bool) -> np.ndarray:
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        w = values[i - left:i + right + 1]
        center = values[i]
        if is_high:
            if center == w.max() and (w == center).sum() == 1:
                mask[i] = True
        else:
            if center == w.min() and (w == center).sum() == 1:
                mask[i] = True
    return mask


def fibonacci_snr(core, sw_len: int = 30, snr_per: int = 20):
    df, o = core.df, core.out
    h, l, c = df["high"], df["low"], df["close"]
    n = len(df)

    swH = h.rolling(sw_len).max()
    swL = l.rolling(sw_len).min()
    hh_idx = h.rolling(sw_len).apply(lambda w: np.argmax(w.values), raw=False)
    ll_idx = l.rolling(sw_len).apply(lambda w: np.argmin(w.values), raw=False)
    upLeg = hh_idx > ll_idx  # highestbars > lowestbars -> swing high lebih baru dari swing low
    rng = swH - swL

    def ret(p):
        return np.where(upLeg, swH - rng * p, swL + rng * p)

    def ext(m):
        return np.where(upLeg, swL + rng * m, swH - rng * m)

    o["upLeg"] = upLeg
    o["fib0"] = np.where(upLeg, swH, swL)
    o["r236"], o["r382"], o["r500"], o["r786"] = ret(0.236), ret(0.382), ret(0.5), ret(0.786)
    o["e127"], o["e161"] = ext(1.272), ext(1.618)

    r618, r65 = ret(0.618), ret(0.65)
    o["gpTop"] = np.maximum(r618, r65)
    o["gpBot"] = np.minimum(r618, r65)
    o["inGP"] = (c <= o["gpTop"]) & (c >= o["gpBot"])
    o["gzTop"] = np.maximum(o["r500"], o["r786"])
    o["gzBot"] = np.minimum(o["r500"], o["r786"])
    o["inGZ"] = (c <= o["gzTop"]) & (c >= o["gzBot"])
    o["ezTop"] = np.maximum(o["e127"], o["e161"])
    o["ezBot"] = np.minimum(o["e127"], o["e161"])
    o["inEZ"] = (c <= o["ezTop"]) & (c >= o["ezBot"])

    # SnR ala ta.valuewhen(pivot snrPer/snrPer)
    piv_h_mask = _pivot_mask(h.values, snr_per, snr_per, True)
    piv_l_mask = _pivot_mask(l.values, snr_per, snr_per, False)
    resV = np.full(n, np.nan)
    supV = np.full(n, np.nan)
    last_res = last_sup = np.nan
    hv, lv = h.values, l.values
    for j in range(n):
        i = j - snr_per
        if i >= 0 and piv_h_mask[i]:
            last_res = hv[i]
        if i >= 0 and piv_l_mask[i]:
            last_sup = lv[i]
        resV[j], supV[j] = last_res, last_sup
    o["resV"], o["supV"] = resV, supV
    snrW = o["atr"] * 0.35
    o["atRes"] = (h >= resV - snrW) & (l <= resV + snrW)
    o["atSup"] = (h >= supV - snrW) & (l <= supV + snrW)
    return core


def smc(core, msb_piv: int = 7):
    df, o = core.df, core.out
    h, l, c, op = df["high"].values, df["low"].values, df["close"].values, df["open"].values
    n = len(df)
    atrv = o["atr"].values

    piv_h = _pivot_mask(h, msb_piv, msb_piv, True)
    piv_l = _pivot_mask(l, msb_piv, msb_piv, False)

    chg = pd.Series(c).diff()
    denom = chg.rolling(50).std(ddof=0)
    momZ = np.where(denom.values == 0, 0.0, ((chg - chg.rolling(50).mean()) / denom).fillna(0).values)

    ph1 = ph2 = ph3 = np.nan
    pl1 = pl2 = pl3 = np.nan
    sBias = 0
    obBT = obBB = obST = obSB = np.nan
    fvUT = fvUB = fvDT = fvDB = np.nan

    ph1_a, ph2_a, ph3_a = (np.full(n, np.nan) for _ in range(3))
    pl1_a, pl2_a, pl3_a = (np.full(n, np.nan) for _ in range(3))
    bosL, bosS, choL, choS = (np.zeros(n, bool) for _ in range(4))
    obL, obS = np.zeros(n, bool), np.zeros(n, bool)
    fvFL, fvFS = np.zeros(n, bool), np.zeros(n, bool)
    neckU_a, neckD_a = np.full(n, np.nan), np.full(n, np.nan)
    obBT_a, obBB_a, obST_a, obSB_a = (np.full(n, np.nan) for _ in range(4))
    fvUT_a, fvUB_a, fvDT_a, fvDB_a = (np.full(n, np.nan) for _ in range(4))

    for i in range(n):
        if piv_h[i]:
            ph3, ph2 = ph2, ph1
            ph1 = h[i]
        if piv_l[i]:
            pl3, pl2 = pl2, pl1
            pl1 = l[i]
        ph1_a[i], ph2_a[i], ph3_a[i] = ph1, ph2, ph3
        pl1_a[i], pl2_a[i], pl3_a[i] = pl1, pl2, pl3

        brkUp = (not np.isnan(ph1)) and c[i] > ph1 and (i > 0 and c[i - 1] <= ph1)
        brkDn = (not np.isnan(pl1)) and c[i] < pl1 and (i > 0 and c[i - 1] >= pl1)
        bosL[i] = brkUp and momZ[i] > 0.5 and sBias >= 0
        bosS[i] = brkDn and momZ[i] < -0.5 and sBias <= 0
        choL[i] = brkUp and momZ[i] > 0.5 and sBias < 0
        choS[i] = brkDn and momZ[i] < -0.5 and sBias > 0
        if brkUp:
            sBias = 1
        if brkDn:
            sBias = -1

        if bosL[i] or choL[i]:
            idx = 1
            for k in range(1, 11):
                if i - k >= 0 and c[i - k] < op[i - k]:
                    idx = k
                    break
            if i - idx >= 0:
                obBT, obBB = h[i - idx], l[i - idx]
        if bosS[i] or choS[i]:
            idx = 1
            for k in range(1, 11):
                if i - k >= 0 and c[i - k] > op[i - k]:
                    idx = k
                    break
            if i - idx >= 0:
                obST, obSB = h[i - idx], l[i - idx]
        if not np.isnan(obBB) and c[i] < obBB - atrv[i] * 0.2:
            obBT = np.nan
        if not np.isnan(obST) and c[i] > obST + atrv[i] * 0.2:
            obST = np.nan
        obL[i] = (not np.isnan(obBT)) and l[i] <= obBT and c[i] > obBB and c[i] > op[i]
        obS[i] = (not np.isnan(obST)) and h[i] >= obSB and c[i] < obST and c[i] < op[i]
        obBT_a[i], obBB_a[i], obST_a[i], obSB_a[i] = obBT, obBB, obST, obSB

        if i >= 2 and l[i] > h[i - 2] and c[i - 1] > h[i - 2]:
            fvUT, fvUB = l[i], h[i - 2]
        if i >= 2 and h[i] < l[i - 2] and c[i - 1] < l[i - 2]:
            fvDT, fvDB = l[i - 2], h[i]
        fvFL[i] = (not np.isnan(fvUB)) and l[i] <= fvUT and l[i] >= fvUB and c[i] > op[i]
        fvFS[i] = (not np.isnan(fvDT)) and h[i] >= fvDB and h[i] <= fvDT and c[i] < op[i]
        fvUT_a[i], fvUB_a[i], fvDT_a[i], fvDB_a[i] = fvUT, fvUB, fvDT, fvDB

        neckD_a[i] = min(pl1, pl2) if not np.isnan(pl2) else np.nan
        neckU_a[i] = max(ph1, ph2) if not np.isnan(ph2) else np.nan

    o["ph1"], o["ph2"], o["ph3"] = ph1_a, ph2_a, ph3_a
    o["pl1"], o["pl2"], o["pl3"] = pl1_a, pl2_a, pl3_a
    o["bosL"], o["bosS"], o["choL"], o["choS"] = bosL, bosS, choL, choS
    o["obL"], o["obS"] = obL, obS
    o["fvFL"], o["fvFS"] = fvFL, fvFS
    o["neckU"], o["neckD"] = neckU_a, neckD_a
    # ekspos level mentah (dibutuhkan lvL/lvS di Fase 5 utk entry/SL/TP)
    o["obBT"], o["obBB"], o["obST"], o["obSB"] = obBT_a, obBB_a, obST_a, obSB_a
    o["fvUT"], o["fvUB"], o["fvDT"], o["fvDB"] = fvUT_a, fvUB_a, fvDT_a, fvDB_a

    hi10, lo10 = o["hi10"].values, o["lo10"].values
    closR, rvol = o["closR"].values, o["rvol"].values
    cdlL, cdlS = o["cdlL"].values, o["cdlS"].values
    o["swpL"] = (l < lo10) & (c > lo10) & (closR > 0.6)
    o["swpS"] = (h > hi10) & (c < hi10) & (closR < 0.4)
    o["fbL"] = (l < lo10) & (c > lo10) & (rvol >= 1.1) & cdlL
    o["fbS"] = (h > hi10) & (c < hi10) & (rvol >= 1.1) & cdlS

    dowUp = (~np.isnan(ph2_a)) & (~np.isnan(pl2_a)) & (ph1_a > ph2_a) & (pl1_a > pl2_a)
    dowDn = (~np.isnan(ph2_a)) & (~np.isnan(pl2_a)) & (ph1_a < ph2_a) & (pl1_a < pl2_a)
    o["dowUp"], o["dowDn"] = dowUp, dowDn
    o["elwU"] = dowUp & (c > ph1_a) & (rvol >= 1.2)
    o["elwD"] = dowDn & (c < pl1_a) & (rvol >= 1.2)
    return core


def chart_patterns(core, pat_tol: float = 0.5):
    df, o = core.df, core.out
    h, l, c, op = df["high"], df["low"], df["close"], df["open"]
    atrv, rvol = o["atr"], o["rvol"]
    ph1, ph2, ph3 = o["ph1"], o["ph2"], o["ph3"]
    pl1, pl2, pl3 = o["pl1"], o["pl2"], o["pl3"]
    neckU, neckD = o["neckU"], o["neckD"]
    c1, h1, l1, h2, l2 = c.shift(1), h.shift(1), l.shift(1), h.shift(2), l.shift(2)

    # ---- f_patA ----
    triC = (~ph2.isna()) & (~pl2.isna()) & (ph1 < ph2) & (pl1 > pl2)
    o["triC"] = triC
    o["triL"] = triC & (c > ph1) & (c1 <= ph1) & (rvol >= 1.2)
    o["triS"] = triC & (c < pl1) & (c1 >= pl1) & (rvol >= 1.2)
    o["dblB"] = (~pl2.isna()) & ((pl1 - pl2).abs() <= atrv * 0.4) & (c > ph1) & (c1 <= ph1)
    o["dblT"] = (~ph2.isna()) & ((ph1 - ph2).abs() <= atrv * 0.4) & (c < pl1) & (c1 >= pl1)
    ib = (h1 < h2) & (l1 > l2)
    o["ibL"] = ib & (c > h1) & (rvol >= 1.1)
    o["ibS"] = ib & (c < l1) & (rvol >= 1.1)
    o["hsL"] = (~pl3.isna()) & (~neckU.isna()) & (pl2 < pl1 - atrv * 0.3) & (pl2 < pl3 - atrv * 0.3) \
        & ((pl1 - pl3).abs() <= atrv * pat_tol) & (c > neckU) & (c1 <= neckU)
    o["hsS"] = (~ph3.isna()) & (~neckD.isna()) & (ph2 > ph1 + atrv * 0.3) & (ph2 > ph3 + atrv * 0.3) \
        & ((ph1 - ph3).abs() <= atrv * pat_tol) & (c < neckD) & (c1 >= neckD)
    o["hound"] = (~ph3.isna()) & (ph2 > ph1) & (ph2 > ph3) & ((ph1 - ph3).abs() <= atrv * pat_tol) \
        & (~neckD.isna()) & (c < neckD) & (c1 >= neckD) & (rvol < 0.9)

    # ---- f_patB ----
    o["trT"] = (~ph3.isna()) & ((ph1 - ph2).abs() <= atrv * pat_tol) & ((ph2 - ph3).abs() <= atrv * pat_tol) \
        & (~neckD.isna()) & (c < neckD) & (c1 >= neckD)
    o["trB"] = (~pl3.isna()) & ((pl1 - pl2).abs() <= atrv * pat_tol) & ((pl2 - pl3).abs() <= atrv * pat_tol) \
        & (~neckU.isna()) & (c > neckU) & (c1 <= neckU)
    o["hrT"] = ((h2 - h).abs() <= atrv * 0.3) & (h1 < pd.concat([h2, h], axis=1).min(axis=1) - atrv * 0.2) & (c < l1)
    o["hrB"] = ((l2 - l).abs() <= atrv * 0.3) & (l1 > pd.concat([l2, l], axis=1).max(axis=1) + atrv * 0.2) & (c > h1)
    rectC = (~ph2.isna()) & (~pl2.isna()) & ((ph1 - ph2).abs() <= atrv * pat_tol) & ((pl1 - pl2).abs() <= atrv * pat_tol)
    o["rectC"] = rectC
    ascT = (~ph2.isna()) & (~pl2.isna()) & ((ph1 - ph2).abs() <= atrv * pat_tol) & (pl1 > pl2 + atrv * 0.2)
    desT = (~ph2.isna()) & (~pl2.isna()) & ((pl1 - pl2).abs() <= atrv * pat_tol) & (ph1 < ph2 - atrv * 0.2)
    o["ascT"], o["desT"] = ascT, desT
    wdgR = (~ph2.isna()) & (~pl2.isna()) & (ph1 > ph2) & (pl1 > pl2) & ((ph1 - pl1) < (ph2 - pl2) * 0.8)
    wdgF = (~ph2.isna()) & (~pl2.isna()) & (ph1 < ph2) & (pl1 < pl2) & ((ph1 - pl1) < (ph2 - pl2) * 0.8)
    o["wdgR"], o["wdgF"] = wdgR, wdgF

    # ---- f_patC (kontinuasi/flag) ----
    polU = (c.shift(5) - c.shift(15)) / atrv > 2.5
    polD = (c.shift(15) - c.shift(5)) / atrv > 2.5
    rng5 = (h.rolling(5).max() - l.rolling(5).min()) / atrv
    o["polU"], o["polD"], o["rng5"] = polU, polD, rng5
    hi5_prev = h.rolling(5).max().shift(1)
    lo5_prev = l.rolling(5).min().shift(1)
    o["contU"] = (rectC & (c > ph1) & (c1 <= ph1) & (rvol >= 1.2)) \
        | (ascT & (c > ph1) & (c1 <= ph1) & (rvol >= 1.2)) \
        | (polU & (rng5 < 1.6) & (c > hi5_prev) & (rvol >= 1.1)) \
        | (polU & triC & (c > ph1) & (c1 <= ph1)) \
        | (wdgF & (c > ph1) & (c1 <= ph1))
    o["contD"] = (rectC & (c < pl1) & (c1 >= pl1) & (rvol >= 1.2)) \
        | (desT & (c < pl1) & (c1 >= pl1) & (rvol >= 1.2)) \
        | (polD & (rng5 < 1.6) & (c < lo5_prev) & (rvol >= 1.1)) \
        | (polD & triC & (c < pl1) & (c1 >= pl1)) \
        | (wdgR & (c < pl1) & (c1 >= pl1))

    pat_priority = [
        ("hsS", "Head and Shoulders"), ("hsL", "Inverted H&S"), ("hound", "Hound Baskervilles"),
        ("trT", "Triple Top"), ("trB", "Triple Bottom"), ("dblT", "Double Top"), ("dblB", "Double Bottom"),
        ("hrT", "Horn Top"), ("hrB", "Horn Bottom"), ("wdgR", "Rising Wedge"), ("wdgF", "Falling Wedge"),
        ("ascT", "Ascending Triangle"), ("desT", "Descending Triangle"), ("rectC", "Rectangle"),
        ("triC", "Symmetrical Triangle"),
    ]
    pat_name = pd.Series("belum ada pola", index=df.index)
    for col, name in reversed(pat_priority):
        pat_name = pat_name.mask(o[col], name)
    o["patTxt"] = pat_name
    return core


def add_patterns(core):
    fibonacci_snr(core)
    smc(core)
    chart_patterns(core)
    return core.out
