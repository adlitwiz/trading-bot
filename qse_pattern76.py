"""
QSE v135 - Fase 3b: Rakitan 76 Pola Lengkap
================================================
Merakit nm (76 nama pola), cLa (76 kondisi LONG), cSa (76 kondisi SHORT)
persis urutan index di Pine Script baris 649-657 (nm/cLa/cSa).

Dipakai setelah qse_core + qse_patterns + qse_signals dijalankan:
    from qse_core import QSECore
    from qse_patterns import add_patterns
    from qse_signals import add_signals
    from qse_pattern76 import build_76

    core = QSECore(df, btc_df=btc_df, symbol="ETHUSDT")
    core.run_all(); add_patterns(core); add_signals(core)
    nm, cLa, cSa = build_76(core)

    # cLa/cSa: DataFrame (index waktu candle, 76 kolom nama pola) isi True/False
    # nm: list nama pola urutan index 0-75, dipakai backtest (Fase 4) & ranking (Fase 5)

CATATAN VALIDASI: urutan & isi formula di bawah adalah hasil transkripsi
manual 1:1 dari baris nm/cLa/cSa Pine Script Anda. Karena jumlahnya 76 x 2
kondisi, kesalahan ketik satu tanda (< jadi <=, and jadi or) sangat mungkin
terjadi. SANGAT DISARANKAN: setelah file ini jalan, kita bandingkan pola apa
yang aktif pada 3-5 candle acak (dari screenshot dashboard QSE Anda) dengan
output kolom 'patTxt76' di bawah, sebelum lanjut ke Fase 4 (backtest),
supaya kesalahan tidak ikut terbawa ke seluruh sistem memori nanti.
"""

import numpy as np
import pandas as pd

NAMES = [
    "Fib Golden Pocket", "Fib Golden Zone", "Golden Pocket Volume", "Fib 0.5", "Fib 0.382",
    "Fib 0.786", "Fib 0% Breakout", "Fib Ext 1.272", "Order Block SMC", "Break of Structure",
    "Change of Character", "Fair Value Gap", "Liquidity Sweep", "Pantul SnR", "Tembus SnR",
    "Breakout Triangle", "Double Top Bottom", "Inside Bar Break", "Candle Engulfing", "Pin Bar Rejection",
    "Morning Evening Star", "Three Soldiers Crows", "Pullback EMA20", "VWAP Volume", "SuperTrend Flip",
    "Donchian Break", "Infleksi Kalkulus", "Hurst Z-score", "Fractal7 Break", "Fractal7 Pullback",
    "GZ Extension", "Retest Extension", "Pantul Trendline", "Tembus Trendline", "Tembus Awan Ichimoku",
    "Ichimoku TK Cross", "MACD Cross", "MACD Divergensi", "RSI Divergensi", "RSI Balik Jenuh",
    "Angka Psikologi", "False Breakout", "Breakaway Gap", "Runaway Gap", "Exhaustion Gap",
    "Bollinger Squeeze", "Bollinger Reversal", "ADX DMI Cross", "OBV Divergensi", "Head and Shoulders",
    "Triple Top Horn", "Pola Lanjutan", "Pullback EMA50", "Pullback EMA200", "Pullback BB Mid",
    "Pantul Kijun", "Pantul Tenkan", "Pantul SuperTrend", "Fib 0.236", "RSI 50 Reclaim",
    "MACD Zero Cross", "Volume Spike Balik", "Outside Bar Balik", "Isi Gap", "Tembus Angka Bulat",
    "Range Expansion", "Tembus Kijun", "Pullback Donchian Mid", "Three Bar Reversal", "OBV Breakout",
    "ADX Awal Tren", "Tembus Golden Pocket", "Elliott Dorongan", "Wedge Break", "Rectangle Break",
    "Dow Konfirmasi",
]
assert len(NAMES) == 76


def build_76(core):
    o = core.df.join(core.out, how="left") if False else core.out  # alias singkat
    idx = core.df.index
    c, op, h, l = core.df["close"], core.df["open"], core.df["high"], core.df["low"]
    c1 = c.shift(1)
    v = core.df["volume"]

    def g(name):
        return o[name]

    cdlL, cdlS = g("cdlL"), g("cdlS")
    tUp, tDn = g("tUp"), g("tDn")
    rvol = g("rvol")
    rv10, rv11, rv12, rv13 = g("rv10"), g("rv11"), g("rv12"), g("rv13")
    a05, a04 = g("a05"), g("a04")
    upLeg = g("upLeg")
    inGP, inGZ, inEZ = g("inGP"), g("inGZ"), g("inEZ")
    r500, r382, r786, r236 = g("r500"), g("r382"), g("r786"), g("r236")
    fib0, e127 = g("fib0"), g("e127")
    gpTop, gpBot = g("gpTop"), g("gpBot")
    cdvB, cdvS = g("cdvB"), g("cdvS")
    obL, obS = g("obL"), g("obS")
    bosL, bosS, choL, choS = g("bosL"), g("bosS"), g("choL"), g("choS")
    fvFL, fvFS = g("fvFL"), g("fvFS")
    swpL, swpS = g("swpL"), g("swpS")
    atSup, atRes, supV, resV = g("atSup"), g("atRes"), g("supV"), g("resV")
    triL, triS, dblB, dblT = g("triL"), g("triS"), g("dblB"), g("dblT")
    ibL, ibS = g("ibL"), g("ibS")
    engL, engS = g("engL"), g("engS")
    pinL, pinS = g("pinL"), g("pinS")
    stLp, stSp = g("stL"), g("stS")
    sol3, cro3 = g("sol3"), g("cro3")
    ema20, ema50, ema200 = g("ema20"), g("ema50"), g("ema200")
    vwapV = g("vwap")
    stDir, stLine = g("stDir"), g("stLine")
    dcU, dcL, dcM = g("dcU"), g("dcL"), g("dcM")
    inflL, inflS = g("inflL"), g("inflS")
    d2, d3 = g("d2"), g("d3")
    hurst, hurstT, tSt, zPr, zThr = g("hurst"), g("hurstT"), g("tSt"), g("zPr"), g("zThr")
    frBU, frBD, frTU, frTD = g("frBU"), g("frBD"), g("frTU"), g("frTD")
    frD1, frU1 = g("frD1"), g("frU1")
    ezTop, ezBot = g("ezTop"), g("ezBot")
    atTlL, atTlS, tlBU, tlBD = g("atTlL"), g("atTlS"), g("tlBU"), g("tlBD")
    atChT, atChB, fanFU, fanFD = g("atChT"), g("atChB"), g("fanFU"), g("fanFD")
    kBU, kBD = g("kBU"), g("kBD")
    tkU, tkD, aboveK, belowK = g("tkU"), g("tkD"), g("aboveK"), g("belowK")
    kijun, tenkan = g("kijun"), g("tenkan")
    macdCU, macdCD, macdH, macdL = g("macdCU"), g("macdCD"), g("macdH"), g("macdL")
    divBM, divSM, divBR, divSR = g("divBM"), g("divSM"), g("divBR"), g("divSR")
    rsiRL, rsiRS, rsiV = g("rsiRL"), g("rsiRS"), g("rsi")
    psyBL, psyBS, psyLv = g("psyBL"), g("psyBS"), g("psyLv")
    fbL, fbS = g("fbL"), g("fbS")
    brkGU, brkGD, runGU, runGD, exhGU, exhGD = g("brkGU"), g("brkGD"), g("runGU"), g("runGD"), g("exhGU"), g("exhGD")
    islB, islT = g("islB"), g("islT")
    sqBU, sqBD, bbRL, bbRS = g("sqBU"), g("sqBD"), g("bbRL"), g("bbRS")
    dmiCU, dmiCD = g("dmiCU"), g("dmiCD")
    obvDL, obvDS, obvV = g("obvDL"), g("obvDS"), g("obv")
    hsL, hsS, hound = g("hsL"), g("hsS"), g("hound")
    trT, trB, hrT, hrB = g("trT"), g("trB"), g("hrT"), g("hrB")
    contU, contD = g("contU"), g("contD")
    bbM = g("bbM")
    pLo20, pHi20 = g("pLo20"), g("pHi20")
    hi10, lo10 = g("hi10"), g("lo10")
    lastGU, lastGD = g("lastGU"), g("lastGD")
    brange, avgRng = g("brange"), g("avgRng")
    adxV, diP, diM = g("adx"), g("diP"), g("diM")
    ph1, pl1 = g("ph1"), g("pl1")
    elwU, elwD = g("elwU"), g("elwD")
    wdgF, wdgR, rectC = g("wdgF"), g("wdgR"), g("rectC")
    dowUp, dowDn = g("dowUp"), g("dowDn")
    bodyR = g("bodyR")
    h1, l1, h2, l2 = h.shift(1), l.shift(1), h.shift(2), l.shift(2)

    cLa = {}
    cSa = {}

    cLa[0] = upLeg & inGP & cdlL
    cSa[0] = (~upLeg) & inGP & cdlS
    cLa[1] = upLeg & inGZ & cdlL
    cSa[1] = (~upLeg) & inGZ & cdlS
    cLa[2] = upLeg & inGP & cdvB & rv11
    cSa[2] = (~upLeg) & inGP & cdvS & rv11
    cLa[3] = upLeg & ((l - r500).abs() <= a05) & cdlL
    cSa[3] = (~upLeg) & ((h - r500).abs() <= a05) & cdlS
    cLa[4] = upLeg & ((l - r382).abs() <= a05) & cdlL
    cSa[4] = (~upLeg) & ((h - r382).abs() <= a05) & cdlS
    cLa[5] = upLeg & ((l - r786).abs() <= a05) & cdlL
    cSa[5] = (~upLeg) & ((h - r786).abs() <= a05) & cdlS
    cLa[6] = upLeg & (c > fib0) & (c1 <= fib0) & rv12
    cSa[6] = (~upLeg) & (c < fib0) & (c1 >= fib0) & rv12
    cLa[7] = upLeg & ((h - e127).abs() <= a05) & cdlL
    cSa[7] = (~upLeg) & ((l - e127).abs() <= a05) & cdlS
    cLa[8] = obL & rv10
    cSa[8] = obS & rv10
    cLa[9] = bosL & rv11
    cSa[9] = bosS & rv11
    cLa[10] = choL & rv12
    cSa[10] = choS & rv12
    cLa[11] = fvFL & tUp
    cSa[11] = fvFS & tDn
    cLa[12] = swpL & rv12
    cSa[12] = swpS & rv12
    cLa[13] = atSup & cdlL & (c > supV)
    cSa[13] = atRes & cdlS & (c < resV)
    cLa[14] = (c > resV) & (c1 <= resV) & rv13
    cSa[14] = (c < supV) & (c1 >= supV) & rv13
    cLa[15] = triL
    cSa[15] = triS
    cLa[16] = dblB
    cSa[16] = dblT
    cLa[17] = ibL
    cSa[17] = ibS
    cLa[18] = engL & tUp
    cSa[18] = engS & tDn
    cLa[19] = pinL & (atSup | inGP)
    cSa[19] = pinS & (atRes | inGP)
    cLa[20] = stLp & rv10
    cSa[20] = stSp & rv10
    cLa[21] = sol3 & tUp
    cSa[21] = cro3 & tDn
    cLa[22] = tUp & (l <= ema20) & (c > ema20) & cdlL
    cSa[22] = tDn & (h >= ema20) & (c < ema20) & cdlS
    cLa[23] = cdvB & (l <= vwapV) & (c > vwapV)
    cSa[23] = cdvS & (h >= vwapV) & (c < vwapV)
    cLa[24] = (stDir < 0) & (stDir.shift(1) > 0) & (c > ema200)
    cSa[24] = (stDir > 0) & (stDir.shift(1) < 0) & (c < ema200)
    cLa[25] = (c > dcU) & rv12
    cSa[25] = (c < dcL) & rv12
    cLa[26] = inflL & (d3 > 0) & cdlL
    cSa[26] = inflS & (d3 < 0) & cdlS
    cLa[27] = (hurst > hurstT) & (tSt.abs() > 2.01) & (zPr < -zThr * 0.6) & (d2 > 0)
    cSa[27] = (hurst > hurstT) & (tSt.abs() > 2.01) & (zPr > zThr * 0.6) & (d2 < 0)
    cLa[28] = frBU & frTU & cdvB
    cSa[28] = frBD & frTD & cdvS
    cLa[29] = frTU & (~frD1.isna()) & (l <= frD1 + a04) & cdlL
    cSa[29] = frTD & (~frU1.isna()) & (h >= frU1 - a04) & cdlS
    cLa[30] = upLeg & inEZ & cdlL & rv11
    cSa[30] = (~upLeg) & inEZ & cdlS & rv11
    cLa[31] = upLeg & (c > e127) & (l <= e127 + a04) & cdlL
    cSa[31] = (~upLeg) & (c < e127) & (h >= e127 - a04) & cdlS
    cLa[32] = (atTlL | atChB) & cdlL
    cSa[32] = (atTlS | atChT) & cdlS
    cLa[33] = (tlBU | fanFU) & rv12
    cSa[33] = (tlBD | fanFD) & rv12
    cLa[34] = kBU & rv12
    cSa[34] = kBD & rv12
    cLa[35] = tkU & aboveK & (c > kijun)
    cSa[35] = tkD & belowK & (c < kijun)
    cLa[36] = macdCU & (macdH > 0) & tUp
    cSa[36] = macdCD & (macdH < 0) & tDn
    cLa[37] = divBM & cdlL
    cSa[37] = divSM & cdlS
    cLa[38] = divBR & cdlL
    cSa[38] = divSR & cdlS
    cLa[39] = rsiRL & (atSup | inGP)
    cSa[39] = rsiRS & (atRes | inGP)
    cLa[40] = psyBL
    cSa[40] = psyBS
    cLa[41] = fbL
    cSa[41] = fbS
    cLa[42] = brkGU
    cSa[42] = brkGD
    cLa[43] = runGU
    cSa[43] = runGD
    cLa[44] = exhGD | islB
    cSa[44] = exhGU | islT
    cLa[45] = sqBU
    cSa[45] = sqBD
    cLa[46] = bbRL
    cSa[46] = bbRS
    cLa[47] = dmiCU
    cSa[47] = dmiCD
    cLa[48] = obvDL & cdlL
    cSa[48] = obvDS & cdlS
    cLa[49] = hsL
    cSa[49] = hsS | hound
    cLa[50] = trB | hrB
    cSa[50] = trT | hrT
    cLa[51] = contU
    cSa[51] = contD
    cLa[52] = tUp & (l <= ema50) & (c > ema50) & cdlL
    cSa[52] = tDn & (h >= ema50) & (c < ema50) & cdlS
    cLa[53] = (l <= ema200) & (c > ema200) & cdlL
    cSa[53] = (h >= ema200) & (c < ema200) & cdlS
    cLa[54] = tUp & (l <= bbM) & (c > bbM) & cdlL
    cSa[54] = tDn & (h >= bbM) & (c < bbM) & cdlS
    cLa[55] = (l <= kijun) & (c > kijun) & cdlL
    cSa[55] = (h >= kijun) & (c < kijun) & cdlS
    cLa[56] = aboveK & (l <= tenkan) & (c > tenkan) & cdlL
    cSa[56] = belowK & (h >= tenkan) & (c < tenkan) & cdlS
    cLa[57] = (stDir < 0) & (l <= stLine) & (c > stLine) & cdlL
    cSa[57] = (stDir > 0) & (h >= stLine) & (c < stLine) & cdlS
    cLa[58] = upLeg & ((l - r236).abs() <= a05) & cdlL
    cSa[58] = (~upLeg) & ((h - r236).abs() <= a05) & cdlS
    cLa[59] = (rsiV > 50) & (rsiV.shift(1) <= 50) & tUp
    cSa[59] = (rsiV < 50) & (rsiV.shift(1) >= 50) & tDn
    cLa[60] = (macdL > 0) & (macdL.shift(1) <= 0)
    cSa[60] = (macdL < 0) & (macdL.shift(1) >= 0)
    cLa[61] = (rvol >= 2.0) & (l <= pLo20) & cdlL
    cSa[61] = (rvol >= 2.0) & (h >= pHi20) & cdlS
    cLa[62] = (h > h1) & (l < l1) & (c > op) & (bodyR > 0.5)
    cSa[62] = (h > h1) & (l < l1) & (c < op) & (bodyR > 0.5)
    cLa[63] = (~lastGD.isna()) & (l <= lastGD) & (c > lastGD) & cdlL
    cSa[63] = (~lastGU.isna()) & (h >= lastGU) & (c < lastGU) & cdlS
    cLa[64] = (c > psyLv) & (c1 <= psyLv) & rv12
    cSa[64] = (c < psyLv) & (c1 >= psyLv) & rv12
    cLa[65] = (brange > avgRng * 1.5) & (c > hi10) & rv11
    cSa[65] = (brange > avgRng * 1.5) & (c < lo10) & rv11
    cLa[66] = (c > kijun) & (c1 <= kijun) & aboveK
    cSa[66] = (c < kijun) & (c1 >= kijun) & belowK
    cLa[67] = tUp & (l <= dcM) & (c > dcM) & cdlL
    cSa[67] = tDn & (h >= dcM) & (c < dcM) & cdlS
    cLa[68] = (l < l1) & (l1 < l2) & (c > h1)
    cSa[68] = (h > h1) & (h1 > h2) & (c < l1)
    cLa[69] = (obvV > obvV.rolling(20).max().shift(1)) & (c > c1)
    cSa[69] = (obvV < obvV.rolling(20).min().shift(1)) & (c < c1)
    cLa[70] = (adxV > 20) & (adxV.shift(1) <= 20) & (diP > diM)
    cSa[70] = (adxV > 20) & (adxV.shift(1) <= 20) & (diM > diP)
    cLa[71] = upLeg & (c > gpTop) & (c1 <= gpTop) & rv11
    cSa[71] = (~upLeg) & (c < gpBot) & (c1 >= gpBot) & rv11
    cLa[72] = elwU
    cSa[72] = elwD
    cLa[73] = wdgF & (c > ph1) & (c1 <= ph1) & rv11
    cSa[73] = wdgR & (c < pl1) & (c1 >= pl1) & rv11
    cLa[74] = rectC & (c > ph1) & (c1 <= ph1) & rv12
    cSa[74] = rectC & (c < pl1) & (c1 >= pl1) & rv12
    cLa[75] = dowUp & (c > ph1) & (c1 <= ph1) & rv11
    cSa[75] = dowDn & (c < pl1) & (c1 >= pl1) & rv11

    cLa_df = pd.DataFrame({NAMES[i]: cLa[i].fillna(False) for i in range(76)}, index=idx)
    cSa_df = pd.DataFrame({NAMES[i]: cSa[i].fillna(False) for i in range(76)}, index=idx)
    return NAMES, cLa_df, cSa_df
