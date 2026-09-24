"""
QSE v135 - Fase 1: Lapisan Indikator Dasar & Statistik
=========================================================
Port dari Pine Script QSE_v135_DASBOR.txt baris ~168-473.

Cakupan fase ini (SUDAH):
- Indikator dasar: ATR, EMA20/50/200, RSI, Bollinger, VWAP, SuperTrend, Donchian, ADX/DMI, MACD, Ichimoku, OBV
- Pola candlestick dasar (f_cdl): cdlL, cdlS, engulfing, pinbar, star, 3 soldiers/crows
- Gap analysis (f_gap)
- Kalkulus tren: turunan d1/d2/d3 dari linreg (f_kOK)
- Statistik: Hurst exponent, korelasi, z-score, random walk block, regime
- Trap detection (bull/bear trap), angka psikologi
- Fractal (pivot 3 bar)
- Trendline sequential (pivot tracking ala Pine `var`)
- BTC gate (btcOkL/btcOkS) + CRS
- Bias lock (biasLg) memakai MTF 1H/4H/1D/1W

BELUM (menyusul fase berikutnya):
- 76 definisi pola lengkap (cLa/cSa) — perlu bagian SMC/chart-pattern
  (order block, BOS/CHoCH, FVG, liquidity sweep, H&S, triangle, wedge, dsb.)
  yang ada di baris ~483-617 pine script, belum saya porting.
- Fibonacci Golden Pocket/Golden Zone level (butuh bagian yang sama).
- Engine backtest 1140-slot + memori pola (baris ~688-910).
- Sistem ranking 5 slot saran + tracking posisi hidup (baris ~912-1250).
- Golden Moment / rapor robot A-D (baris ~1300-1383, f_lib).

KEPUTUSAN MTF (dikonfirmasi user): pakai pendekatan 4H SAJA, tanpa fetch
candle 1H terpisah, demi hemat API call untuk 400 pasar. Konsekuensinya:
  - m60 (bias 1H di Pine) DIAPROKSIMASI memakai m240 (bias 4H) --
    lihat parameter df_1h di mtf_bias(), default None -> pakai fallback ini.
  - 1D dan 1W tetap EXACT karena diturunkan dari resample 4H (bukan aproksimasi).
  - Konsekuensi: htfBull/htfBear (gerbang needMTF di Pine) akan sedikit lebih
    longgar dibanding TradingView asli, karena kondisi "m60==1 OR m240==1"
    otomatis terpenuhi begitu m240==1. Presisi turun terutama saat 1H dan 4H
    baru saja berbeda arah (whipsaw pendek). Ini trade-off yang disetujui user.
"""

import numpy as np
import pandas as pd


# ============================================================
# UTIL DASAR
# ============================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing, dipakai ta.rsi / ta.atr / ta.dmi di Pine."""
    return series.ewm(alpha=1 / length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return rma(tr, length)


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ru = rma(up, length)
    rd = rma(down, length)
    rs = ru / rd.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def linreg(series: pd.Series, length: int, offset: int = 0) -> pd.Series:
    """Setara ta.linreg: nilai regresi linier di ujung window, digeser offset bar."""
    def _fit(window):
        y = window.values
        x = np.arange(len(y))
        if np.any(np.isnan(y)):
            return np.nan
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * (len(y) - 1 - offset)
    return series.rolling(length).apply(_fit, raw=False)


def wilson_score(w: float, n: float) -> float:
    """f_wil: batas bawah interval Wilson (persen)."""
    if n <= 0:
        return 0.0
    p = w / n
    num = p + 1.9208 / n - 1.96 * np.sqrt((p * (1 - p) + 0.9604 / n) / n)
    return max(0.0, num / (1 + 3.8416 / n)) * 100


# ============================================================
# KELAS UTAMA
# ============================================================

class QSECore:
    """
    Menghitung seluruh lapisan indikator+statistik (non-pola, non-memori)
    untuk satu simbol dari OHLCV 4H, mengikuti QSE v135.

    df: DataFrame dengan kolom ['open','high','low','close','volume'],
        index datetime UTC, urut naik, interval 4H tetap.
    btc_df: DataFrame 4H BTCUSDT (untuk BTC gate & CRS), boleh None kalau
            simbolnya BTC sendiri.
    """

    def __init__(self, df: pd.DataFrame, btc_df: pd.DataFrame | None = None,
                 symbol: str = "", atr_len: int = 14):
        self.df = df.copy()
        self.btc_df = btc_df
        self.symbol = symbol
        self.atr_len = atr_len
        self.out = pd.DataFrame(index=df.index)

    # ---------- dasar ----------
    def basic(self):
        df, o = self.df, self.out
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

        o["atr"] = atr(df, self.atr_len)
        o["brange"] = h - l
        o["avgRng"] = o["brange"].rolling(20).mean()
        # ta.percentrank(atrV,100): posisi persentil ATR sekarang vs 100 bar
        o["atrPc"] = o["atr"].rolling(100).apply(
            lambda w: (w < w.iloc[-1]).sum() / (len(w) - 1) * 100 if len(w) > 1 else np.nan,
            raw=False)
        o["isSpk"] = o["brange"] > o["avgRng"] * 2.5  # spikeX default
        bars_since_spike = (~o["isSpk"]).groupby((o["isSpk"]).cumsum()).cumcount()
        o["mktOk"] = (~o["isSpk"]) & (bars_since_spike.shift(1).fillna(999) > 5) \
            & (o["atrPc"] <= 92) & (o["atrPc"] >= 8)

        o["bodyR"] = np.where(o["brange"] > 0, (c - df["open"]).abs() / o["brange"], 0)
        o["closR"] = np.where(o["brange"] > 0, (c - l) / o["brange"], 0.5)

        o["ema20"] = ema(c, 20)
        o["ema50"] = ema(c, 50)
        o["ema200"] = ema(c, 200)
        o["tUp"] = (c > o["ema50"]) & (o["ema50"] > o["ema200"])
        o["tDn"] = (c < o["ema50"]) & (o["ema50"] < o["ema200"])

        o["rvol"] = v / v.rolling(20).mean()
        o["rsi"] = rsi(c, 14)

        bbM = c.rolling(20).mean()
        bbSd = c.rolling(20).std(ddof=0)
        o["bbM"], o["bbU"], o["bbL"] = bbM, bbM + 2 * bbSd, bbM - 2 * bbSd

        tp = (h + l + c) / 3
        o["vwap"] = (tp * v).cumsum() / v.cumsum()  # catatan: Pine ta.vwap reset tiap sesi;
        # untuk crypto 24 jam dampaknya kecil, akan direvisi kalau perlu presisi sesi.

        o["dcU"] = h.rolling(20).max().shift(1)
        o["dcL"] = l.rolling(20).min().shift(1)
        o["dcM"] = (o["dcU"] + o["dcL"]) / 2

        o["capMove"] = (h.rolling(10).max() - l.rolling(10).min()).rolling(50).mean() / o["atr"]
        o["pLo20"] = l.rolling(20).min()
        o["pHi20"] = h.rolling(20).max()
        o["hi10"] = h.rolling(10).max().shift(1)
        o["lo10"] = l.rolling(10).min().shift(1)
        o["sw8L"] = l.rolling(8).min()
        o["sw8H"] = h.rolling(8).max()
        o["emaSlope"] = (o["ema20"] - o["ema20"].shift(3)) / o["atr"]
        return self

    # ---------- SuperTrend ----------
    def supertrend(self, mult: float = 3.0, length: int = 10):
        df, o = self.df, self.out
        h, l, c = df["high"], df["low"], df["close"]
        atrv = atr(df, length)
        hl2 = (h + l) / 2
        upper = hl2 + mult * atrv
        lower = hl2 - mult * atrv
        st = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        for i in range(len(df)):
            if i == 0:
                st.iloc[i] = upper.iloc[i]
                direction.iloc[i] = 1
                continue
            prev_st = st.iloc[i - 1]
            prev_dir = direction.iloc[i - 1]
            cur_upper = upper.iloc[i] if (upper.iloc[i] < prev_st or c.iloc[i - 1] > prev_st) else prev_st
            cur_lower = lower.iloc[i] if (lower.iloc[i] > prev_st or c.iloc[i - 1] < prev_st) else prev_st
            if prev_dir == 1:
                if c.iloc[i] < cur_lower:
                    direction.iloc[i] = -1
                    st.iloc[i] = cur_upper
                else:
                    direction.iloc[i] = 1
                    st.iloc[i] = cur_lower
            else:
                if c.iloc[i] > cur_upper:
                    direction.iloc[i] = 1
                    st.iloc[i] = cur_lower
                else:
                    direction.iloc[i] = -1
                    st.iloc[i] = cur_upper
        o["stLine"] = st
        o["stDir"] = direction  # Pine: stDir<0 artinya uptrend (arah line di bawah)
        return self

    # ---------- pola candle dasar (f_cdl) ----------
    def candle_patterns(self):
        df, o = self.df, self.out
        c, op, h, l = df["close"], df["open"], df["high"], df["low"]
        c1, op1 = c.shift(1), op.shift(1)
        c2, op2 = c.shift(2), op.shift(2)
        brange, bodyR, closR = o["brange"], o["bodyR"], o["closR"]

        o["cdlL"] = (c > op) & (bodyR >= 0.5) & (closR >= 0.65)
        o["cdlS"] = (c < op) & (bodyR >= 0.5) & (closR <= 0.35)
        o["engL"] = (c > op) & (c > h.shift(1)) & (op <= c1) & (c1 < op1)
        o["engS"] = (c < op) & (c < l.shift(1)) & (op >= c1) & (c1 > op1)
        o["pinL"] = (np.minimum(op, c) - l > brange * 0.55) & (c > op) & (brange > o["atr"] * 0.6)
        o["pinS"] = (h - np.maximum(op, c) > brange * 0.55) & (c < op) & (brange > o["atr"] * 0.6)
        o["stL"] = (c2 < op2) & ((c1 - op1).abs() < (h.shift(1) - l.shift(1)) * 0.4) \
            & (c > op) & (c > (op2 + c2) / 2)
        o["stS"] = (c2 > op2) & ((c1 - op1).abs() < (h.shift(1) - l.shift(1)) * 0.4) \
            & (c < op) & (c < (op2 + c2) / 2)
        o["sol3"] = (c > op) & (c1 > op1) & (c2 > op2) & (c > c1) & (c1 > c2) & (bodyR > 0.5)
        o["cro3"] = (c < op) & (c1 < op1) & (c2 < op2) & (c < c1) & (c1 < c2) & (bodyR > 0.5)

        o["psyLv"] = c.apply(self._psy_level)
        o["psyNear"] = (c - o["psyLv"]).abs() <= o["atr"] * 0.30
        o["psyBL"] = o["psyNear"] & (l <= o["psyLv"]) & (c > o["psyLv"]) & o["cdlL"]
        o["psyBS"] = o["psyNear"] & (h >= o["psyLv"]) & (c < o["psyLv"]) & o["cdlS"]

        o["trapU"] = (c < o["hi10"]) & (pd.concat([h, h.shift(1), h.shift(2)], axis=1).max(axis=1) > o["hi10"]) \
            & (o["rvol"] >= 1.2)
        o["trapD"] = (c > o["lo10"]) & (pd.concat([l, l.shift(1), l.shift(2)], axis=1).min(axis=1) < o["lo10"]) \
            & (o["rvol"] >= 1.2)
        return self

    @staticmethod
    def _psy_level(p):
        if p <= 0 or np.isnan(p):
            return np.nan
        mag = 10 ** np.floor(np.log10(p))
        return round(p / (mag / 4)) * (mag / 4)

    # ---------- gap analysis (f_gap) ----------
    def gaps(self, gap_min: float = 0.30):
        df, o = self.df, self.out
        h, l, c = df["high"], df["low"], df["close"]
        gU = l > h.shift(1)
        gD = h < l.shift(1)
        gATR = np.where(gU, (l - h.shift(1)) / o["atr"], np.where(gD, (l.shift(1) - h) / o["atr"], 0.0))
        o["gU"], o["gD"], o["gATR"] = gU, gD, gATR
        h20 = h.rolling(20).max().shift(1)
        l20 = l.rolling(20).min().shift(1)
        o["brkGU"] = gU & (gATR >= gap_min * 1.6) & (o["rvol"] >= 1.5) & (c > h20)
        o["brkGD"] = gD & (gATR >= gap_min * 1.6) & (o["rvol"] >= 1.5) & (c < l20)
        o["runGU"] = gU & (gATR >= gap_min) & o["tUp"] & (o["rvol"] >= 1.2) & (~o["brkGU"])
        o["runGD"] = gD & (gATR >= gap_min) & o["tDn"] & (o["rvol"] >= 1.2) & (~o["brkGD"])
        o["exhGU"] = gU & (gATR >= gap_min * 1.6) & (o["rsi"] > 70)
        o["exhGD"] = gD & (gATR >= gap_min * 1.6) & (o["rsi"] < 30)
        return self

    # ---------- indikator lanjutan ----------
    def indicators(self):
        df, o = self.df, self.out
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

        # DMI / ADX
        up_move = h.diff()
        down_move = -l.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = rma(tr, 14)
        plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), 14) / atr14
        minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), 14) / atr14
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        o["diP"], o["diM"], o["adx"] = plus_di, minus_di, rma(dx, 14)

        # OBV
        direction = np.sign(c.diff()).fillna(0)
        o["obv"] = (direction * v).cumsum()
        o["obvE"] = ema(o["obv"], 21)

        # Ichimoku
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
        senkouA = ((tenkan + kijun) / 2).shift(26)
        senkouB = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        o["tenkan"], o["kijun"] = tenkan, kijun
        o["kumoT"] = pd.concat([senkouA, senkouB], axis=1).max(axis=1)
        o["kumoB"] = pd.concat([senkouA, senkouB], axis=1).min(axis=1)
        o["aboveK"] = c > o["kumoT"]
        o["belowK"] = c < o["kumoB"]

        # MACD
        macdL = ema(c, 12) - ema(c, 26)
        macdSg = ema(macdL, 9)
        o["macdL"], o["macdSg"], o["macdH"] = macdL, macdSg, macdL - macdSg
        return self

    # ---------- kalkulus tren (d1/d2/d3) ----------
    def calculus(self, d1_thr: float = 0.015, d2_thr: float = 0.010):
        o = self.out
        c = self.df["close"]
        lrN = linreg(c, 50, 0)
        lrN_prev = linreg(c, 50, 1)
        o["d1"] = (lrN - lrN_prev) / o["atr"]
        o["d2"] = ((c - c.shift(5)) - (c.shift(5) - c.shift(10))) / (5 * o["atr"])
        o["d3"] = (((c - c.shift(3)) - (c.shift(3) - c.shift(6)))
                   - ((c.shift(3) - c.shift(6)) - (c.shift(6) - c.shift(9)))) / (3 * o["atr"])
        o["areaI"] = (c - o["ema50"]).rolling(50).mean() / o["atr"]
        o["kScL"] = ((o["d1"] > d1_thr).astype(int) + (o["d2"] > d2_thr).astype(int)
                     + (o["d3"] > 0).astype(int) + (o["areaI"] > 0).astype(int))
        o["kScS"] = ((o["d1"] < -d1_thr).astype(int) + (o["d2"] < -d2_thr).astype(int)
                     + (o["d3"] < 0).astype(int) + (o["areaI"] < 0).astype(int))
        return self

    # ---------- statistik ----------
    def statistics(self, stat_len: int = 100, hurst_thr: float = 0.55):
        o = self.out
        c = self.df["close"]
        bbSd = o["bbM"].rolling(20).std(ddof=0) if False else (c.rolling(20).std(ddof=0))
        zPr = (c - o["bbM"]) / bbSd.replace(0, np.nan)
        o["zPr"] = zPr

        idx = np.arange(len(c))
        def _corr(window):
            y = window.values
            x = idx[: len(y)]
            if np.any(np.isnan(y)):
                return np.nan
            return np.corrcoef(x, y)[0, 1]
        corrR = c.rolling(50).apply(_corr, raw=False)
        o["corrR"] = corrR
        o["rSq"] = corrR ** 2

        chg1 = c.diff(1)
        chg10 = c.diff(10)
        a = chg1.rolling(stat_len).std(ddof=0) ** 2
        b = chg10.rolling(stat_len).std(ddof=0) ** 2
        o["hurst"] = np.where(a > 0, 0.5 * np.log(b / (10 * a)) / np.log(10) + 0.5, 0.5)
        o["rwBlock"] = (o["hurst"] > 0.45) & (o["hurst"] < 0.55) & (o["rSq"] < 0.20) & (o["adx"] < 18)

        o["regime"] = np.select(
            [o["atrPc"] > 92, o["rSq"] < 0.25, o["d1"] > 0],
            ["VOLATILE", "SIDEWAYS", "TREND NAIK"], default="TREND TURUN")
        o["trending"] = (o["adx"] >= 25) & (o["rSq"] >= 0.35)
        o["ranging"] = (o["rSq"] < 0.25) | (o["adx"] < 20)
        return self

    # ---------- fractal (pivot 3 bar) ----------
    def fractals(self, frlen: int = 3):
        df, o = self.df, self.out
        h, l = df["high"], df["low"]
        n = len(df)
        frU = pd.Series(np.nan, index=df.index)
        frD = pd.Series(np.nan, index=df.index)
        for i in range(frlen, n - frlen):
            window_h = h.iloc[i - frlen:i + frlen + 1]
            if h.iloc[i] == window_h.max() and (window_h == h.iloc[i]).sum() == 1:
                frU.iloc[i] = h.iloc[i]
            window_l = l.iloc[i - frlen:i + frlen + 1]
            if l.iloc[i] == window_l.min() and (window_l == l.iloc[i]).sum() == 1:
                frD.iloc[i] = l.iloc[i]
        o["frU"], o["frD"] = frU, frD
        return self

    # ---------- trendline sequential (pivot tracking, mirip Pine `var`) ----------
    def trendlines(self, tl_per: int = 10, tol: float = 0.35):
        """Sequential loop, mengikuti persis pola `var float th1/th2 ...` di Pine."""
        df, o = self.df, self.out
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        n = len(df)
        atrv = o["atr"].values

        # pivot high/low (ta.pivothigh/pivotlow: perlu tl_per bar kiri & kanan, dikonfirmasi telat tl_per bar)
        piv_h = np.full(n, np.nan)
        piv_l = np.full(n, np.nan)
        for i in range(tl_per, n - tl_per):
            wh = h[i - tl_per:i + tl_per + 1]
            if h[i] == wh.max() and (wh == h[i]).sum() == 1:
                piv_h[i] = h[i]
            wl = l[i - tl_per:i + tl_per + 1]
            if l[i] == wl.min() and (wl == l[i]).sum() == 1:
                piv_l[i] = l[i]

        th1 = th2 = th1i = th2i = np.nan
        tl1 = tl2 = tl1i = tl2i = np.nan
        tlHv = np.full(n, np.nan)
        tlLv = np.full(n, np.nan)
        tlBU = np.zeros(n, dtype=bool)
        tlBD = np.zeros(n, dtype=bool)

        for i in range(n):
            if not np.isnan(piv_h[i]):
                th2, th2i = th1, th1i
                th1, th1i = piv_h[i], i - tl_per
            if not np.isnan(piv_l[i]):
                tl2, tl2i = tl1, tl1i
                tl1, tl1i = piv_l[i], i - tl_per

            slpH = (th1 - th2) / (th1i - th2i) if (not np.isnan(th1) and not np.isnan(th2) and th1i != th2i) else 0.0
            slpL = (tl1 - tl2) / (tl1i - tl2i) if (not np.isnan(tl1) and not np.isnan(tl2) and tl1i != tl2i) else 0.0
            cur_th = np.nan if np.isnan(th1) else th1 + slpH * (i - th1i)
            cur_tl = np.nan if np.isnan(tl1) else tl1 + slpL * (i - tl1i)
            tlHv[i], tlLv[i] = cur_th, cur_tl

            if i > 0:
                prev_th = tlHv[i - 1] if not np.isnan(tlHv[i - 1]) else cur_th
                prev_tl = tlLv[i - 1] if not np.isnan(tlLv[i - 1]) else cur_tl
                tlBU[i] = (not np.isnan(cur_th)) and c[i] > cur_th and c[i - 1] <= prev_th
                tlBD[i] = (not np.isnan(cur_tl)) and c[i] < cur_tl and c[i - 1] >= prev_tl

        o["tlHv"], o["tlLv"] = tlHv, tlLv
        o["tlBU"], o["tlBD"] = tlBU, tlBD
        tlTolP = atrv * tol
        o["atTlL"] = (~np.isnan(tlLv)) & (l <= tlLv + tlTolP) & (l >= tlLv - tlTolP * 2) & (c > tlLv)
        o["atTlS"] = (~np.isnan(tlHv)) & (h >= tlHv - tlTolP) & (h <= tlHv + tlTolP * 2) & (c < tlHv)
        return self

    # ---------- BTC gate & CRS ----------
    def btc_gate(self, use_gate: bool = True):
        o = self.out
        is_btc = "BTC" in self.symbol.upper()
        if self.btc_df is None or is_btc:
            o["btcUp"] = o["btcDn"] = False
            o["btcOkL"] = o["btcOkS"] = True
            return self
        b = self.btc_df.reindex(self.df.index, method="ffill")
        bC = b["close"].shift(1)
        bC1 = b["close"].shift(2)
        bE20 = ema(b["close"], 20).shift(1)
        bE50 = ema(b["close"], 50).shift(1)
        btcUp = (bC > bE20) & (bE20 > bE50)
        btcDn = (bC < bE20) & (bE20 < bE50)
        btcMom = bC > bC1
        o["btcUp"], o["btcDn"], o["btcMom"] = btcUp, btcDn, btcMom
        btc_alt = not is_btc
        o["btcOkL"] = (not use_gate) or (not btc_alt) or (~(btcDn & (~btcMom)))
        o["btcOkS"] = (not use_gate) or (not btc_alt) or (~(btcUp & btcMom))
        return self

    # ---------- MTF bias (pakai resample 1D/1W dari 4H; 1H butuh fetch terpisah -- lihat catatan atas) ----------
    def mtf_bias(self, df_1h: pd.DataFrame | None = None, bias_hold: int = 3):
        o = self.out
        df4h = self.df

        def _core(frame):
            c = frame["close"]
            a, b = ema(c, 20), ema(c, 50)
            t = np.select([(c > a) & (a > b), (c < a) & (a < b)], [1, -1], default=0)
            return pd.Series(t, index=frame.index).shift(1)

        m240 = _core(df4h)
        if df_1h is not None and len(df_1h) > 60:
            m60_raw = _core(df_1h)
            m60 = m60_raw.reindex(df4h.index, method="ffill")
        else:
            m60 = m240  # fallback approksimasi kalau 1H belum tersedia

        daily = df4h.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        weekly = df4h.resample("1W").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        mDy = _core(daily).reindex(df4h.index, method="ffill")
        mWk = _core(weekly).reindex(df4h.index, method="ffill")

        htfBull = (m60 >= 0) & (m240 >= 0) & ((m60 == 1) | (m240 == 1))
        htfBear = (m60 <= 0) & (m240 <= 0) & ((m60 == -1) | (m240 == -1))
        o["m60"], o["m240"], o["mDy"], o["mWk"] = m60, m240, mDy, mWk
        o["htfBull"], o["htfBear"] = htfBull, htfBear

        n = len(df4h)
        biasLk = np.zeros(n, dtype=int)
        biasCt = np.zeros(n, dtype=int)
        tUp = o["tUp"].values
        htfBull_v, htfBear_v = htfBull.values, htfBear.values
        mDy_v, mWk_v = mDy.values, mWk.values
        cur_lk, cur_ct = 0, 0
        for i in range(n):
            if cur_lk == 0:
                cur_lk = 1 if (htfBull_v[i] or (tUp[i] and not htfBear_v[i])) else -1
            nb = 1 if (htfBull_v[i] or (tUp[i] and not htfBear_v[i])) else -1
            if mDy_v[i] == mWk_v[i] and mDy_v[i] != 0:
                nb = mDy_v[i]
            if nb == cur_lk:
                cur_ct = 0
            else:
                cur_ct += 1
                if cur_ct >= bias_hold:
                    cur_lk = nb
                    cur_ct = 0
            biasLk[i], biasCt[i] = cur_lk, cur_ct
        o["biasLg"] = biasLk == 1
        o["biasCt"] = biasCt
        return self

    # ---------- jalankan semua ----------
    def run_all(self, df_1h: pd.DataFrame | None = None) -> pd.DataFrame:
        (self.basic().supertrend().candle_patterns().gaps().indicators()
         .calculus().statistics().fractals().trendlines().btc_gate())
        self.mtf_bias(df_1h=df_1h)
        return self.out
