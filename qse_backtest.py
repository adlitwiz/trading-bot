"""
QSE v135 - Fase 4b: Engine Backtest & Memori Pola
======================================================
Port baris 688-837 Pine Script. INI JANTUNG sistem "belajar" QSE.

Fakta penting yang saya konfirmasi ulang dari input() (jangan diubah tanpa
alasan, ini sesuai default asli Pine):
  vMax=0, jMax=3  -> hanya varian v=0 dan TP index j=0..3 yang AKTIF
                     (dari kapasitas 1140 slot, cuma 76*4=304 yang kepakai).
  tpv = [0.8, 1.4, 1.8, 2.4, 3.2]  (useTpX=true -> tpv[0]=0.8)
  minWinD=5, minSmp=12, minNetD=1.5, minPFd=1.5, minWRd=55  (syarat "proven")
  banAfter=8, banNetR=-4.0   (syarat dibuang/ban)
  minKeep=12   (minimal pola yang boleh hidup, auto un-ban kalau kurang)
  wGrA=58, eGrA=0.25, pGrA=1.60   (syarat Grade A: Wilson/Expectancy/PF)
  wGrB=48, eGrB=0.10, pGrB=1.25  (syarat Grade B)
  needRg=True  (pola wajib tidak rugi di "regime" pasar saat ini)

BELUM diimplementasi di fase ini (ban manual/permanen dari histori entry
pengguna sendiri -- binL/binS/perL/perS -- karena bagian itu terkait
interaksi live "Saya entry" yang baru relevan di Fase 5):
  binL/binS/perL/perS semua diasumsikan False sepanjang backtest historis.
"""

import numpy as np
import pandas as pd

TPV = [0.8, 1.4, 1.8, 2.4, 3.2]
V_MAX = 0    # hanya v=0 dipakai (sesuai source asli)
J_MAX = 3    # index TP 0..3 dipakai (4 dari 5 target)
N_PAT = 76


def compute_mulai_uji(index: pd.DatetimeIndex, interval_ms: int) -> int:
    """Replikasi rumus `mulaiUji` Pine: titik mulai belajar/backtest,
    dihitung dari waktu candle terakhir, bukan dari jumlah bar tetap."""
    last_bar_time = int(index[-1].timestamp() * 1000)
    uji_w = 3000 * interval_ms
    tgl_uji = np.floor(last_bar_time / (uji_w / 2)) * (uji_w / 2) - uji_w
    pos = index.searchsorted(pd.Timestamp(tgl_uji, unit="ms", tz=index.tz))
    return int(max(0, min(pos, len(index) - 1)))


def wilson_lb(w: float, n: float) -> float:
    if n <= 0:
        return 0.0
    p = w / n
    num = p + 1.9208 / n - 1.96 * np.sqrt((p * (1 - p) + 0.9604 / n) / n)
    return max(0.0, num / (1 + 3.8416 / n)) * 100


def grade(wilson: float, expectancy: float, pf: float,
          w_a=58, e_a=0.25, p_a=1.60, w_b=48, e_b=0.10, p_b=1.25) -> int:
    if wilson >= w_a and expectancy >= e_a and pf >= p_a:
        return 1  # Grade A
    if wilson >= w_b and expectancy >= e_b and pf >= p_b:
        return 2  # Grade B
    return 0


class QSEBacktest:
    """
    Menjalankan simulasi backtest 304-slot aktif (76 pola x 4 target TP)
    sepanjang riwayat candle sejak `mulaiUji`, mengumpulkan statistik
    hidup per pola (net R, win, loss, PF, WR, Wilson, ban, proven).
    """

    def __init__(self, core, cLa: pd.DataFrame, cSa: pd.DataFrame, names: list,
                 interval_ms: int = 4 * 3600 * 1000,
                 min_win_d=5, min_smp=12, min_net_d=1.5, min_pf_d=1.5, min_wr_d=55,
                 ban_after=8, ban_net_r=-4.0, min_keep=12, need_rg=True):
        self.core = core
        self.df = core.df
        self.o = core.out
        self.cLa = cLa.values
        self.cSa = cSa.values
        self.names = names
        self.n = len(self.df)
        self.interval_ms = interval_ms
        self.min_win_d = min_win_d
        self.min_smp = min_smp
        self.min_net_d = min_net_d
        self.min_pf_d = min_pf_d
        self.min_wr_d = min_wr_d
        self.ban_after = ban_after
        self.ban_net_r = ban_net_r
        self.min_keep = min_keep
        self.need_rg = need_rg

    def run(self) -> dict:
        df, o, n = self.df, self.o, self.n
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        okL = o["okL"].fillna(False).values
        okS = o["okS"].fillna(False).values
        g0L = o["g0L"].fillna(False).values
        g0S = o["g0S"].fillna(False).values
        kOKL = o["kOKL"].fillna(False).values
        kOKS = o["kOKS"].fillna(False).values
        slLv = o["slLv"].values
        slSv = o["slSv"].values
        mktOk = o["mktOk"].fillna(False).values

        mulai_uji = compute_mulai_uji(df.index, self.interval_ms)

        n_slots = N_PAT * 15
        bS = np.zeros(n_slots, dtype=np.int8)
        bD = np.zeros(n_slots, dtype=np.int8)
        bSL = np.full(n_slots, np.nan)
        bTP = np.full(n_slots, np.nan)
        bN = np.zeros(n_slots, dtype=np.int32)
        bW = np.zeros(n_slots, dtype=np.int32)
        bGW = np.zeros(n_slots)
        bGL = np.zeros(n_slots)
        bEB = np.zeros(n_slots, dtype=np.int64)

        mRL = np.zeros(N_PAT); mWL = np.zeros(N_PAT, dtype=np.int32); mLL = np.zeros(N_PAT, dtype=np.int32)
        mRS = np.zeros(N_PAT); mWS = np.zeros(N_PAT, dtype=np.int32); mLS = np.zeros(N_PAT, dtype=np.int32)
        slcL = np.zeros(N_PAT, dtype=np.int32)
        slcS = np.zeros(N_PAT, dtype=np.int32)
        # regime bucket 0-3 dari kolom 'regime' (Phase1 statistics()): mapping sederhana
        regime_map = {"TREND NAIK": 0, "TREND TURUN": 1, "SIDEWAYS": 2, "VOLATILE": 3}
        rg_idx_arr = o["regime"].map(regime_map).fillna(2).astype(int).values
        rgR = np.zeros(N_PAT * 4)
        rgWn = np.zeros(N_PAT * 4, dtype=np.int32)
        rgLs = np.zeros(N_PAT * 4, dtype=np.int32)

        cLa, cSa = self.cLa, self.cSa

        for t in range(mulai_uji, n):
            rg_idx = rg_idx_arr[t]
            for i in range(N_PAT):
                c0L = cLa[t, i] and okL[t] and g0L[t] and kOKL[t]
                c0S = cSa[t, i] and okS[t] and g0S[t] and kOKS[t]
                v = 0
                iv_active = c0L or c0S
                if not iv_active:
                    # tetap proses slot yang sudah terbuka sebelumnya
                    pass
                for j in range(J_MAX + 1):
                    ix = i * 15 + v * 5 + j
                    if bS[ix] == 0 and mktOk[t] and (c0L or c0S):
                        dr = 1 if c0L else -1
                        sv = slLv[t] if c0L else slSv[t]
                        rq = abs(close[t] - sv)
                        if rq > 0:
                            bD[ix] = dr
                            bSL[ix] = sv
                            bTP[ix] = close[t] + dr * rq * TPV[j]
                            bEB[ix] = t
                            bS[ix] = 1
                    if bS[ix] == 1:
                        dd = bD[ix]
                        hs = low[t] <= bSL[ix] if dd == 1 else high[t] >= bSL[ix]
                        ht = high[t] >= bTP[ix] if dd == 1 else low[t] <= bTP[ix]
                        if ht and not hs:
                            bN[ix] += 1
                            bW[ix] += 1
                            bGW[ix] += TPV[j]
                            bS[ix] = 0
                            if j == 0:
                                if dd == 1:
                                    mRL[i] += TPV[j]; mWL[i] += 1
                                    slcL[i] = max(0, slcL[i] - 2)
                                else:
                                    mRS[i] += TPV[j]; mWS[i] += 1
                                    slcS[i] = max(0, slcS[i] - 2)
                                rgi = i * 4 + rg_idx
                                rgR[rgi] += TPV[j]; rgWn[rgi] += 1
                        elif hs:
                            bN[ix] += 1
                            bGL[ix] += 1.0
                            bS[ix] = 0
                            if j == 0:
                                if dd == 1:
                                    mLL[i] += 1; slcL[i] += 1
                                else:
                                    mLS[i] += 1; slcS[i] += 1
                                rgi = i * 4 + rg_idx
                                rgLs[rgi] += 1

        # ---- agregasi akhir (mengikuti blok Pine baris 788-828) ----
        results = []
        hidup = 0
        raw = []
        for i in range(N_PAT):
            for d, (w_arr, l_arr, g_arr, slc) in enumerate([
                (mWL, mLL, mRL, slcL), (mWS, mLS, mRS, slcS)
            ]):
                w, ls, gg = w_arr[i], l_arr[i], g_arr[i]
                nt = gg - ls
                pf = (gg / ls) if ls > 0 else (9.9 if gg > 0 else 0.0)
                wrd = (w / (w + ls) * 100.0) if (w + ls) > 0 else 0.0
                rgi = i * 4 + rg_idx_arr[-1]
                rg_ok = (not self.need_rg) or (rgWn[rgi] + rgLs[rgi]) < 2 or (rgR[rgi] - rgLs[rgi]) >= 0
                p0 = (w >= self.min_win_d and (w + ls) >= self.min_smp and nt >= self.min_net_d
                      and pf >= self.min_pf_d and wrd >= self.min_wr_d)
                bn = (slc[i] >= self.ban_after and nt <= self.ban_net_r)
                hidup += 0 if bn else 1
                wilson = wilson_lb(w, w + ls)
                exp_r = nt / (w + ls) if (w + ls) > 0 else 0.0
                gr = grade(wilson, exp_r, pf)
                raw.append(dict(idx=i, name=self.names[i], arah="LONG" if d == 0 else "SHORT",
                                 win=int(w), loss=int(ls), net_r=round(nt, 2), pf=round(pf, 2),
                                 wr=round(wrd, 1), wilson=round(wilson, 1), expectancy=round(exp_r, 3),
                                 grade=gr, proven=bool(p0 and not bn and rg_ok), banned=bool(bn), p0=bool(p0)))

        # aturan minKeep: kalau pola hidup < minKeep, un-ban pola dgn net>=0
        if hidup < self.min_keep:
            for r in raw:
                if r["banned"] and r["net_r"] >= 0:
                    r["banned"] = False
                    r["proven"] = r["p0"]

        self.mulai_uji_idx = mulai_uji
        self.mulai_uji_time = df.index[mulai_uji]
        # simpan array per-slot mentah utk sistem scoring (Fase 5)
        self.bN, self.bW, self.bGW, self.bGL = bN, bW, bGW, bGL
        return {"per_pattern": raw, "mulai_uji": self.mulai_uji_time, "hidup": hidup}
