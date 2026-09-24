"""
QSE v135 - Fase 5: Scoring, Seleksi Saran, Zona Entry/SL/TP, Rapor
========================================================================
Port baris 862-1000 & 1246-1250 Pine Script. Dijalankan SETELAH Fase 1-4
(qse_core, qse_patterns, qse_signals, qse_gates, qse_pattern76, qse_backtest).

Konstanta di bawah SEMUA dari input() asli (sudah di-grep ulang dari source,
bukan tebakan):
  wrTop=75, pfTop=2.0, wrDrop=5, pfDrop=0.2, tierHard=False
  grAbon=90, tierBon=60, freeDir=False, allowAnti=False, antiMinR=8.0
  gzBon=10, retBon=8, provW=6.0, biasBon=6.0, laxSmp=4, farPen=6
  memCap=8.0, memBon=20, rgBon=14, r1MinN=5, priorK=8.0
  tpReach=1.4, minRR=1.0, sigGrd="A + B", fibBon=4.0, bayesMin=58
  maxFar=3.5, mktTol=0.10, useRegFit=True, trendPen=10, corrBon=8
  famA = klasifikasi keluarga tiap 76 pola (0=netral,1=reversal,2=breakout)

PENYEDERHANAAN vs Pine asli (demi kepraktisan, sudah wajar untuk kebutuhan
sinyal Telegram per-candle, BUKAN dashboard interaktif TradingView):
  - Sistem "5 slot saran" Pine aslinya PERSISTEN antar-candle (slot yang
    sudah terisi bertahan sampai SL/TP/kadaluarsa, baru diganti). Karena
    bot kita jalan sebagai proses baru tiap 4 jam (bukan sesi live
    berkelanjutan), kita HITUNG ULANG top-5 dari nol tiap kali jalan --
    tanpa histori "posisi saran yang sedang berjalan". Konsekuensinya:
    tidak ada de-duplikasi lintas-run (pola yang sama bisa muncul lagi di
    run berikutnya walau "sebenarnya" masih posisi yang sama secara Pine).
    Ini keterbatasan yang disadari, bisa diperbaiki nanti dengan menyimpan
    state di file (mirip rencana commit JSON di Fase 6).
  - Rapor A-D disederhanakan jadi status EKSEKUSI/TAHAN + grade (bukan teks
    alasan panjang ala Pine), karena yang dibutuhkan untuk filter Telegram
    hanya "grade A dan EKSEKUSI".
  - TP1/TP2: Pine aslinya hanya py 1 TP per pola (dari tpv[sJ] yang menang
    skor). Karena permintaan awal Anda minta TP1 & TP2, saya definisikan
    TP1 = target utama (f_zone asli), TP2 = target lanjutan pakai
    tpv[sJ+1] (satu tingkat lebih jauh), dikarenakan tidak ada definisi
    TP2 eksplisit di source. INI INTERPRETASI SAYA, bukan dari Pine asli --
    tolong dikonfirmasi apakah ini sesuai maksud Anda.
"""

import numpy as np
import pandas as pd

FAM_A = [1, 1, 1, 1, 1, 1, 2, 0, 0, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 1, 1, 0, 0, 0, 0, 2, 1, 1, 2, 1,
         0, 0, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 2, 0, 2, 2, 0]

LV_L_KEYS = [
    "avg_gpTop_gpBot", "avg_gzTop_gzBot", "avg_gpTop_gpBot", "r500", "r382", "r786", "fib0", "e127",
    "obBT", "ph1", "ph1", "fvUB", "lo10", "supV", "resV", "ph1", "ph1", "h1", "ema20",
    "avg_gpTop_gpBot", "ema20", "ema20", "ema20", "vwap", "stLine", "dcU", "lrN", "lrN", "frU1",
    "frD1", "avg_ezTop_ezBot", "e127", "chBot_or_tlLv", "tlHv", "kumoT", "kijun", "ema20", "lo10",
    "lo10", "supV", "psyLv", "lo10", "lastGU", "lastGU", "lastGU", "bbU", "bbL", "ema20", "lo10",
    "neckU", "neckU", "ph1", "ema50", "ema200", "bbM", "kijun", "tenkan", "stLine", "r236", "ema20",
    "ema20", "lo10", "l1", "lastGD_or_ema20", "psyLv", "hi10", "kijun", "dcM", "h1", "ema20", "ema20",
    "gpTop", "ph1_or_ema20", "ph1_or_ema20", "ph1_or_ema20", "ph1_or_ema20",
]
LV_S_KEYS = [
    "avg_gpTop_gpBot", "avg_gzTop_gzBot", "avg_gpTop_gpBot", "r500", "r382", "r786", "fib0", "e127",
    "obSB", "pl1", "pl1", "fvDT", "hi10", "resV", "supV", "pl1", "pl1", "l1", "ema20",
    "avg_gpTop_gpBot", "ema20", "ema20", "ema20", "vwap", "stLine", "dcL", "lrN", "lrN", "frD1",
    "frU1", "avg_ezTop_ezBot", "e127", "chTop_or_tlHv", "tlLv", "kumoB", "kijun", "ema20", "hi10",
    "hi10", "resV", "psyLv", "hi10", "lastGD", "lastGD", "lastGD", "bbL", "bbU", "ema20", "hi10",
    "neckD", "neckD", "pl1", "ema50", "ema200", "bbM", "kijun", "tenkan", "stLine", "r236", "ema20",
    "ema20", "hi10", "h1", "lastGU_or_ema20", "psyLv", "lo10", "kijun", "dcM", "l1", "ema20", "ema20",
    "gpBot", "pl1_or_ema20", "pl1_or_ema20", "pl1_or_ema20", "pl1_or_ema20",
]


def _last_lvl(o, key, i):
    """Ambil nilai level (skalar, bar terakhir) sesuai kode kunci LV_L_KEYS/LV_S_KEYS."""
    last = lambda col: o[col].iloc[-1]
    ema20 = last("ema20")
    if key == "avg_gpTop_gpBot":
        return (last("gpTop") + last("gpBot")) / 2
    if key == "avg_gzTop_gzBot":
        return (last("gzTop") + last("gzBot")) / 2
    if key == "avg_ezTop_ezBot":
        return (last("ezTop") + last("ezBot")) / 2
    if key == "h1":
        return o["high"].iloc[-2] if "high" in o else np.nan
    if key == "l1":
        return o["low"].iloc[-2] if "low" in o else np.nan
    if key == "chBot_or_tlLv":
        v = last("chBot")
        return v if not np.isnan(v) else last("tlLv")
    if key == "chTop_or_tlHv":
        v = last("chTop")
        return v if not np.isnan(v) else last("tlHv")
    if key == "lastGD_or_ema20":
        v = last("lastGD")
        return v if not np.isnan(v) else ema20
    if key == "lastGU_or_ema20":
        v = last("lastGU")
        return v if not np.isnan(v) else ema20
    if key.endswith("_or_ema20"):
        base = key.replace("_or_ema20", "")
        v = last(base)
        return v if not np.isnan(v) else ema20
    return last(key)


def score_and_select(core, cLa: pd.DataFrame, cSa: pd.DataFrame, names: list, bt,
                      per_pattern: list, wr_top=75.0, pf_top=2.0, wr_drop=5.0, pf_drop=0.2,
                      tier_hard=False, gr_a_bon=90.0, tier_bon=60.0, free_dir=False,
                      allow_anti=False, anti_min_r=8.0, gz_bon=10.0, ret_bon=8.0, prov_w=6.0,
                      bias_bon=6.0, lax_smp=4, far_pen=6.0, mem_cap=8.0, mem_bon=20.0, rg_bon=14.0,
                      r1_min_n=5, prior_k=8.0, fib_bon=4.0, use_reg_fit=True, trend_pen=10.0,
                      corr_bon=8.0, w_gr_a=58, e_gr_a=0.25, p_gr_a=1.60, w_gr_b=48, e_gr_b=0.10,
                      p_gr_b=1.25, tp_reach=1.4, min_rr=1.0, top_n=5):
    """
    Menghasilkan daftar TOP-N saran (biasanya 5) untuk bar TERAKHIR (candle
    4H yang baru saja tutup) -- inilah yang dikirim ke Telegram di Fase 6.
    """
    df, o = core.df, core.out
    n = len(df)
    t = n - 1  # bar terakhir yang sudah tutup

    # ---- ringkasan per pola (dari hasil backtest Fase 4) ----
    net = {}
    win = {}
    loss = {}
    pf_ = {}
    wr_ = {}
    proven = {}
    banned = {}
    grade_ = {}
    for r in per_pattern:
        key = (r["idx"], r["arah"] == "LONG")
        net[key] = r["net_r"]; win[key] = r["win"]; loss[key] = r["loss"]
        pf_[key] = r["pf"]; wr_[key] = r["wr"]; proven[key] = r["proven"]
        banned[key] = r["banned"]; grade_[key] = r["grade"]

    def f_netD(i, isL): return net.get((i, isL), 0.0)
    def f_winD(i, isL): return win.get((i, isL), 0)
    def f_losD(i, isL): return loss.get((i, isL), 0)
    def f_pfD(i, isL): return pf_.get((i, isL), 0.0)
    def f_wrD(i, isL): return wr_.get((i, isL), 0.0)
    def f_proven(i, isL): return proven.get((i, isL), False)
    def f_ban(i, isL): return banned.get((i, isL), False)
    def f_haf(i, isL): return 1 if proven.get((i, isL), False) else 0  # aproksimasi "hafal"

    # ---- SCORING (sSc/sJ/sWb/sEx/sPF per pola, ambil bar terakhir) ----
    hurst_now = o["hurst"].iloc[-1]
    hurstT = o["hurstT"].iloc[-1]
    confMax = max(o["confL"].iloc[-1], o["confS"].iloc[-1])
    trending_now = bool(o["trending"].iloc[-1])
    ranging_now = bool(o["ranging"].iloc[-1])
    pPr = min(0.72, max(0.35, 0.42 + confMax * 0.035 + (0.05 if hurst_now > hurstT else 0)))

    bN, bW, bGW, bGL = bt.bN, bt.bW, bt.bGW, bt.bGL
    tpv = [0.8, 1.4, 1.8, 2.4, 3.2]
    brkA_map = {i: bool(v) for i, v in enumerate(
        [False, False, False, False, False, False, True, False, False, True, True, False, False,
         False, True, True, True, True, False, False, False, False, False, False, False, True,
         False, False, True, False, False, False, False, True, True, False, False, False, False,
         False, False, False, True, False, False, True, False, True, False, True, True, True,
         False, False, False, False, False, False, False, False, False, False, False, False, True,
         True, True, False, False, True, False, True, False, True, True, True])}

    sSc = {}
    sJ, sWb, sEx, sPF, sNn = {}, {}, {}, {}, {}
    for i in range(76):
        best_sc, best = -99999.0, None
        for j in range(4):  # jMax=3 -> j 0..3
            ix = i * 15 + 0 * 5 + j
            nn = bN[ix]
            if nn >= 2:
                wr = bW[ix] / nn * 100.0
                ex = (bW[ix] / nn) * tpv[j] - (1 - bW[ix] / nn)
                wb = _wilson(bW[ix] + pPr * prior_k, nn + prior_k)
                pfv = (bGW[ix] / bGL[ix]) if bGL[ix] > 0 else (9.9 if bGW[ix] > 0 else 0.0)
                fam = FAM_A[i]
                fitB = 0.0
                if use_reg_fit:
                    if trending_now:
                        fitB = -trend_pen if fam == 2 else (corr_bon if fam == 1 else corr_bon * 0.8)
                    elif ranging_now:
                        fitB = 4.0 if fam == 2 else (corr_bon if fam == 1 else -4.0)
                sc = (wb * 0.5 + wr * 0.15 + min(pfv, 3.0) * 9.0 + ex * 22.0 + min(nn, 30) * 0.45
                      + (fib_bon + gz_bon if (i <= 7 or i == 30 or i == 31) else 0.0)
                      + (0.0 if brkA_map[i] else ret_bon) + (-12.0 if nn < r1_min_n else 0.0) + fitB)
                if ex > -0.15 and sc > best_sc:
                    best_sc, best = sc, (j, wb, ex, pfv, nn)
        if best is not None:
            sSc[i] = best_sc
            sJ[i], sWb[i], sEx[i], sPF[i], sNn[i] = best

    def f_grade(wb, ex, pf):
        if wb >= w_gr_a and ex >= e_gr_a and pf >= p_gr_a:
            return 1
        if wb >= w_gr_b and ex >= e_gr_b and pf >= p_gr_b:
            return 2
        return 3

    biasLg = bool(o["biasLg"].iloc[-1])
    atrV = o["atr"].iloc[-1]
    close = df["close"].iloc[-1]

    def f_lvl(s, isL):
        keys = LV_L_KEYS if isL else LV_S_KEYS
        v = _last_lvl(o, keys[s], s)
        return v if not (v is None or (isinstance(v, float) and np.isnan(v))) else o["ema20"].iloc[-1]

    usd = set()
    hasil = []
    for k in range(top_n):
        wrK = max(wr_top - k * wr_drop, 55.0)
        pfK = max(pf_top - k * pf_drop, 1.5)
        bi, bd, bv = -1, biasLg, -99998.0
        for i in range(76):
            if i in usd:
                continue
            tgL = f_wrD(i, True) >= wrK and f_pfD(i, True) >= pfK
            tgS = f_wrD(i, False) >= wrK and f_pfD(i, False) >= pfK
            wb_i, ex_i, pf_i = sWb.get(i, 0), sEx.get(i, 0), sPF.get(i, 0)
            gr2 = wb_i >= w_gr_b and ex_i >= e_gr_b and pf_i >= p_gr_b
            grA = wb_i >= w_gr_a and ex_i >= e_gr_a and pf_i >= p_gr_a
            oL2 = f_proven(i, True) and gr2 and (tgL or not tier_hard)
            oS2 = f_proven(i, False) and gr2 and (tgS or not tier_hard)
            if not free_dir:
                if biasLg:
                    oS2 = oS2 and allow_anti and k >= 1 and f_netD(i, False) >= anti_min_r
                else:
                    oL2 = oL2 and allow_anti and k >= 1 and f_netD(i, True) >= anti_min_r
            if oL2:
                lv = f_lvl(i, True)
                skL = (sSc.get(i, -99999) - abs(close - lv) / atrV * far_pen + f_netD(i, True) * prov_w
                       + (bias_bon if biasLg else 0.0) + (4.0 if f_haf(i, True) > 0 else 0.0)
                       + (tier_bon if tgL else 0.0) + (gr_a_bon if grA else 0.0))
                if skL > bv:
                    bv, bi, bd = skL, i, True
            if oS2:
                lv = f_lvl(i, False)
                skS = (sSc.get(i, -99999) - abs(close - lv) / atrV * far_pen + f_netD(i, False) * prov_w
                       + (0.0 if biasLg else bias_bon) + (4.0 if f_haf(i, False) > 0 else 0.0)
                       + (tier_bon if tgS else 0.0) + (gr_a_bon if grA else 0.0))
                if skS > bv:
                    bv, bi, bd = skS, i, False
        if bi >= 0:
            usd.add(bi)
            grade_final = f_grade(sWb.get(bi, 0), sEx.get(bi, 0), sPF.get(bi, 0))
            hasil.append(_build_zone(core, bi, bd, names, sJ, tpv, grade_final, f_lvl, f_netD, f_wrD, f_pfD,
                                      wrK, pfK, tp_reach, min_rr))
    return hasil


def _wilson(w, nn):
    if nn <= 0:
        return 0.0
    p = w / nn
    num = p + 1.9208 / nn - 1.96 * np.sqrt((p * (1 - p) + 0.9604 / nn) / nn)
    return max(0.0, num / (1 + 3.8416 / nn)) * 100


def _build_zone(core, idx, is_long, names, sJ, tpv, grade_final, f_lvl, f_netD, f_wrD, f_pfD,
                 wr_req, pf_req, tp_reach, min_rr, sl_buf=0.45, sl_min_a=1.0, sl_max_a=2.5, max_far=3.5,
                 mkt_tol=0.10):
    df, o = core.df, core.out
    close = df["close"].iloc[-1]
    atrV = o["atr"].iloc[-1]
    sw8L, sw8H = o["sw8L"].iloc[-1], o["sw8H"].iloc[-1]
    wkBL, wkBS = o["wkBL"].iloc[-1], o["wkBS"].iloc[-1]
    capMove = o["capMove"].iloc[-1]
    lv = f_lvl(idx, is_long)

    base = sw8L - atrV * sl_buf if is_long else sw8H + atrV * sl_buf
    wSL = lv - wkBL if is_long else lv + wkBS
    inner = min(base, wSL) if is_long else max(base, wSL)
    rr = min(max(abs(lv - inner), atrV * sl_min_a), atrV * sl_max_a)
    sl = lv - rr if is_long else lv + rr

    j = sJ.get(idx, 0)
    obU = [o["resV"].iloc[-1], o["fib0"].iloc[-1], o["e127"].iloc[-1], o["dcU"].iloc[-1],
           o["kumoT"].iloc[-1], o["psyLv"].iloc[-1], o["bbU"].iloc[-1]]
    obD = [o["supV"].iloc[-1], o["fib0"].iloc[-1], o["e127"].iloc[-1], o["dcL"].iloc[-1],
           o["kumoB"].iloc[-1], o["psyLv"].iloc[-1], o["bbL"].iloc[-1]]

    def f_obs(level, up):
        b = level + atrV * 8 if up else level - atrV * 8
        tol = atrV * 0.4
        pool = obU if up else obD
        for v in pool:
            if np.isnan(v):
                continue
            if up:
                if v > level + tol and v < b:
                    b = v
            else:
                if v < level - tol and v > b:
                    b = v
        return b

    def tp_for_j(jj):
        obs = f_obs(lv, is_long)
        tp_c = min(lv + rr * tpv[jj], obs - atrV * 0.25) if is_long else max(lv - rr * tpv[jj], obs + atrV * 0.25)
        tp_c = min(tp_c, lv + capMove * tp_reach * atrV) if is_long else max(tp_c, lv - capMove * tp_reach * atrV)
        return max(tp_c, lv + rr * min_rr) if is_long else min(tp_c, lv - rr * min_rr)

    tp1 = tp_for_j(j)
    tp2 = tp_for_j(min(j + 1, 4))  # interpretasi TP2 (lihat catatan docstring modul)

    dz = abs(close - lv) / atrV
    brk = j in (0,) and False  # placeholder; brkA sudah dipakai di f_lvl-choice tapi tak dibawa ke sini
    order_ty = "TUNGGU" if dz > max_far else ("MARKET" if dz <= mkt_tol else ("LIMIT" if dz <= 1.2 else
               ("SCALED LIMIT" if dz <= 2.5 else "TUNGGU")))
    eksekusi = grade_final <= 2 and order_ty != "TUNGGU"

    return dict(
        pola=names[idx], arah="LONG" if is_long else "SHORT", grade={1: "A", 2: "B", 3: "C"}[grade_final],
        entry=round(float(lv), 6), sl=round(float(sl), 6), tp1=round(float(tp1), 6), tp2=round(float(tp2), 6),
        order_type=order_ty, eksekusi=bool(eksekusi), net_r=f_netD(idx, is_long),
        wr=f_wrD(idx, is_long), pf=f_pfD(idx, is_long), syarat_wr=round(wr_req, 1), syarat_pf=round(pf_req, 2),
        jarak_atr=round(float(dz), 2),
    )
