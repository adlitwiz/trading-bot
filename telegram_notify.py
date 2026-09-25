"""
QSE v135 - Fase 6b: Kirim Notifikasi ke Telegram
=====================================================
Pakai Bot API Telegram biasa (requests, tanpa library tambahan).
Token & chat_id diambil dari environment variable (diisi GitHub Secrets),
TIDAK PERNAH ditulis langsung di kode.
"""

import os
import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram(text: str, token: str = None, chat_id: str = None, parse_mode: str = "HTML"):
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID belum diset di environment.")
    url = TELEGRAM_API.format(token=token)
    # Telegram batas 4096 karakter per pesan -> pecah kalau kepanjangan
    chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)] or [text]
    for chunk in chunks:
        r = requests.post(url, data={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode,
                                      "disable_web_page_preview": True}, timeout=15)
        if r.status_code != 200:
            print(f"[WARN] gagal kirim Telegram: {r.status_code} {r.text}")


def format_signal(symbol: str, sig: dict) -> str:
    arah_emoji = "🟢" if sig["arah"] == "LONG" else "🔴"
    return (
        f"{arah_emoji} <b>{symbol}</b> — {sig['arah']} | Grade <b>{sig['grade']}</b>\n"
        f"Pola: {sig['pola']}\n"
        f"Order: {sig['order_type']}\n"
        f"Entry: <code>{sig['entry']}</code>\n"
        f"SL: <code>{sig['sl']}</code>\n"
        f"TP1: <code>{sig['tp1']}</code>   TP2: <code>{sig['tp2']}</code>\n"
        f"WR {sig['wr']}% | PF {sig['pf']} | Net {sig['net_r']}R | Jarak {sig['jarak_atr']} ATR\n"
    )


def build_report(all_signals: dict, run_time_str: str) -> str:
    """all_signals: {symbol: [sig_dict, ...]} -- hasil sudah difilter grade A + EKSEKUSI."""
    if not all_signals:
        return f"📊 QSE v135 Bot — {run_time_str}\n\nTidak ada sinyal Rapor A + EKSEKUSI candle 4H ini."
    lines = [f"📊 <b>QSE v135 Bot</b> — {run_time_str}", f"Sinyal Rapor A + EKSEKUSI: {sum(len(v) for v in all_signals.values())}\n"]
    for symbol, sigs in all_signals.items():
        for sig in sigs:
            lines.append(format_signal(symbol, sig))
    return "\n".join(lines)
