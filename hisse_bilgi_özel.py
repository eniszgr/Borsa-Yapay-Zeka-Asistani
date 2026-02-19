import os
from dotenv import load_dotenv

import sys
import time
import warnings
import pandas as pd
import yfinance as yf
from ddgs import DDGS

from indicators.technical import teknik_analiz
from ai.pythorc import deeplearning
from ai.llm import GroqLLM, OllamaLLM


sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

load_dotenv()

pd.options.display.float_format = '{:.2f}'.format

BIST30_LISTESI = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", "BRSAN.IS",
    "CIMSA.IS", "SISE.IS", "CVKMD.IS", "PETKM.IS", "TKFEN.IS", "ANHYT.IS",
    "CCOLA.IS", "DOAS.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS",
    "GARAN.IS", "GUBRF.IS", "ULKER.IS", "ISCTR.IS", "KCHOL.IS", "KONTR.IS",
    "KRDMD.IS", "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS",
    "SAHOL.IS", "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS",
    "TUPRS.IS", "YKBNK.IS", "SMRTG.IS", "POLHO.IS", "ECILC.IS", "BORSK.IS",
    "ISCTR.IS", "ZOREN.IS", "ALFAS.IS"
]

BIST100_LISTESI = [
    "AEFES.IS", "AGHOL.IS", "AKBNK.IS", "AKCNS.IS", "AKSA.IS", "AKSEN.IS",
    "ALARK.IS", "ALBRK.IS", "ALGYO.IS", "ALKIM.IS", "ARCLK.IS", "ASELS.IS",
    "ASTOR.IS", "BERA.IS", "BIMAS.IS", "BRSAN.IS", "BRYAT.IS", "BUCIM.IS",
    "CANTE.IS", "CCOLA.IS", "CEMTS.IS", "CIMSA.IS", "DOAS.IS", "DOHOL.IS",
    "ECILC.IS", "EGEEN.IS", "EKGYO.IS", "ENJSA.IS", "ENKAI.IS", "EREGL.IS",
    "EUREN.IS", "FROTO.IS", "GARAN.IS", "GENIL.IS", "GESAN.IS", "GLYHO.IS",
    "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "IPEKE.IS", "ISCTR.IS", "ISDMR.IS",
    "ISGYO.IS", "ISMEN.IS", "IZMDC.IS", "KARSN.IS", "KCAER.IS", "KCHOL.IS",
    "KONTR.IS", "KORDS.IS", "KOZAA.IS", "KRDMD.IS", "MGROS.IS",
    "ODAS.IS", "OTKAR.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS",
    "SASA.IS", "SISE.IS", "SKBNK.IS", "SMRTG.IS", "SNGYO.IS", "SOKM.IS",
    "TAVHL.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS",
    "TTKOM.IS", "TTRAK.IS", "TUKAS.IS", "TUPRS.IS", "ULKER.IS", "VAKBN.IS",
    "VESBE.IS", "VESTL.IS", "YKBNK.IS", "YYLGD.IS", "ZOREN.IS"
]


def sembol_temizle(metin: str) -> str:
    tr_map = str.maketrans("igusocIGUSOC", "igusocIGUSOC")
    temiz_metin = metin.translate(tr_map).upper().strip()
    if not temiz_metin.endswith(".IS"):
        temiz_metin += ".IS"
    return temiz_metin


def temel_veriler(hisse):
    info = hisse.info
    temel = {
        "FK Orani (P/E)": info.get('trailingPE', 'Veri Yok'),
        "PD/DD (P/B)": info.get('priceToBook', 'Veri Yok'),
        "Kar Marji (%)": info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 'Veri Yok',
        "Brut Kar": info.get('grossProfits', 'Veri Yok'),
        "Toplam Gelir": info.get('totalRevenue', 'Veri Yok'),
        "Hisse Basina Kar (EPS)": info.get('trailingEps', 'Veri Yok'),
        "Sektor": info.get('sector', 'Bilinmiyor'),
        "Oneri": info.get('recommendationKey', 'Yok')
    }
    return temel


def input_alma() -> tuple:
    while True:
        ham_girdi = input("Bilgi almak istediginiz hissenin ismini giriniz: ").upper()
        sembol = sembol_temizle(ham_girdi)
        try:
            hisse = yf.Ticker(sembol)
            df = hisse.history(period="1y")
            if df.empty:
                print("Veri bulunamadi.")
                return input_alma()
            return hisse, sembol, df
        except Exception as e:
            print(f"Baglanti hatasi: {e}")


def sinyal_kontrol(df):
    try:
        son = df.iloc[-1]
        wonderkid = (son['Width'] < 0.15) and (son['RSI'] < 60)
        erken_uyari = (son['MACD_signal'] == 1) and (son['Signal'] == 1)
        ralli = (son['MACD_signal'] == 1) and (son['Signal'] == 1) and (son['Volume_signal'] == 1)

        if ralli:
            return True, "Ralli modu"
        elif wonderkid:
            return True, "Wonderkid modu"
        elif erken_uyari:
            return True, "Erken uyari"
        return False, "Temiz"
    except KeyError as e:
        return False, f"Eksik indikatör verisi: {e}"


def haber_verileri(sembol):
    haberler_listesi = []
    try:
        with DDGS() as ddgs:
            query = f"{sembol.replace('.IS', '')} hisse haberleri"
            result = ddgs.news(query, region="tr-tr", safesearch="off", max_results=5)
            for r in result:
                tarih = r.get('date', '')[:10]
                baslik = r.get('title', 'Baslik yok')
                kaynak = r.get('source', 'Bilinmiyor')
                haberler_listesi.append(f"-[{tarih}]{kaynak}:{baslik}")
    except Exception as e:
        print(f"Haber verisi cekilemedi. Hata: {e}")
    return haberler_listesi


def muhasebeci(hisse, bot=None):
    if bot is None:
        bot = deeplearning()
    try:
        df_muhasebeci = hisse.history(period="4y")
        sonuc = bot.analiz_et(df_muhasebeci)
        return (
            f"Yapay Zeka (Deep Learning) Modeli; {sonuc['suanki_fiyat']} TL olan guncel fiyatin, "
            f"%{sonuc['guven']} guven skoruyla {sonuc['tahmin']} TL hedefine ulasacagini "
            f"ve ana yonun {sonuc['yon']} olacagini ongöruyor."
        )
    except Exception as e:
        return f"Deeplearning'de hata: {e}"


def mod_tekli_detayli(groq_bot, ollama_bot, dl_bot):
    hisse, sembol, df = input_alma()

    try:
        print(f"\n{sembol} icin teknik ve temel veriler hesaplaniyor...")
        df = teknik_analiz(df)
        temel = temel_veriler(hisse)
        ai_rapor = muhasebeci(hisse, dl_bot)

        print("Internetten guncel haberler taraniyor...")
        haberler_listesi = haber_verileri(sembol)

        df_export = df.drop(["Dividends", "Stock Splits", "Volume"], axis=1, errors="ignore")
        df_export.index = df_export.index.tz_localize(None)
        excel_isim = f"{sembol}_detayli_analiz.xlsx"
        df_export.to_excel(excel_isim)
        print(f"Veriler {excel_isim} dosyasina kaydedildi.")

        print("\n[Groq] Analiz yapiliyor, lutfen bekleyin...")
        analiz_sonucu = groq_bot(temel, sembol, df, haberler_listesi, ai_rapor)

        print("[Ollama] Rapor denetleniyor, lutfen bekleyin...")
        final_rapor = ollama_bot(temel, sembol, df, haberler_listesi, ai_rapor, analiz_sonucu)

        print("=" * 60)
        print(f"BUYUK RESIM (OLLAMA + GROQ):\n{final_rapor}")
        print("=" * 60)
        print(f"DEEP LEARNING: {ai_rapor}")

    except Exception as e:
        print(f"Analiz sirasinda beklenmeyen hata: {e}")


def mod_bist30_tarama(groq_bot, ollama_bot, dl_bot):
    print("\nBIST 30 Taramasi Basliyor...")
    firsat_listesi = []

    for sembol in BIST30_LISTESI:
        try:
            hisse = yf.Ticker(sembol)
            df = hisse.history(period="1y")
            if df.empty:
                continue

            df = teknik_analiz(df)
            durum, sinyal = sinyal_kontrol(df)

            if durum:
                print(f"[+] Firsat tespit edildi: {sembol} ({sinyal}) listeye ekleniyor...")
                firsat_listesi.append((sembol, hisse, df))
            else:
                print(f"[-] {sembol}: {sinyal}")

        except Exception as e:
            print(f"Hata ({sembol}): {e}")

    if not firsat_listesi:
        print("\nBu BIST30 listesinde aktif yukselis trendi bulunan hisse bulunamadi :(")
        return

    toplam = len(firsat_listesi)
    print(f"\n{toplam} adet hisse tespit edildi. Detayli analiz basliyor...\n")

    for idx, (sembol, hisse, df) in enumerate(firsat_listesi):
        print(f"\n>>> {sembol} ANALIZ EDILIYOR ({idx+1}/{toplam}) <<<")
        temel = temel_veriler(hisse)
        time.sleep(2)
        haberler = haber_verileri(sembol)
        ai_rapor = muhasebeci(hisse, dl_bot)

        print("[Groq] Analiz yapiliyor...")
        analiz_sonucu = groq_bot(temel, sembol, df, haberler, ai_rapor)

        print("[Ollama] Rapor denetleniyor...")
        final_rapor = ollama_bot(temel, sembol, df, haberler, ai_rapor, analiz_sonucu)

        print(50 * '*')
        print(final_rapor)
        print(50 * '*')

        if idx < toplam - 1:
            bekleme_suresi = 10
            print(f"\n[Bekleme] Sonraki hisse icin {bekleme_suresi} saniye bekleniyor...")
            for i in range(bekleme_suresi, 0, -1):
                print(f"  Kalan: {i} saniye...  ", end="\r")
                time.sleep(1)
            print("Devam ediliyor!\n")


def mod_mega_tarama(dl_bot):
    print("\nMEGA TARAMA MODU BASLATILIYOR (Sadece Yerel Yapay Zeka)")
    yukselis_beklenenler = []

    for sembol in BIST100_LISTESI:
        try:
            print(f"Analiz ediliyor: {sembol:<10}", end="\r")
            hisse = yf.Ticker(sembol)
            df = hisse.history(period="1y")
            if df.empty:
                continue

            sonuc = dl_bot.analiz_et(df)
            yazi_rengi = "YUKSELIS" if "YÜKSELİŞ" in str(sonuc.get('yon', '')).upper() else "DUSUS"
            guven = sonuc.get('guven', 0)

            print(f"[{sembol:<10}] -> %{guven:<4} {sonuc.get('yon', 'N/A')} [{yazi_rengi}]")

            if "YÜKSELİŞ" in str(sonuc.get('yon', '')).upper() and guven > 60:
                yukselis_beklenenler.append((sembol, guven))

            time.sleep(0.2)

        except Exception as e:
            print(f"[{sembol}] Analiz Hatasi: {e}")

    print("\n" + "=" * 40)
    print(f"TARAMA BITTI! FIRSAT LISTESI ({len(yukselis_beklenenler)} Adet)")
    print("=" * 40)

    yukselis_beklenenler.sort(key=lambda x: x[1], reverse=True)
    for hisse, guven in yukselis_beklenenler:
        print(f"* {hisse:<10} - Guven: %{guven}")
    print("=" * 40 + "\n")


def main():
    groq_api_key = os.getenv("GROQ_API_KEY", "API_KEY_YOK")

    if groq_api_key == "API_KEY_YOK":
        print("[!] UYARI: GROQ_API_KEY bulunamadi! .env dosyanizi kontrol edin.")

    groq_yorumla = GroqLLM(api_key=groq_api_key)
    ollama_yorumla = OllamaLLM(model="qwen3:4b")
    dl_bot = deeplearning()

    while True:
        print("\n" + "=" * 40)
        print("HISSE ANALIZ YAPAY ZEKA ASISTANI")
        print("=" * 40)
        print("1. Tek Hisse Detayli Analiz (Groq + Ollama + PyTorch)")
        print("2. BIST30 Firsat Taramasi (Detayli)")
        print("3. Tek Hisse Sadece Sayisal Tahmin (PyTorch)")
        print("4. BIST100 Mega Tarama (Sadece PyTorch Hizli Tarama)")
        print("0. Cikis")
        print("=" * 40)

        secim = input("Lutfen bir islem seciniz (0-4): ").strip()

        if secim == "0":
            print("Sistemden cikiliyor. Bol kazanclar!")
            break
        elif secim == "1":
            mod_tekli_detayli(groq_yorumla, ollama_yorumla, dl_bot)
        elif secim == "2":
            mod_bist30_tarama(groq_yorumla, ollama_yorumla, dl_bot)
        elif secim == "3":
            hisse, sembol, df = input_alma()
            ai_rapor = muhasebeci(hisse, dl_bot)
            print(f"\n{sembol} Icin Sayisal Rapor:\n{ai_rapor}")
        elif secim == "4":
            mod_mega_tarama(dl_bot)
        else:
            print("Gecersiz tuslama! Lutfen menudeki numaralardan birini girin.")


if __name__ == "__main__":
    main()