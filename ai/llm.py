import os
import time
from groq import Groq
import ollama


class BaseLLM:
    def build_prompt(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, prompt):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        prompt = self.build_prompt(*args, **kwargs)
        return self.generate(prompt)


class GroqLLM(BaseLLM):
    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"[Groq] Model: {self.model}")

    def _veri_ozet(self, df):
        onemli_sutunlar = ['Close', 'RSI', 'MACD', 'Signal', 'BB_upper', 'BB_lower', 'Width', 'Volume_signal']
        mevcut = [s for s in onemli_sutunlar if s in df.columns]
        return df[mevcut].tail(20).round(2).to_string()

    def _istatistikler(self, df):
        try:
            guncel       = round(df['Close'].iloc[-1], 2)
            onceki       = round(df['Close'].iloc[-2], 2)
            haftalik_max = round(df['Close'].tail(252).max(), 2)
            haftalik_min = round(df['Close'].tail(252).min(), 2)
            ort_hacim    = round(df['Volume'].tail(20).mean(), 0) if 'Volume' in df.columns else "Veri yok"
            son_hacim    = round(df['Volume'].iloc[-1], 0) if 'Volume' in df.columns else "Veri yok"
            return guncel, onceki, haftalik_max, haftalik_min, ort_hacim, son_hacim
        except Exception:
            return "?", "?", "?", "?", "?", "?"

    def build_prompt(self, temel, sembol, df, haberler_listesi, ai_rapor):
        son_veriler = self._veri_ozet(df) if not df.empty else "Yeterli veri yok."
        guncel, onceki, yil_max, yil_min, ort_hacim, son_hacim = self._istatistikler(df)
        temel_metin  = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadi."
        haberler_metni = "\n".join(haberler_listesi[:3]) if haberler_listesi else "Haber yok."

        return f"""Sen bir borsa uzmanissin. Asagidaki verilere gore {sembol} icin kapsamli analiz yap.
Yatirim tavsiyesi verme, sadece verileri yorumla.
ONEMLI: Yaniti YALNIZCA TURKCE olarak yaz.

=== PİYASA VERİLERİ ===
Guncel Fiyat          : {guncel} TL
Onceki Gun Kapanis    : {onceki} TL
52 Haftalik Yuksek    : {yil_max} TL
52 Haftalik Dusuk     : {yil_min} TL
Son Gun Hacmi         : {son_hacim}
20 Gunluk Ort. Hacim  : {ort_hacim}

=== TEMEL VERİLER ===
{temel_metin}

=== SON HABERLER ===
{haberler_metni}

=== TEKNİK GÖSTERGELER (Son 20 Gün) ===
{son_veriler}

=== PYTORCH MODEL TAHMİNİ ===
{ai_rapor}

GOREV: Asagidaki formatta eksiksiz Turkce rapor yaz:

--- GROQ ANALİZ RAPORU ---
Guncel Fiyat              : {guncel} TL
52H Aralik                : {yil_min} TL - {yil_max} TL

[GUNLUK AL-SAT]
Gunluk Fiyat Tahmini      : [yarin kapanista beklenen fiyat] TL  |  Yon: [Yukselis/Dusus/Yatay]
Gunluk Al Bolgesi         : [bugun icin ideal alis fiyat araligi] TL
Gunluk Sat Bolgesi        : [bugun icin ideal satis fiyat araligi] TL
Gunluk Stop (Long)        : [long pozisyon icin stop seviyesi] TL
Gunluk Stop (Short)       : [short pozisyon icin stop seviyesi] TL

[VADE ANALİZİ]
Kisa Vade (1-4 hafta)     : [hedef fiyat] TL  |  Yon: [Yukselis/Dusus/Yatay]
Orta Vade (1-3 ay)        : [hedef fiyat] TL  |  Yon: [Yukselis/Dusus/Yatay]
Genel Stop Seviyesi       : [swing trade icin genel stop] TL

[DEĞERLENDİRME]
Hacim Analizi             : [son hacim ortalamaya gore yorumu]
Tahmini Dogruluk          : %[0-100]
Gelecek Senaryo           : [2-3 cumle]
Ozet                      : [1 cumle net degerlendirme]
"""

    def generate(self, prompt):
        max_deneme = 3
        bekleme = 10

        for deneme in range(max_deneme):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500
                )
                return response.choices[0].message.content

            except Exception as e:
                hata = str(e)
                if "429" in hata or "rate" in hata.lower():
                    if deneme < max_deneme - 1:
                        print(f"\n[!] Groq rate limit! {bekleme}s bekleniyor... (Deneme {deneme+1}/{max_deneme})")
                        for kalan in range(bekleme, 0, -1):
                            print(f"    Kalan: {kalan}s   ", end="\r")
                            time.sleep(1)
                        print()
                        bekleme += 10
                    else:
                        return "Groq rate limit: 3 denemede yanit alinamadi."
                else:
                    return f"Groq Hatasi: {e}"


class OllamaLLM(BaseLLM):
    def __init__(self, model="qwen3:4b"):
        self.model = model
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)
        print(f"[OllamaLLM] Baglanti hedefi: {self.host} | Model: {self.model}")

    @staticmethod
    def ollama_safe(text):
        if not isinstance(text, str):
            return str(text)
        return text.encode("ascii", "ignore").decode("utf-8")

    def build_prompt(self, temel, sembol, df, haberler_listesi, ai_rapor, groq_raporu):
        son_veriler    = df.tail(20).to_string()
        guncel         = round(df['Close'].iloc[-1], 2) if not df.empty else "Bilinmiyor"
        onceki         = round(df['Close'].iloc[-2], 2) if len(df) >= 2 else "Bilinmiyor"
        yil_max        = round(df['Close'].tail(252).max(), 2) if not df.empty else "?"
        yil_min        = round(df['Close'].tail(252).min(), 2) if not df.empty else "?"
        ai_rapor_safe  = self.ollama_safe(ai_rapor)
        groq_raporu_safe = self.ollama_safe(groq_raporu)
        temel_metin    = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadi."
        haberler_metni = "\n".join(haberler_listesi) if haberler_listesi else "Haber verisi bulunamadi."

        return self.ollama_safe(f"""Sen bir borsa analistisin. Groq raporunu denetle ve bagimsiz tahmin yap.
ONEMLI: Yaniti YALNIZCA TURKCE olarak yaz.

=== PİYASA VERİLERİ ===
Guncel Fiyat       : {guncel} TL
Onceki Gun Kapanis : {onceki} TL
52H Yuksek         : {yil_max} TL
52H Dusuk          : {yil_min} TL

=== TEMEL ANALİZ ===
{temel_metin}

=== HABERLER ===
{haberler_metni}

=== TEKNİK VERİLER (Son 20 Gün) ===
{son_veriler}

=== PYTORCH TAHMİNİ ===
{ai_rapor_safe}

=== GROQ RAPORU ===
{groq_raporu_safe}

GOREV: Asagidaki formatta Turkce rapor yaz:

--- OLLAMA DOGRULAMA RAPORU ---
Guncel Fiyat              : {guncel} TL

[GUNLUK AL-SAT]
Gunluk Fiyat Tahmini      : [yarin kapanista beklenen fiyat] TL  |  Yon: [Yukselis/Dusus/Yatay]
Gunluk Al Bolgesi         : [bugun icin ideal alis fiyat araligi] TL
Gunluk Sat Bolgesi        : [bugun icin ideal satis fiyat araligi] TL
Gunluk Stop (Long)        : [long pozisyon icin stop seviyesi] TL
Gunluk Stop (Short)       : [short pozisyon icin stop seviyesi] TL

[VADE ANALİZİ]
Kisa Vade (1-4 hafta)     : [hedef fiyat] TL  |  Yon: [Yukselis/Dusus/Yatay]
Orta Vade (1-3 ay)        : [hedef fiyat] TL  |  Yon: [Yukselis/Dusus/Yatay]
Genel Stop Seviyesi       : [swing trade icin genel stop] TL

[DEĞERLENDİRME]
Groq ile Uzlasma          : [Evet / Kismi / Hayir]
Tahmini Dogruluk          : %[0-100]
Teknik Tutarlilik         : [Groq raporundaki tutarsizliklari belirt]
Nihai Yorum               : [1-2 cumle net degerlendirme]
""")

    def generate(self, prompt):
        try:
            try:
                self.client.show(self.model)
            except Exception:
                print(f"[OllamaLLM] '{self.model}' bulunamadi, indiriliyor...")
                self.client.pull(self.model)

            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.message.content

        except Exception as e:
            return f"Ollama Hatasi: {e}"