import os
from google import genai
import ollama


class BaseLLM:
    def build_prompt(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, prompt):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        prompt = self.build_prompt(*args, **kwargs)
        return self.generate(prompt)


class Gemini(BaseLLM):
    def __init__(self, api_key, model="gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def build_prompt(self, temel, sembol, df, haberler_listesi, ai_rapor):
        son_veriler = df.tail(20).to_string() if not df.empty else "Yeterli veri yok."
        temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadı."
        haberler_metni = "\n".join(haberler_listesi) if haberler_listesi else "Haber verisi bulunamadı."

        return f"""Sen dünyanın en iyi hedge fonlarında çalışan bir borsa uzmanısın. 
        Sen karşındaki kişinin yatırım asistanısın; samimi, abartısız ve net bir dil kullanabilirsin. 
        Sakın yatırım tavsiyesi verme sadece elindeki bilgileri yorumla!

        ELİNDEKİ VERİLER {sembol} İÇİN:
        1. TEMEL ANALİZ: {temel_metin}
        2. HABER AKIŞI: {haberler_metni}
        3. TEKNİK VERİLER: {son_veriler}
        4. AI BOTU RAPORU: {ai_rapor}

        GÖREVİN: Gelecek senaryosu, Hedef Fiyat, Stop Seviyesi ve Güven Skoru içeren profesyonel bir rapor hazırla.
        """

    def generate(self, prompt):
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_output_tokens": 4096
                }
            )
            return response.text
        except Exception as e:
            return f"Gemini Hatası: {e}"


class OllamaLLM(BaseLLM):
    def __init__(self, model="qwen3:4b"):
        self.model = model
        # Docker içinde otomatik olarak ollama-server'a bağlanır,
        # Docker dışında (lokalde) localhost kullanır.
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)
        print(f"[OllamaLLM] Bağlantı hedefi: {self.host} | Model: {self.model}")

    @staticmethod
    def ollama_safe(text):
        if not isinstance(text, str):
            return str(text)
        return text.encode("ascii", "ignore").decode("utf-8")

    def build_prompt(self, temel, sembol, df, haberler_listesi, ai_rapor, analiz_sonucu):
        son_veriler = df.tail(20).to_string()
        ai_rapor_safe = self.ollama_safe(ai_rapor)
        analiz_sonucu_safe = self.ollama_safe(analiz_sonucu)
        temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadı."
        haberler_metni = "\n".join(haberler_listesi) if haberler_listesi else "Haber verisi bulunamadı."

        return self.ollama_safe(f"""GÖREVİN: Gemini'den gelen borsa raporunu denetle.
        Sadece teknik tutarsızlıkları (örn: RSI 80 ama 'Al' denmişse) tespit et.

        {sembol} İÇİN:
        1. TEMEL ANALİZ: {temel_metin}
        2. HABER AKIŞI: {haberler_metni}
        3. TEKNİK VERİLER: {son_veriler}
        4. AI SKORU: {ai_rapor_safe}
        5. GEMINI RAPORU: {analiz_sonucu_safe}

        ÇIKTI FORMATI:
        [MANTIKÇI NOTLARI]
        Onaylananlar:
        Duzeltmeler:
        Eklenenler:
        """)

    def generate(self, prompt):
        try:
            # Model yüklü değilse otomatik indir
            try:
                self.client.show(self.model)
            except Exception:
                print(f"[OllamaLLM] '{self.model}' bulunamadı, indiriliyor...")
                self.client.pull(self.model)

            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            # Yeni ollama kütüphanesi (>=0.1.0) obje döndürür, dict değil
            return response.message.content

        except Exception as e:
            return f"Ollama Hatası: {e}"