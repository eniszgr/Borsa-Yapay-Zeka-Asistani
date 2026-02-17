from google import genai
import ollama

class BaseLLM:
    def build_prompt(self,*args,**kwargs):
        raise NotImplementedError

    def generate(self,prompt):
        raise NotImplementedError
    def __call__(self,*args,**kwargs):
        prompt=self.build_prompt(*args,**kwargs)
        return self.generate(prompt)

class Gemini(BaseLLM):
    def __init__(self,api_key,model="models/gemini-flash-latest"):
        self.client=genai.Client(api_key=api_key)
        self.model=model
    
    def build_prompt(self,temel,sembol,df,haberler_listesi,ai_rapor):

        son_veriler = df.tail(20).to_string() if not df.empty else "Yeterli veri yok."
        
        temel_metin = "\n".join([f"- {k}: {v}" for k, v in temel.items()]) if temel else "Temel veri bulunamadı."
        haberler_metni="\n".join(haberler_listesi) if haberler_listesi else "Haber verisi bulunamadı."
        
        return f"""Sen dünyanın en iyi hedge fonlarında çalışan bir borsa uzmanısın. 
        Sen karşındaki kişinin yatırım asistanısın; samimi, abartısız ve net bir dil kullanabilirsin (arkadaşça ama profesyonel). Sakın yatırım tavsiyesi verme sadece elindeki bilgileri yorumla !

        ÖNEMLİ: Yaptıgın son yorumda "Neden?" sorusuna cevap ver. Terimlere bogmadan, çokta uzatmadan, sonucun hangi veriden kaynaklandıgını açıkla. (Örn: "RSI 30'un altında oldugu için ucuz dedim" gibi).

        ELİNDEKİ VERİLER {sembol} İÇİN:

        1. TEMEL ANALİZ:
        {temel_metin}

        2. HABER AKIŞI (Son 1 Ay):
        {haberler_metni}
        (Haberlerin fiyat üzerindeki duygu durumunu -Sentiment- analiz et.)

        3. TEKNİK VERİLER (Son 20 Gün):
        {son_veriler}

        4. Aİ BOTU YARDIMI:
        {ai_rapor}
        (bu rapor tamamen sayısal verilerle hesaplanmıştır bunU AYNEN YAZDIR ve yorumunda kullan!)

        KARAR MEKANİZMAN (Bu kurallara sadık kal):
        • RSI: <30 (Aşırı Ucuz/Al Fırsatı), >70 (Aşırı Pahalı/Sat Fırsatı), 30-70 (Nötr/Trendi Takip Et).
        • MACD: 1 (Al/Yükseliş), -1 (Sat/Düşüş).
        • SMA 50/200: Fiyat ortalamanın üzerindeyse POZİTİF, altındaysa NEGATİF.
        • VOLUME_SIGNAL: 1 ise Yükseliş gerçek (Güven artır), 0 ise Yükseliş zayıf (Tuzak olabilir).
        • BOLLINGER: Width (Bant Genişligi) düşüyorsa "SIKIŞMA" var (Patlama Yakın). Signal 1 ise yukarı, 0 ise yatay.
        • PIVOT: Fiyat > Pivot ise Hedef R1. Fiyat < Pivot ise Destek S1.
        • VOLATİLİTE: Yüksekse stop seviyesini biraz daha geniş tut, düşükse dar tut.

        GÖREVİN:
        Tüm verileri (Temel + Teknik + Haber) birleştir. Teknik veriler "AL" derken Haberler "KÖTÜ" ise güven skorunu düşür. Çelişkileri belirt.

        ÇIKTI FORMATIN (Tam olarak bu başlıkları kullan):

        📊 GELECEK SENARYOSU:
        (İki üç cümle ile ne bekliyorsun? Yükseliş/Düşüş/Yatay)
        Karar mekanizmanda kullandıgın(MACD,SMA50,SMA200,VOLUME_SİGNAL,BOLLINGER,PİVOT,VOLATİLİTE,WİDTH) degerlerini burda satır satır göster ve yorumla !

        🎯 HEDEF FİYAT:
        (R1 veya teknik analize göre net bir rakam ver)

        🛑 STOP SEVİYESİ:
        (S1 veya risk yönetimine göre net bir rakam ver)

        🔥 GÜVEN SKORU:
        (0-100 arası. Neden bu puanı verdigini parantez içinde tek cümleyle açıkla.)

        📰 HABER VE TEMEL ETKİ:
        (Haberler teknigi destekliyor mu? Şirket temel olarak saglam mı?(kar marjını burda kullan) - En fazla 3 cümle)

        📈 TEKNİK ÖZET:
        (Göstergeler uyumlu mu? Hangi indikatör en baskın sinyali veriyor?)

        📌 SON KARAR:
        (GÜÇLÜ AL / AL / TUT / SAT / GÜÇLÜ SAT)
        VERILER:
        {son_veriler}

        AI RAPOR:
        {ai_rapor}
        """
    
    def generate(self, prompt):
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature":0.7,
                    "top_p":0.95,
                    "max_output_tokens":4096
                }
            )
            return response.text
        except Exception as e:
            return f"Hata {e}"
        

class OllamaLLM(BaseLLM):
    
    def __init__(self,model="qwen3:4b"):
        self.model=model

    @staticmethod
    def ollama_safe(text):
        if not isinstance(text, str):
            return text
        return text.encode("ascii", "ignore").decode("utf-8")
    
    def build_prompt(self, temel, sembol, df, haberler_listesi, ai_rapor, analiz_sonucu, **kwargs):
        son_veriler = df.tail(20).to_string()
        ai_rapor_safe = self.ollama_safe(ai_rapor)
        analiz_sonucu_safe = self.ollama_safe(analiz_sonucu)
        
        return  self.ollama_safe(f"""GÖREVİN: Sen dünyanın en iyi hedge fonlarında çalışan bir denetleyicisin sana gelen metini elindeki veriler ile denetle.
        AMACIN:
        Metini yeniden YAZMA. Sadece rapordaki mantıksal hatalar ve eksik verileri tespit et. 

        1. TEKNİK_VERİLER:
        {son_veriler}

        2. AI_SKORU:
        {ai_rapor_safe}

        3. GEMINI_RAPORU (Bunu denetliyorsun):
        {analiz_sonucu_safe}

        KURALLAR:
        - Gemini'nin edebi diline karışma.
        - Sadece sayılar ve teknik indikatörler (RSI, MACD, Bollinger) dogru yorumlanmış mı ona bak.
        - Eger Gemini "Yükseliş" demiş ama RSI 90 ise (aşırı pahalı), bunu uyarı olarak ekle.
        - Eger Gemini önemli bir veriyi (örn: Hacim patlamasını) atlamışsa, onu ekle.

        ÇIKTI FORMATI (Sadece aşagıdakini yaz):

        [MANTIKÇI NOTLARI]
        ✅ ONAYLANANLAR:
        ⚠️ DÜZELTMELER:
        ➕ EKLENENLER:
        DENETLE:
        1. TEKNIK VERILER:

        2. AI RAPOR:

        3. GEMINI RAPOR:

        CIKTI:
        [MANTIKCI NOTLARI]
        """)
    
    def generate(self, prompt):
        try:
            response=ollama.chat(
                model=self.model,
                messages=[{"role":"user","content":prompt}]
            )
            return response["message"]["content"]
        except Exception as e :
            return f"Ollama hata {e}"