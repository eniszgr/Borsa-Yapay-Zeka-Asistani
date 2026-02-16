from sklearn.ensemble import RandomForestClassifier # Tekrar Classifier'a döndük
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class borsa_muhasebe:
    def __init__(self):
        # Ayarları biraz daha 'Genel' tutuyoruz ki ezberlemesin
        self.model = RandomForestClassifier(
            n_estimators=400, 
            min_samples_leaf=3,     # Çok detaya inip kaybolmasın
            min_samples_split=8,
            random_state=42,
            class_weight='balanced', # Düşüş ve yükselişleri eşit önemse
            max_depth=12
        )
    
    def analiz_et(self, df):
        if df is None or df.empty:
            return {"yön": "VERİ YOK", "güven": "0"}
        
        # --- ÖZELLİKLER (Feature Engineering) ---
        df['Getiri'] = df['Close'].pct_change()
        df['Hacim_degisimi'] = df['Volume'].pct_change()
        df['Oynaklık'] = (df['High'] - df['Low']) / df['Close']
        
        #macd
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Fark'] = df['MACD'] - df['Signal_Line'] # Histogram
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0))
        lose = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_lose = lose.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_lose
        df['RSI'] = 100 - (100 / (1 + rs))
        
        #bollinger
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
        df['Bollinger_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
        df['Bollinger_Konum'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])
        
        #SMA_50,SMA_200
        df['SMA_50'] = df['Close'] / df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'] / df['Close'].rolling(window=200).mean()
        
        features_to_lag = ['Getiri', 'RSI', 'Hacim_degisimi', 'MACD_Fark']
        for feature in features_to_lag:
            df[f'{feature}_Lag1'] = df[feature].shift(1)
            df[f'{feature}_Lag2'] = df[feature].shift(2)
        
        df['Gun'] = df.index.dayofweek
        df['Momentum'] = df['Close'] / df['Close'].shift(10)
        
        self.ozellikler = ['Getiri', 'Hacim_degisimi', 'RSI', 'MACD_Fark', 
                           'Bollinger_Konum', 'SMA_50', 'SMA_200', 'Gun',
                           'Getiri_Lag1', 'RSI_Lag1', 'Hacim_degisimi_Lag1', 'MACD_Fark_Lag1',
                           'Getiri_Lag2', 'RSI_Lag2']
        
        gelecek_getiri = df['Close'].shift(-1) / df['Close'] - 1
        df['Target'] = (gelecek_getiri > 0.005).astype(int)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        bugun = df.iloc[[-1]][self.ozellikler]
        gecmis = df.dropna()

        # Son satırın target'ı belirsizdir (yarını bilmiyoruz), onu eğitimden çıkar
        X = gecmis[self.ozellikler].iloc[:-1]
        Y = gecmis['Target'].iloc[:-1]

        # --- EĞİTİM ve TEST ---
        tscv = TimeSeriesSplit(n_splits=5)
        skorlar = []
        
        for train_index, test_index in tscv.split(X):
            X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
            Y_train_cv, Y_test_cv = Y.iloc[train_index], Y.iloc[test_index]
            
            self.model.fit(X_train_cv, Y_train_cv)
            skor = self.model.score(X_test_cv, Y_test_cv)
            skorlar.append(skor)
        
        # 5 sınavın ortalamasını alıyoruz
        ortalama_basari = np.mean(skorlar)
        self.model.fit(X, Y)
        
        # Skor Hesapla
        print(f"🎯 Modelin Yön Bilme Başarısı: %{ortalama_basari*100:.2f}")
        
        improtance=self.model.feature_importances_
        en_onemli_indeks=np.argmax(improtance)
        en_onemli_ozellik=self.ozellikler[en_onemli_indeks]
        print(f"Karar verirkrn en öenmli özellik {en_onemli_ozellik} ")
        # Son vuruş: Tüm veriyle eğit
        self.model.fit(X, Y)

        if bugun.isnull().values.any():
            return {"yön": "HESAPLANAMADI", "güven": 0}

        # --- 5. TAHMİN ---
        olasiliklar = self.model.predict_proba(bugun)[0]
        yukselis_ihtimali = olasiliklar[1]
        
        # Yön Belirleme (Eşik Değeri %55)
        guven_yuzdesi = round(yukselis_ihtimali * 100, 2)
        
        if yukselis_ihtimali > 0.55:
            return {"yön": "YÜKSELİŞ", "güven": guven_yuzdesi}
        elif yukselis_ihtimali < 0.45:
            # Yükseliş ihtimali düşükse DÜŞÜŞ bekliyoruz demektir
            dusurs_guveni = round((1 - yukselis_ihtimali) * 100, 2)
            return {"yön": "DÜŞÜŞ", "güven": dusurs_guveni}
        else:
            return {"yön": "NÖTR (Yatay)", "güven": 50}
