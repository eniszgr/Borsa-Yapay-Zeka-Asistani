FROM python:3.10-slim

# standart olarak bütün imaginelar burada saklanıyor.
WORKDIR /app

# ilk satır ile linux paketleri güncellenir.
# curl: ollama healthcheck için gerekli (internete çıkıp bilgi alışverişi için)
# indirilen geçici paket listesi temizlenir.
RUN apt-get update && apt-get install -y \    
    curl \
    && rm -rf /var/lib/apt/lists/*

# . ile tüm dosyaları kopyalasaydık tüm kütüphaneler tekrar indirilmek zorunda kalırdı    
# pip ile indirdiğimizde bir kopyasını saklamasını engelliyoruz.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# her dosya klasör kopyalanıp taşınıyor
COPY . .

# proje ayağa kalktığında çalışacak komut
CMD ["python", "hisse_bilgi_özel.py"]