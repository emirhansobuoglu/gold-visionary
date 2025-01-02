<h2>Altın Fiyat Tahmin Programı</h2>
Bu proje, Türkiye’deki altın fiyatlarının tarihsel verilerini otomatik olarak toplayarak gelecekteki fiyat hareketlerini tahmin etmek için kullanılacak bir veri seti oluşturmayı amaçlamaktadır. 
Veriler, Investing.com gibi sitelerden çekilmekte ve analizler için düzenlenmiş bir formatta kaydedilmektedir. Bu proje, altın fiyat tahminine yönelik veri toplama aşamasında kullanılmak üzere Python, Selenium ve Pandas kütüphanelerini kullanır.

<h4>Özellikler</h4>  
Veri Toplama: Selenium ile otomatik veri çekme.  
Dinamik Tarih Aralığı: Belirli bir tarih aralığındaki altın fiyatları çekilebilir.  
Pop-Up Yönetimi: Çeşitli pop-up engelleyiciler ile kesintisiz veri toplama.  
Veri Saklama: Pandas kullanılarak veriler .csv dosyasına kaydedilir.  

<h4>Gereksinimler</h4>  
Bu projeyi çalıştırmak için aşağıdaki araç ve kütüphaneler gereklidir:  
Python (>=3.7)  
Selenium  
Pandas  
Google Chrome ve uygun ChromeDriver  

<h4>Kurulum</h4>  
Python indirip kurun.  
Gerekli Python kütüphanelerini yükleyin:  
selenium, pandas  
ChromeDriver kurun ve sistem yolunuza ekleyin. ChromeDriver versiyonu, kullandığınız Chrome tarayıcının versiyonuyla uyumlu olmalıdır.  

<h4>Kullanım</h4>  
Python scriptini çalıştırın:  
goldvisionary.py  

Program, Investing.com sitesine bağlanarak otomatik olarak veri toplamaya başlar.  
Pop-up’ları kapatır, tarih aralığını seçer ve veriyi çeker.  
Elde edilen veriler altin_verileri.csv dosyasına kaydedilir.  

<h4>Veri Seti</h4>  
Veriler, altin_verileri.csv dosyasında saklanır ve her satırda tarih ve fiyat bilgilerini içerir.  
