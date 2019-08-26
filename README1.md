## Dockerfile Kullanımı:
Öncelikle projemizi çalıştırmak için interpreter ayarı yapmalıyız. Bunun için de Dockerfile'ımızı build etmeliyiz.
````dockerfile
docker build -t cbir/_withsift docker/
````
komutu ile Dockerfile'ımız build oldu. Build ettiğimiz Dockerfile artık local makinemizde bir Image haline geliyor . 
Geriye Image'ı run etmek kalıyor . Image'ı run etmek için de run komutuna Image ID veya Image tag vermeliyiz.
```dockerfile
docker run -it --rm -v ... cbir/_withsift 
```
komutu ile Image'ımızdan bir Container elde etmiş olduk sonrasında da bu Container'dan bir Volume oluşturup Container'ı 
silmiş olduk. Projemizde artık "interpreter" olarak interpreter seçeneklerinden Docker sekmesinden Dockerfile için 
verdiğimiz tag'i seçerek değişiklikleri onaylıyoruz. Böylece projemiz çalışmaya ve debug edilmeye hazır .

#Kod Anlatımı
##Database'teki Dosyalar için Özellik Çıkarımı
Kodun çalışması için "run" edeceğiniz dosyaların sırası rastgele olmamalıdır.Kodun doğru çalışması için yapılacak
işlem adımları:
* Başlangıç olarak "featureExtraction.py" dosyasına gidip bu dosyayı "run" layabilirsiniz. "featureExtraction.py" 
kısacası argparser kullanarak resimleri bulundurduğumuz (database) klasörün path'ini elde ediyoruz. Sonrasında path
kullanılarak database'den resimleri çekiyoruz.Databaseden çektiğimiz resimlerin SİFT İle keypointlerini , descriptorlarını
ve R-G-B piksellerinin histogramlarını hesaplıyoruz.Sonrasında "compute_codebook" fonksiyonu ile verdiğimiz k paramet-resine göre 
descriptor'ların centroid'lerini döndürüyoruz. Bu döndürülen değerler "codebook.pkl" olarak kaydediliyor. 
* SİFT ile bulduğumuz descriptor'ları kullanarak benzerlik kıyaslama algoritması olan BOW(bag of words) ile hesaplama 
yapılıyor. Bu hesaplama k-means ile gruplama ile histogram kullanımı içerir.
````python
 code, distortion = vq(f, codebook)
````
satırında code:gruplanmış descriptorları ifade eder ve shape=1xdesc sayısıdır.Sonrasında da gruplanmış descriptor'ların
k değişkeni ile birlikte histogramı çıkarlır. Ve oluşan histogram değerleri bow listesine eklenir ve "bow.pkl" olarak 
kayıt edilir. 
* Sıra TF-İDF algoritmasında ; bow algoritması ile birbirine çok benzemektedir.Burda "term frequently" durumu söz konusudur.
Histograma tabii tutulup bow listesine eklenip "bow.pkl" olarak kaydedilen veriler "load" işlemi ile tekrar yüklenir ve 
yüklenen pickle verileri transform fonskiyonları ile tekrar bow fonksiyonu güncellenir ve "tfidf.pkl" olarak kayıt edilir.
* En son aşama olan baseline özellikleri ise en başta hesaplanan R-G-B histogram değerlerini dikkate alır ve bu değerleri 
"base.pkl" olarak kayıt eder . 

**Böylelikle ilk aşama olan Özellik Çıkarımının sonuna gelmiş olduk.**

##Query Image ile Resim Benzerliği
Bir önceki aşamada database'deki resimlerin feature'larını çıkarmıştık.Bu aşamada ise "çıkardığımız feature'lara göre
vereceğimiz "query image"'ın feature'ları birbirine ne kadar benziyor ?" bu sorunun cevabını aramaktayız .
*Başlangıçta argparser ile query image'ın pathi verilir ve bu path kullanılarak , Query image'ın SİFT ile keypoint ve 
descriptor'ları , ayrıca "baseline" metodu ile Query Image'ın R-G-B histogram değerleri hesaplanır.
* Hesaplanan bu değerler birbiriyle kıyaslanır. Uzaklık hesabı yapılır.
* Uzaklık (benzerlik oranımız)  "cosine similarity" ile hesaplanmaktadır. Cosine Similarity bilmeyen arkadaşlar için
[inceleyiniz](http://www.selcukbasak.com/download/TurkceDokumanBenzerligi.pdf) dökümanı temel düzeyde yeterli olacaktır. 
* Uzaklık değerlerinin bulunduğu D dizisi "argsort" işlemi ile argüman sıralamasına tabii tutulur.Argüman sıralamasına
tabii tutulmuş dizinin ilk elemanı resmin kendisi olacağından ikinci elemanını alarak query image'a en yakın resmi çekmiş
oluruz.
**İşlemler temel olarak bütün karşılaşma adımlarında aynıdır. Bow için aynı işlem adımları , Tfidf için aynı işlem adımları
ve son olarak baseline için de aynı işlem adımları tekrar edilir.**
