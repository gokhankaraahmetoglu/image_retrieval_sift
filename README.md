# Başlangıç
<p>Feature Extraction için featureExtraction.py'de feat ve base_feat olmak üzere iki boş dizi
oluşturulur . Sonrasında for döngüsü altında 500 tane fotoğraf çekilir ve her fotoğraf çekimi
esnasında ;</p> 
feat.append(computeFeatures(img))
base_feat.append(computeFeatures_baseline(img))

<p>işlemi yapılır. computeFeatures fonksiyonunda resim ilk önce RGB to Gray işlemine tabii 
tutulur . Sonrasında SİFT algoritması ile keypoint ve descriptorlar bulunur . Daha sonrasında 
da resimleri birbirinden ayıran özellik olan descriptorlar döndürülür.

```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.0)
kps, des = sift.detectAndCompute(gray, None)
```

computeFeatures_baseline fonksiyonunda ise resmin RGB halinin her bir channel'ı için histogram hesaplanır .</p> 

```python
rhist, rbins = np.histogram(img[:,:,0], 64, normed=True)
ghist, gbins = np.histogram(img[:,:,1], 64, normed=True)
bhist, bbins = np.histogram(img[:,:,2], 64, normed=True)
```

Daha sonrasında hesaplanan bu histogram değerleri concatenate(rhist,ghist,bhist) ile birleştirilir
ve return edilir . Bu işelm adımlarından sonra shape'lerden bahsedicek olursak :

500 tane resim çekildiği için feat.shape = 500 , base_feat.shape = 500 olacaktır . Dizilerin
elemanlarını inceleyecek olursak feat dizisi 500 elemanlı bir dizi olup her elemanı da bir dizi
ifade ediyor . Her elemanı dizi olan bu dizinin alt dizisinin elemanları descriptorlardan oluşmaktadır ve
her birinin shape'i descriptor sayısıx128 olacaktır.
base_feat  dizisi ise 3 channel için histogram yapıp range'i 64 olduğu için bu 500 dizinin her bir 
dizisi 1x192 eleman içermektedir .

compute Feature kısmını da açıkladıktan sonra esas koda dönecek olursak k means için k=50 parametesi 
verilir .

```python
codebook, distortion = kmeans(alldes, k)
code, distortion = vq(alldes, codebook) 
```

k-means uygulanır ve değişkenleri inceleyecek olursak alldes değişkeni feat anlamına gelmekte .
feat dizisi k-means k=50 'ye tabii tutulur . codebook.shape=kx128'dir. codebook merkezleri
bulundurur . distortion = result(lowest..)  code,distortion = vq(alldes , codebook) ile de
code.shape = 1xdescriptor sayısı ve code içerik olarak descriptorları gruplamıştır. distortion=1xdescriptor
sayısı ve descriptorları tutacaktır . Daha sonrasında "codebook" codebook pickle olarak kaydedilir.

Daha sonrasında Bag of Words ve Term Frequency - Inverse Document Frequency algoritmalar feature hesabı için
bow[] boş listi olurduk . Sonrasında Bag of Words için SIFT ile hesaplanan feat dizisinin
elemanlarını gezerken 

 code, distortion = vq(f, codebook) satırı ile code:gruplanmış descriptorlar , shape=1xdesc sayısı
 olur .Sonrasında da gruplanmış desc'lerin k değişkeni ile birlikte histogramı çıkarlır.
 ```python
bow_hist, _ = np.histogram(code, k, normed=True)
```
ve bow_hist her seferinde bow listesine eklenir . En son reshape işlemleri yapılarak bow.pkl kayıt
edilir.

Bag Of Words'ü hallettiğimize göre sıra Tf-idf algoritmasında . Tf-idf fonksiyonunda bow parametre alıp
fonksiyonda Transformer'lardan geçip döndürülür. Sonrasında da reshape işlemleri uygulanır ve
pickle olarak kayıt edilir.
Yazının en bşaında bahsetmiş olduğum base_Feat listesi de base.pkl olarak kayıt edilir.
Böylelikle featureExtraction kısmının sonuna gelmiş bulunmaktayız.
#Query İmage
Sorgu kısmında ilk başta soruglanacak resim dosyasından çekilir. Sonrasında BoW ile karşılaştırmaya
sokarız . Bu karşılaştırmalar için bir daha resim çekmeye gerek yoktur . Her şeyi biz pickle
dosyalarımıza kaydetmiştik zaten. 
```python
fv = pickle.load(open("bow.pkl", "rb") )
```
ile bow.pkl dosyamızı yükledik . Sonrasında query image'ın feature'larını hesaplarız ve newfeat
dizisinde tutarız . Hemen sonrasında ;
```python 
    codebook = pickle.load(open("codebook.pkl", "rb"))
```
codebook'u yükleriz .
```python
code, distortion = vq(newfeat, codebook)
```
ile newfeat'in k değişkenine göre gruplanmış halini elde ederiz ve code değişkenine atarız. 
code.shape=1xdesc sayısı 
```python
bow_hist, _ = np.histogram(code, k, normed=True)
```
sonrasında code k değişkeni ile histograma tabii tutulur . bow_hist.shape = 1x50 .
Böylelikle newfeat'i histograma tabi tuttuk . Sıra reshape edip fv'ye atayıp uzaklık hesaplamakta .
```python
D = computeDistances(fv)
```
ile computeDistances fonksiyonunu çağırır ve böylece uzaklığı döndürüp tüm uzaklıkları D listesine
ekleriz  . Sonrasında argüman sıralaması yaparız ve birinci indisimiz query image olacağından
ikinci indisi alırız. İkinci indis query image'a uzaklığı en düşük olan indistir. Sonrasında
bu resmi matplotlib kütüphanesi ile ekrana çizdiririz.  
tf-idf feature hesabı için de aynı hesaplamalar yapılır ancak bu sefer tf-idfk.pkl yüklenir ve
onla kıyaslanır.

Baseline-feature zaten histogram hesabı ile bulunmuştu. newfeat'in baseline feature'ları da bulunur
sonra computeDistances'a gönderilir ve gene aynı şekilde argüman sıralaması ile ikinci indis
alınır ve matplotlib kütphanesi ile ekrana çizdirilir. 


## TODO

- [ ] Plot bow_hist one or more time, and save to results/histogram folder
- [ ] Add functions like **compute_bow()**
- [ ] Create bow_not_normalized.pkl without histogram normalization
- [ ] Add argparser for query and featureExtraction 
- [ ] Create class called **Features**. Move all the feature computations (bow, tf-idf, ..) to Features class. 