# Installation

## Docker
Öncelikle projemizi çalıştırmak için interpreter ayarı yapmalıyız. Bunun için de Dockerfile'ımızı build etmeliyiz.
````dockerfile
docker build -t cbir:sift docker/
````
komutu ile Dockerfile'ımız build oldu. Build ettiğimiz Dockerfile artık local makinemizde bir **Image** haline geliyor . 
Geriye Image'ı run etmek kalıyor . Image'ı run etmek için de run komutuna **Image ID** veya **Image tag** vermeliyiz.
```dockerfile
docker run -it --rm -v `pwd`/workspace cbir:sift 
```
komutu ile **Image'ımızdan** bir Container elde etmiş olduk sonrasında da bu Container'dan bir Volume oluşturup Container'ı 
silmiş olduk. Projemizde artık "interpreter" olarak interpreter seçeneklerinden Docker sekmesinden Dockerfile için 
verdiğimiz tag'i seçerek değişiklikleri onaylıyoruz. Böylece projemiz çalışmaya ve debug edilmeye hazır .

# Code

## Feature Extraction (Create Database)

```
python3 featureExtraction.py
```

## Query New Image

```
python3 query2.py --db_path=data/12.jpg
```

# Kod Anlatımı
## Database'teki Dosyalar için Özellik Çıkarımı
Kodun çalışması için "run" edeceğiniz dosyaların sırası rastgele olmamalıdır.Kodun doğru çalışması için yapılacak
işlem adımları:
* Başlangıç olarak "**featureExtraction.py**" dosyasına gidip bu dosyayı "run" 'layabilirsiniz.

"featureExtraction.py" dosyasında **argparser** kullanarak resimleri bulundurduğumuz (database) klasörün **path'ini** elde
ediyoruz. Sonrasında path kullanılarak database'den resimleri for döngüsü altında çekiyoruz.Databaseden çektiğimiz resimlerin
**SİFT** ile **keypointlerini** , **descriptorlarını** ve **R-G-B piksellerinin histogramlarını** hesaplıyoruz.
```python
feat.append(computeFeatures(img))
base_feat.append(computeFeatures_baseline(img))
```
**computeFeatures** fonksiyonu SİFT ile keypoint ve descriptor hesabı yapıyor.

**computeFeatures_baseline** fonksiyonu da R-G-B histogramlarını hesaplıyor.

**SIFT ile Hesaplama:**
```python
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.0)
kps, des = sift.detectAndCompute(gray, None)
```
Öncesinde resim "gray" yapılarak channel (RGB) tek channel'a inmiş oluyor ve işlem kolaylığı sağlıyor.
Daha sonrasında ise SİFT hazır fonksiyonu uygulanıyor ve sonucunda "des" değeri **(descriptor)** döndürülüyor.
Bu değerler feat [] dizisine **append** ediliyor.

**Baseline ile Hesaplama**
```python
rhist, rbins = np.histogram(img[:,:,0], 64, normed=True)
ghist, gbins = np.histogram(img[:,:,1], 64, normed=True)
bhist, bbins = np.histogram(img[:,:,2], 64, normed=True)
```
Alınan resmin R-G-B değerleri direkt olarak histograma tabi tutuluyor ve her histogram sonucu **concatenate** ile
bir dizide tutuluyor ve dizi return ediliyor.
Bu değerler base_feat [] dizisine **append** ediliyor.

Bu işlemler için oluşturulmuş feat ve base_feat dizileri 1x500'lük dizilerdir.Ancak tüm elemanları da dizidir.
**feat[]** dizisi içinde **keypointsizex128**eleman(shape) bulunduran 500 tane diziden oluşmuştur.
**baseline_feat[]** dizisi içinde **1x192**eleman(shape) bulunduran 500 tane diziden oluşmuştur.

Sonrasında **compute_codebook** fonksiyonu ile ;

```python
alldes = np.vstack(feat)
```
İlk olarak feat dizisi vstack edilerek **tüm keypointlerin sayısı x128**'lik bir shape elde edilir. 

Sonrasında verdiğimiz **k** parametresine göre (k=50) descriptor'ların centroid'lerini döndürüyoruz.Bu işlem

````python
codebook, _ = kmeans(alldes, k)
````
 ile yapılır. Ve **k-means**'ten döndürülen **codebook** değeri **codebook.pkl** olarak kaydedilir.
 codebook shape olarak **50x128** olup içinde descriptor centroidleri bulundurur.
  
* Daha sonrasında SİFT ile bulduğumuz **feat[]** yani descriptor'ları **(keypointx128)** kullanarak benzerlik kıyaslama 
algoritması olan **BOW(bag of words)** ile hesaplama yapılıyor. Bu hesaplama **k-means** gruplaması ile **histogram** kullanımı içerir.
İlk önce ;
```python
codebook = pickle.load(open("codebook.pkl", "rb"))
```
ile **codebook.pkl** değerleri **codebook**'a yüklenir.Sonrasında her **feat[]** dizisi elemanı için
````python
 code, distortion = vq(f, codebook)
````
işlemi yapılır. Bu işlemde **code**:gruplanmış descriptorları ifade eder ve shape=1xkeypointsayısı olacaktır.Sonrasında da 
gruplanmış descriptor'ların **k** değişkeni ile birlikte histogramı çıkarlır. Ve oluşan histogram değerleri bow listesine eklenir ve **bow.pkl** olarak 
kayıt edilir. 
```python
bow_hist, _ = np.histogram(code, k, normed=True)
bow.append(bow_hist)
```
Bu işlem her bir fotoğraf için yapıldıktan sonra **bow[]** dizisi **vstack** edilerek **bow.pkl** olarak kayıt
edilir.

* Sıra **TF-İDF** algoritmasında ; bow algoritması ile birbirine çok benzemektedir.Burda **term frequently** durumu söz konusudur.
Histograma tabii tutulup bow listesine eklenip **bow.pkl** olarak kaydedilen veriler **load** işlemi ile tekrar yüklenir.
```python
all_bow = pickle.load(open("bow.pkl", "rb"))
```
Sonrasında yüklenen pickle verileri **transform fonskiyonları** ile tekrar bow fonksiyonu güncellenir ve **tfidf.pkl** olarak kayıt edilir.
```python
transformer = TfidfTransformer(smooth_idf=True)
t = transformer.fit_transform(all_bow).toarray()

t = normalize(t, norm='l2', axis=1)

pickle.dump(t, open("tfidf.pkl", "wb"))
```

* En son aşama olan **baseline özellikleri** ise en başta hesaplanan **R-G-B histogram değerlerini** vstack yapar ve bu değerleri 
**base.pkl** olarak kayıt eder . 
```python
base_feat = np.vstack(base_feat)

pickle.dump(base_feat, open("base.pkl", "wb"))
```
**Böylelikle ilk aşama olan Özellik Çıkarımının sonuna gelmiş olduk.**

## Query Image ile Resim Benzerliği
Bir önceki aşamada database'deki resimlerin feature'larını çıkarmıştık.Bu aşamada ise "çıkardığımız feature'lara göre
vereceğimiz **query image**'ın feature'ları birbirine ne kadar benziyor ?" bu sorunun cevabını aramaktayız .
*Başlangıçta **argparser** ile **query image'ın pathi** verilir ve bu path kullanılarak , Query image'ın **SİFT** ile **keypoint ve 
descriptor'ları** , ayrıca **"baseline" metodu** ile Query Image'ın **R-G-B histogram değerleri** hesaplanır. Bir önceki adımdaki
hesaplarla birebir aynıdır.Burada sadece tüm veri seti yerine sadece bir resmin feature'ları çıkartılır.
Sıra çıkarılan feature'lar ile pkl olarak kaydettiğimiz verilerin feature'larını kıyaslamaya geldi. 
* Uzaklık hesabı yapılır. 
* Uzaklık (benzerlik oranımız)  **cosine similarity** ile hesaplanmaktadır. Cosine Similarity bilmeyen arkadaşlar için
[inceleyiniz](http://www.selcukbasak.com/download/TurkceDokumanBenzerligi.pdf) dökümanı temel düzeyde yeterli olacaktır. 
```python
D = computeDistances(fv)
nearest_idx = np.argsort(D[0, :]);
```
işlemi; D dizisinin **Argsort** işlemi ile indeks sıralaması yapılmasını sağlar ve **nearest_idx** dizisi olarak tutarız. Daha sonrasında ;
```python
nearest_ids.append(nearest_idx[1])
```
Birbirine en yakın resimler bastırılacağı için **nearest_idx**'in ikinci indisi bir dizide tutulur.Dizide tutmamızın sebebi
her metodun **featureExtraction**'ı için **nearest_idx**'leri bu dizide tutacağız.Peki neden ikinci indis diye soracak olursanız
resme en yakın resim kendisi olacağından (insert işlemi yaptırmıştık.) ilk indis resmin kendisidir , bize en yakın olan resim 
ikinci indis olacaktır. Her resmin uzaklık değerini de plotlib ile figürde yazdıracağımız için bir dizide tutarız.
```python
        closest_distance1 = D[0][nearest_idx[1]]
        closest_dists.append(closest_distance1)
```        
closest_dists ile her metodun **featureExtraction**'ı için **closest_distance**'ları bu dizide tutacağız.
Gerekli tüm parametreleri elde ettikten sonra artık **save_figs** fonksiyonu ile artık figürü çizdirebiliriz.
**for loop** yapısı incelenecek olursa başta figürü 1 kere oluşturmak için **if i==0** şartı koşulur ve i=0 aşamasında;
```python
    fig = plt.figure()
    for i in range(3):
        if i ==0:
            img1 = mpimg.imread(queryfile)
            a = fig.add_subplot(1, 4, i+1)
            imgplot_1 = plt.imshow(img1)
            a.set_title(titles[0])
```
Başlangıçta figür oluşturulur. Sonrasında **for** loop üç kere dönecektir . 3 farklı featureExtraction modumuz olduğundan ötürü.
Başlangıçta **i ==0** durumunda iken ilk olarak query image'ın path'i verilir ve ekrana bastırılır. Sonrasında **subplot** ile figürün
**(1,4)** boyutta olacağı belirtilir ve **title**'ı verilir. 
 
Yukarıdaki durum sadece başlangıçta çalışacaktır . Tekrarlanacak her adım şu şekilde olacaktır;
```python

img2 = mpimg.imread("images/" + str(nearest_ids[i]) + ".jpg")
a = fig.add_subplot(1, 4, i+2)
plt.xlabel('Distance: ' + str(closest_dists[i]))
imgplot = plt.imshow(img2)
a.set_title(titles[i+1])
```
En yakın resimler featureExtraction metotlarına gelmeye başlayacaktır artık . Bulduğumuz **nearest_ids**'ten tek tek indislerle
path'ler elde edilir ve path'ten resim çekilir.Sonrasında bu resim subplot ile ekleneceği belirtilir , **title** ı verilir.
İşlemler tamamlandığında en son ;
```python
    fig.set_size_inches((12, 12), forward=False)
    plt.savefig("results/mm_model15.png", format="png")
    plt.show()
```
işlemleri uygulanır ve resimlerin size_inches değerleri ayarlanır. **savefig()** ile elde edilen figür kaydedilmiş olur. Sonrasında
**plt.show()** ile figür çizdirilmiş olur.
