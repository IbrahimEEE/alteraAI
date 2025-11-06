# posix_ipc kutuphanesi POSIX semaforlarini kullanmak icin ekleniyor
import posix_ipc
# cv2 modulu goruntu isleme islemleri icin kullaniliyor
import cv2
# glob modulu dosya desenleri ile calismak icin ekleniyor
import glob
# numpy modulu sayisal islemler icin kullaniliyor
import numpy as np
# time modulu gecikme ve zamanlama icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu paylasimli bellek ve cihaz haritalama icin kullaniliyor
import mmap

# socket modulu ag islemleri icin kullaniliyor (bu dosyada kullanilmiyor)
import socket

# UIO donusum yapisi kurulum islemleri baslatiliyor

# Bellek boyutu 0x1000 bayt olarak ayarlaniyor
size = 0x1000
# Bellek ofseti sifir olarak ayarlaniyor
offset = 0

# UIO cihaz dosyasi okuma yazma modunda aciliyor
mmap_file = os.open('/dev/uio2', os.O_RDWR | os.O_SYNC)
# Cihaz belleği MAP_SHARED ile esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor cunku haritalama tamamlandi
os.close(mmap_file)
# Donanim kayitlarina 32 bit unsigned integer olarak erisim saglaniyor
uio_map = np.frombuffer(mem, np.uint32, size >> 2)
# Donanim kayitlarina 32 bit float olarak erisim saglaniyor
uio_fp = np.frombuffer(mem, np.float32, size >> 2)

# uio_map[0] biti donanim baslatmak / resetlemek icin 1 yapiliyor
uio_map[0] = 1
# Kisa bir gecikme uygulanıyor
time.sleep(0.1)
# uio_map[0] sifirlanarak baslatma impulsu tamamlanıyor
uio_map[0] = 0
# uio_map[1] DMA icin satir boyu (stride) olarak 32'ye ayarlaniyor
uio_map[1] = 32
# uio_map[2] goruntunun aktif genisligini 28 olarak sakliyor
uio_map[2] = 28 
# uio_map[3] goruntunun aktif yuksekligini 28 olarak sakliyor
uio_map[3] = 28

# uio_fp[16], [17], [18] kanallara ait carpma faktorlerini tutuyor
uio_fp[16] = 1.0
uio_fp[17] = 1.0
uio_fp[18] = 1.0
# uio_fp[32], [33], [34] kanal bazli ortalama cikarma degerlerini tutuyor
uio_fp[32] = -103.94
uio_fp[33] = -116.78
uio_fp[34] = -123.68

# Goruntuler getiriliyor

# BMP (JPG) goruntulerinin bulundugu dizin tanimlaniyor
image_directory = '../resnet-50-tf/mnimg'

# Dizin altindaki butun JPG dosyalari listeleniyor
bmp_files = glob.glob(image_directory + '/*.JPG')
# Bulunan dosya yolları yazdiriliyor
print(bmp_files)
# Hazirlanacak goruntuler icin liste olusturuluyor
images = []
# Izleme amacli goruntuler icin ikinci liste olusturuluyor
images2view = []

# Goruntu sayaci sifirlanıyor
count = 0

# Her JPG dosyasi icin dongu baslatiliyor
for bmp_file in bmp_files:
    # Dosya yolu ekrana yazdiriliyor
    print(bmp_file)
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 28x28 boyutuna yeniden boyutlandiriliyor
    im = cv2.resize(im, (28,28), interpolation= cv2.INTER_LINEAR)
    # Goruntu izleme listesine ekleniyor
    images2view.append(im)
    # Bas kanal icin sifirlarla dolu matris olusturuluyor
    lead_zeros = np.zeros((28,28,1),dtype=np.uint8)
    # Sifir kanali goruntu ile birlestirilip veriye ekleniyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))



# Paylasimli bellek boyutu 40 bayt olarak belirleniyor
size = 40

# Paylasimli bellek dosyasi olusturuluyor
with open('shared.mem', "wb") as f:
        # Dosyanin son baytina atlanıyor
        f.seek(size)  # Go to the last byte
        # Dosyanin boyutunu garantiye almak icin bos byte yaziliyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


# Cikarsama uygulamasina hazir oldugumuzu bildiriyoruz

# POSIX semaforu olusturulup aciliyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Ilk bekleme icin bayrak ayarlaniyor
firstTime = True

# Semafor hazir olana kadar dongu calisiyor
while True:

    try:
        # Semaforu engellemeden almaya calisiyoruz
        sem.acquire(0)
    except:
        # Ilk denemede mesaj yazdiriliyor
        if (firstTime):
          firstTime = False
          print("Waiting for streaming_inference_app to become ready.")
    else:
        # Semafor alindiysa tekrar brakilarak diger tarafa hazir oldugumuz bildirilir
        sem.release()
        break


    # Yeniden denemeden once kisa bekleme yapiliyor
    time.sleep(0.1)
# Semafor kapatiliyor
sem.close()


# Paylasimli bellek haritalamasi baslatiliyor
offset = 0
# shared.mem dosyasi okuma yazma modunda aciliyor
mmap_file = os.open('shared.mem', os.O_RDWR | os.O_SYNC)
# shared.mem dosyasi bellege esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor
os.close(mmap_file)
# Cikti verileri float32 olarak okunuyor
output_map = np.frombuffer(mem, np.float32, size >> 2)


# Sigmoid aktivasyon fonksiyonu tanimlaniyor
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# output_map potansiyel yeniden sekillendirme ornegi (kullanilmiyor)
#output_map = np.reshape(output_map, (1,425,13,13))

# Goruntu akisi baslatiliyor

# Dongu icin sayac sifirlanıyor
count = 0


# Sonsuz dongu ile surekli akıs saglaniyor
while True:


    # Yazilan byte sayisini tutmak icin degisken sifirlaniyor
    nwritten = 0
    # Gosterim icin guncel goruntu seciliyor
    image2view = images2view[count % len(images)]
    # DMA akıs cihazi dogrudan aciliyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Goruntunun ham byte verisi cihaza yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Kacinci goruntunun yazildigi ve kac byte oldugu yazdiriliyor
    print(count, nwritten)
    # Sonraki goruntu icin sayac artiriliyor
    count += 1
    # Yazma sonrasi donanimin islemesi icin bekleniyor
    time.sleep(1)
    # Paylasimli bellekten cikti dizisi yazdiriliyor
    print((output_map))
    # Cikti dizisindeki maksimum degerin indexi (tahmin) yazdiriliyor
    print(np.argmax(output_map))
    # Sonraki dongu turundan once bekleniyor
    time.sleep(1)
    
