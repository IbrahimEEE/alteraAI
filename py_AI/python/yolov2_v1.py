# posix_ipc kutuphanesi POSIX semafor islemlerini yapmak icin kullaniliyor
import posix_ipc
# cv2 modulu goruntu okuma ve boyutlandirma icin kullaniliyor
import cv2
# glob modulu dosya desenlerini bulmak icin kullaniliyor
import glob
# numpy modulu sayisal islemler icin tercih ediliyor
import numpy as np
# time modulu gecikme eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu cihaz bellegi eslemek icin kullaniliyor
import mmap

# socket modulu ag iletisimine hazir olmak icin dahil ediliyor
import socket

# UIO donusum yapisi kurulum islemleri baslatiliyor

# UIO aygitindan eslenecek bellek boyutu 0x1000 olarak seciliyor
size = 0x1000
# Haritalama icin baslangic ofseti sifir olarak belirleniyor
offset = 0

# /dev/uio2 cihaz dosyasi okuma-yazma haklariyla aciliyor
mmap_file = os.open('/dev/uio2', os.O_RDWR | os.O_SYNC)
# Cihaz belleği MAP_SHARED ve RW izinleriyle uygulamaya esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatilip sadece haritalama kullaniliyor
os.close(mmap_file)
# Haritalanan bolge 32 bit unsigned integer olarak yorumlanarak register erisimi saglaniyor
uio_map = np.frombuffer(mem, np.uint32, size >> 2)
# Ayni bolge 32 bit float olarak goruntuleniyor
uio_fp = np.frombuffer(mem, np.float32, size >> 2)

# uio_map[0] bitini 1 yaparak donanima tetik sinyali gonderiliyor
uio_map[0] = 1
# Tetikleme impulsunun islenmesi icin kisa bekleme yapiliyor
time.sleep(0.1)
# uio_map[0] sifirlanarak reset/tetikleme tamamlanıyor
uio_map[0] = 0
# uio_map[1] DMA stride degerini 32 olarak sakliyor
uio_map[1] = 32
# uio_map[2] giris goruntusu genisligini 416 olarak tutuyor
uio_map[2] = 416
# uio_map[3] giris goruntusu yuksekligini 416 olarak tutuyor
uio_map[3] = 416

# uio_fp[16], [17], [18] kanal carpma faktorlerini (gain) tutuyor
uio_fp[16] = 1.0
uio_fp[17] = 1.0
uio_fp[18] = 1.0
# uio_fp[32], [33], [34] kanal ortalama cikarma degerlerini tutuyor
uio_fp[32] = -103.94
uio_fp[33] = -116.78
uio_fp[34] = -123.68

# Goruntuler hazirlaniyor

# Trafik orneklerinin bulundugu dizin tanimlaniyor
image_directory = '../car_image/sparse_traffic/'

# Dizindeki tum .jpg dosyalari listeleniyor
bmp_files = glob.glob(image_directory + '/*.jpg')
# Bulunan dosya adlari ekrana yaziliyor
print(bmp_files)
# Donanima gonderilecek goruntuler icin liste
images = []
# Izleme/diagnostik icin ikinci liste
images2view = []


# Tum goruntu dosyalari icin dongu baslatiliyor
for bmp_file in bmp_files:
    # Islenen dosya adi yazdiriliyor
    print(bmp_file)
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 416x416 boyutuna yeniden boyutlandiriliyor
    im = cv2.resize(im, (416,416), interpolation= cv2.INTER_LINEAR)
    # Orijinal goruntu izleme listesine ekleniyor
    images2view.append(im)
    # Donanim RGB bekledigi icin kanal sirasi ters cevrilerek BGR->RGB yapiliyor
    im = np.flip(im,axis=-1)
    # Donanim icin lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((416,416,1),dtype=np.uint8)
    # Lider kanal ve goruntu birlestirilerek cikti listesine ekleniyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))



# Paylasimli bellek boyutu YOLO cikti boyutuna gore 71825*4 bayt olarak ayarlaniyor
size = 71825*4

# shared.mem dosyasi olusturularak boyutu garanti altina aliniyor
with open('shared.mem', "wb") as f:
        # Dosyanin son byte'ina gidiliyor
        f.seek(size)  # Go to the last byte
        # Dosya icine bir byte yazilarak boyut olusturuluyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


# Cikarsama uygulamasina hazir oldugumuzu bildirmek icin semafor kullaniliyor

# POSIX semaforu olusturuluyor veya aciliyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Bilgilendirme mesajini tek sefer yazmak icin bayrak
firstTime = True

# Semafor hazir sinyali verene kadar bekleyen dongu
while True:

    try:
        # Semaforu engellemeden almaya calisiyoruz
        sem.acquire(0)
    except:
        # Ilk yakalamada mesaj yazdiriliyor
        if (firstTime):
          firstTime = False
          print("Waiting for streaming_inference_app to become ready.")
    else:
        # Semafor alindiysa tekrar salinip cikiliyor
        sem.release()
        break


    # Tekrar denemeden once kisa gecikme uygulanıyor
    time.sleep(0.1)
# Semafor islemi tamamlaninca kapatiliyor
sem.close()


# Paylasimli bellek haritalamasi yapiliyor
offset = 0
# shared.mem dosyasi okuma-yazma kipinde aciliyor
mmap_file = os.open('shared.mem', os.O_RDWR | os.O_SYNC)
# Dosya bellege MAP_SHARED olarak esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor
os.close(mmap_file)
# Cikti verisi float32 olarak okunuyor
output_map = np.frombuffer(mem, np.float32, size >> 2)

# Cikti dizisi YOLO beklenen sekline (1,425,13,13) yeniden sekillendiriliyor
output_map = np.reshape(output_map, (1,425,13,13))

# Goruntu akisi baslatiliyor

# Dongu sayaci sifirlanıyor
count = 0

# Sonsuz dongu ile goruntuler akitiliyor
while True:


    # Yazilan byte sayisini tutmak icin degisken sifirlaniyor
    nwritten = 0
    # Izleme icin guncel goruntu seciliyor
    image2view = images2view[count % len(images)]
    # DMA akisi icin cihaz dosyasi aciliyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Donanima gonderilecek goruntu baytlari yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Goruntu sayaci ve yazilan byte sayisi yazdiriliyor
    print(count, nwritten)
    # Sonraki goruntu icin sayac artiriliyor
    count += 1
