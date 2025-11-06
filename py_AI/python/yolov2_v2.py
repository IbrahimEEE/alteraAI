# posix_ipc kutuphanesi POSIX semaforlarini kullanmak icin dahil ediliyor
import posix_ipc
# cv2 modulu goruntu okuma ve isleme icin kullaniliyor
import cv2
# glob modulu belirli desene uyan dosyalari bulmak icin kullaniliyor
import glob
# numpy modulu sayisal islemler icin kullaniliyor
import numpy as np
# time modulu gecikme eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu paylasimli bellek eslemesi icin kullaniliyor
import mmap

# socket modulu ag iletisimine uygun olacak sekilde ekleniyor
import socket

# UIO donusum yapisinin kurulumu

# Haritalanacak UIO alan boyutu 0x1000 olarak seciliyor
size = 0x1000
# Haritalama ofseti sifir olarak ayarlaniyor
offset = 0

# /dev/uio2 cihaz dosyasi okuma-yazma haklariyla aciliyor
mmap_file = os.open('/dev/uio2', os.O_RDWR | os.O_SYNC)
# Cihaz belleği MAP_SHARED ve RW izinleriyle bellege esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatilip esleme kullanılmaya devam ediyor
os.close(mmap_file)
# Haritalanan bolge 32 bit unsigned integer olarak yorumlanarak register erisimine imkan veriyor
uio_map = np.frombuffer(mem, np.uint32, size >> 2)
# Ayni bolge 32 bit float olarak da inceleniyor
uio_fp = np.frombuffer(mem, np.float32, size >> 2)

# uio_map[0] biti donanimi tetiklemek icin 1 yapiliyor
uio_map[0] = 1
# Tetikleme impulsu icin kisa bekleme yapiliyor
time.sleep(0.1)
# uio_map[0] sifirlanarak tetikleme sonlandiriliyor
uio_map[0] = 0
# uio_map[1] DMA stride degerini 32 olarak tutuyor
uio_map[1] = 32
# uio_map[2] giris goruntusu genisligini 416 olarak sakliyor
uio_map[2] = 416
# uio_map[3] giris goruntusu yuksekligini 416 olarak sakliyor
uio_map[3] = 416

# uio_fp[16], [17], [18] kanal carpma faktorlerini tutuyor
uio_fp[16] = 1.0
uio_fp[17] = 1.0
uio_fp[18] = 1.0
# uio_fp[32], [33], [34] kanal ortalama cikarma degerlerini iceriyor
uio_fp[32] = -103.94
uio_fp[33] = -116.78
uio_fp[34] = -123.68

# Goruntu listesi hazirlaniyor

# Arac goruntulerinin bulundugu dizin tanimlaniyor
image_directory = '../car_image/'

# Dizin altindaki tum .jpg dosyalari glob ile listeleniyor
bmp_files = glob.glob(image_directory + '/*.jpg')
# Bulunan dosya adlari yazdiriliyor
print(bmp_files)
# Donanima gonderilecek goruntuler icin liste
images = []
# Izleme/analiz icin ikinci liste
images2view = []


# Her goruntu dosyasi icin dongu calistiriliyor
for bmp_file in bmp_files:
    # Islenen dosyanin adi yazdiriliyor
    print(bmp_file)
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 416x416 piksele yeniden boyutlandiriliyor
    im = cv2.resize(im, (416,416), interpolation= cv2.INTER_LINEAR)
    # Orijinal goruntu izleme listesine ekleniyor
    images2view.append(im)
    # Donanimin bekledigi renk sirasi icin BGR->RGB donusumu
    im = np.flip(im,axis=-1)
    # Donanimin bekledigi lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((416,416,1),dtype=np.uint8)
    # Kanallar birlestirilerek donanima gonderilecek goruntu olusturuluyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))



# Cikti paylasimli bellek boyutu 71825*4 bayt olarak ayarlaniyor
size = 71825*4

# shared.mem dosyasi olusturulup yeterli boyuta getiriliyor
with open('shared.mem', "wb") as f:
        # Dosyanin son byte'ina gidiliyor
        f.seek(size)  # Go to the last byte
        # Dosya boyutu garantileniyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


# Cikarsama uygulamasina hazirlik sinyali veriliyor

# POSIX semaforu olusturuluyor/aciliyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Mesaj tek sefer basilsin diye bayrak
firstTime = True

# Hazirlik semaforu gelene kadar bekleniyor
while True:

    try:
        # Semafor engellemeden alinmaya calisiliyor
        sem.acquire(0)
    except:
        # Ilk yakalamada bilgi mesaji gosteriliyor
        if (firstTime):
          firstTime = False
          print("Waiting for streaming_inference_app to become ready.")
    else:
        # Semafor basariyla alindiysa tekrar salinip cikiliyor
        sem.release()
        break


    # Beklemeye devam etmeden once kisa uyku
    time.sleep(0.1)
# Islem bitince semafor kapatiliyor
sem.close()


# Paylasimli bellek esleniyor
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

# YOLO ciktilari (1,425,13,13) sekline donusturuluyor
output_map = np.reshape(output_map, (1,425,13,13))

# Goruntu akisi icin sayac baslatiliyor

count = 0


# Tum goruntu listesi uzerinde birer kez dolasma dongusu
for i in range(len(images2view)):

    # Ayirici satir yazdiriliyor
    print("\n\r#######################################################################\n\r")

    # Yazilan byte sayisini izlemek icin degisken sifirlanıyor
    nwritten = 0
    # Guncel goruntu izleme icin seciliyor
    image2view = images2view[count % len(images)]
    # DMA akisi icin cihaz dosyasi aciliyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Donanima goruntu baytlari yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Goruntu indexi ve yazilan byte miktari yazdiriliyor
    print(count, nwritten)
    # Sonraki goruntu icin sayac artiriliyor
    count += 1

    # Donanimin cikti olusturmasi icin bekleniyor
    time.sleep(0.1)
    
    # YOLO ciktilari uzerinde sinif, konum ve guven degerlerini tarayan ic donguler
    for x in range(5):
        # Izgara satiri dongusu
        for j in range (13):
            # Izgara sutunu dongusu
            for k in range(13):
                # Guven degeri 0.4'ten buyukse sonuc raporlanıyor
                if output_map[0,x*85+4,j,k] > 0.4:
                    # Konum ve sinif bilgilerini formatlayip yazdiriyoruz
                    txt = "\n\r-> [{0:2d},{1:2d},{2:2d}] {4} Conf:{3:1.3f}\n\r"
                    print(txt.format(j,k,x,output_map[0,x*85+4,j,k],output_map[0,x*85:x*85+4,j,k]))
                    # Sinif olasiliklari yazdiriliyor
                    print(output_map[0,x*85+5:x*85+85,j,k])
    
    # Sonraki goruntuye gecmeden once daha uzun bekleniyor
    time.sleep(5)

