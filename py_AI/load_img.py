# posix_ipc kutuphanesi POSIX semaforlarina erismek icin kullaniliyor
import posix_ipc
# cv2 modulu goruntu okumak ve boyutlandirmak icin dahil ediliyor
import cv2
# glob modulu belirli desendeki dosyalari listelemek icin kullaniliyor
import glob
# numpy modulu sayisal islemler ve diziler icin kullaniliyor
import numpy as np
# time modulu gecikmeler eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu cihaz belleligini eslemek icin kullaniliyor
import mmap

# UIO aygiti icin toplam haritalanacak alan 0x1000 bayt olarak belirleniyor
size = 0x1000
# UIO belleği haritalarken ofset sifir olarak seciliyor
offset = 0

# /dev/uio2 cihaz dosyasi okuma-yazma ve senkron kipinde aciliyor
mmap_file = os.open('/dev/uio2', os.O_RDWR | os.O_SYNC)
# mmap ile cihaz belleği MAP_SHARED ve RW izinleriyle esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor; haritalama acik kalmaya devam ediyor
os.close(mmap_file)
# Haritalanan bellek 32 bit unsigned integer olarak yorumlanarak register erisimi saglaniyor
uio_map = np.frombuffer(mem, np.uint32, size >> 2)

# uio_map[0] biti donanimi tetiklemek/resetlemek icin 1 yapiliyor
uio_map[0] = 1
# Kisa bir bekleme ile tetikleme impulsunun algilanmasi saglaniyor
time.sleep(0.1)
# uio_map[0] sifirlanarak tetikleme tamamlanıyor
uio_map[0] = 0
# uio_map[1] DMA icin satir uzunlugunu (stride) 32 olarak tutuyor
uio_map[1] = 32
# uio_map[2] giris goruntusunun genisligini 224 olarak sakliyor
uio_map[2] = 224
# uio_map[3] giris goruntusunun yuksekligini 224 olarak sakliyor
uio_map[3] = 224
# uio_map[16] R kanali icin carpma faktorunu IEEE754 float karsiligi olarak tutuyor
uio_map[16] = 0x3F800000
# uio_map[17] G kanali carpma faktorunu IEEE754 formatinda sakliyor
uio_map[17] = 0x3F800000
# uio_map[18] B kanali carpma faktorunu IEEE754 formatinda sakliyor
uio_map[18] = 0x3F800000
# uio_map[32] R kanali ortalama cikarma degerini IEEE754 float olarak tutuyor
uio_map[32] = 0xC2D1EB86
# uio_map[33] G kanali ortalama cikarma degerini IEEE754 float olarak tutuyor
uio_map[33] = 0xC2E9D1EB
# uio_map[34] B kanali ortalama cikarma degerini IEEE754 float olarak tutuyor
uio_map[34] = 0xC2F7AE28

# shared.mem dosyasi olusturularak paylasimli bellek icin yer aciliyor
with open('shared.mem', "wb") as f:
        # Dosyanin son byte'ina gidilip boyut garanti altina aliniyor
        f.seek(63)  # Go to the last byte
        # Bir byte yazilarak dosya boyutu 64 bayta tamamlanıyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


# ornek BMP goruntulerinin yer aldigi dizin tanimlaniyor
image_directory = '../resnet-50-tf/sample_images/'

# Dizindeki tum .bmp dosyalari glob ile bulunuyor
bmp_files = glob.glob(image_directory + '/*.bmp')

# Islenecek goruntuler icin bos liste olusturuluyor
images = []

# Her BMP dosyasi icin dongu baslatiliyor
for bmp_file in bmp_files:
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 224x224 boyutuna yeniden boyutlandiriliyor
    im = cv2.resize(im, (224,224), interpolation= cv2.INTER_LINEAR)
    # BGR > RGB donusumu icin kanallar ters siralanıyor
    im = np.flip(im,axis=-1)
    # Donanimin bekledigi bos lider kanal icin sifir matrisi olusturuluyor
    lead_zeros = np.zeros((224,224,1),dtype=np.uint8)
    # Lider kanal ve goruntu birlestirilip listeye ekleniyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))

# POSIX semaforu olusturularak streaming uygulamasi ile senkronizasyon saglaniyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Ilk bilgi mesaji icin bayrak tanimlaniyor
firstTime = True

# Semafor hazir sinyalini verene kadar bekleyen sonsuz dongu
while True:

    try:
        # Semaforu engellemeden almaya calisiyoruz
        sem.acquire(0)
    except:
        # Sadece ilk hatada bilgi mesaji veriliyor
        if (firstTime):
          firstTime = False
          print("Waiting for streaming_inference_app to become ready.")
    else:
        # Semafor basariyla alindiysa tekrar salinip dongu kiriliyor
        sem.release()
        break


    # Tekrar denemeden once kisa bekleme yapiliyor
    time.sleep(0.1)
# Islem bittiginde semafor kapatiliyor
sem.close()

# Cikti paylasimli belleginin boyutu 40 bayt olarak ayarlaniyor
size = 40
# Paylasimli bellek ofseti sifirlanıyor
offset = 0
# shared.mem dosyasi okuma-yazma kipinde aciliyor
mmap_file = os.open('shared.mem', os.O_RDWR | os.O_SYNC)
# Dosya bellege MAP_SHARED ile esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor
os.close(mmap_file)
# Cikti verisi 32 bit unsigned integer dizisi olarak okunuyor
output_map = np.frombuffer(mem, np.uint32, size >> 2)

# Gonderilecek goruntuler icin sayac sifirlaniyor
count = 0

# Sonsuz dongu ile goruntu akisi saglaniyor
while True:


    # Yazilan byte sayisini takip etmek icin degisken sifirlaniyor
    nwritten = 0
    # DMA akisi icin /dev/msgdma_stream0 dogrudan aciliyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Hazirlanan goruntu dogrudan byte dizisi olarak yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Kacinci goruntu ve kac byte yazildigi yazdiriliyor
    print(count, nwritten)
    # Bir sonraki goruntu icin sayac artiriliyor
    count += 1
    
    # Donanim ciktisini gormek icin kisa bekleme ekleniyor
    time.sleep(0.5)
    # Paylasimli bellekten okunan output degeri yazdiriliyor
    print(output_map)
    # Yeni dongu turunden once tekrar bekleniyor
    time.sleep(0.5)
