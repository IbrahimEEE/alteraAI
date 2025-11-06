# posix_ipc kutuphanesi POSIX semaforlarini kullanmak icin dahil ediliyor
import posix_ipc
# cv2 modulu goruntuleri okumak ve boyutlandirmak icin kullaniliyor
import cv2
# glob modulu dosya deseni ile listeleme yapmak icin kullaniliyor
import glob
# numpy modulu sayisal islemler icin kullaniliyor
import numpy as np
# time modulu gecikme eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu paylasimli bellek eslemeleri icin kullaniliyor
import mmap


# UIO donusum yapisinin kurulumu baslatiliyor

# Eslenen UIO bellek boyutu 0x1000 olarak belirleniyor
size = 0x1000
# Haritalama ofseti sifir olarak ayarlaniyor
offset = 0

# /dev/uio2 cihaz dosyasi okuma-yazma modunda aciliyor
mmap_file = os.open('/dev/uio2', os.O_RDWR | os.O_SYNC)
# Cihaz belleği MAP_SHARED ve RW izinleriyle esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor
os.close(mmap_file)
# Haritalanan alan 32 bit unsigned integer olarak okunuyor
uio_map = np.frombuffer(mem, np.uint32, size >> 2)
# Ayni alan 32 bit float olarak yorumlaniyor
uio_fp = np.frombuffer(mem, np.float32, size >> 2)

# uio_map[0] biti donanimi tetiklemek icin 1 yapiliyor
uio_map[0] = 1
# Tetikleme impulsu icin kisa bekleme ekleniyor
time.sleep(0.1)
# uio_map[0] sifirlanarak tetikleme tamamlanıyor
uio_map[0] = 0
# uio_map[1] DMA stride degerini 32 olarak sakliyor
uio_map[1] = 32
# uio_map[2] giris goruntu genisligini 224 olarak tutuyor
uio_map[2] = 224
# uio_map[3] giris goruntu yuksekligini 224 olarak tutuyor
uio_map[3] = 224

# uio_fp[16], [17], [18] kanal carpma faktorlerini sakliyor
uio_fp[16] = 1.0
uio_fp[17] = 1.0
uio_fp[18] = 1.0
# uio_fp[32], [33], [34] kanal ortalama cikarma degerlerini sakliyor
uio_fp[32] = -103.94
uio_fp[33] = -116.78
uio_fp[34] = -123.68

# Goruntuler hazirlaniyor

# Ornek goruntu dosyalarinin bulundugu dizin tanimlaniyor
image_directory = '../resnet-50-tf/sample_images/'

# Dizin altindaki tum .bmp dosyalari listeleniyor
bmp_files = glob.glob(image_directory + '/*.bmp')

# Donanima gonderilecek goruntuler icin liste baslatiliyor
images = []

# Her goruntu dosyasi icin dongu calistiriliyor
for bmp_file in bmp_files:
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 224x224 boyutuna yeniden boyutlandiriliyor
    im = cv2.resize(im, (224,224), interpolation= cv2.INTER_LINEAR)
    # BGR -> RGB donusumu icin kanallar ters cevriliyor
    im = np.flip(im,axis=-1)
    # Lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((224,224,1),dtype=np.uint8)
    # Kanallar birlestirilip goruntu listeye ekleniyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))

# Cikarsama uygulamasina hazir oldugumuzu bildirmek icin semafor kullaniliyor

# POSIX semaforu olusturuluyor/aciliyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Bilgilendirme mesaji icin bayrak
firstTime = True

# Hazirlik semaforu gelene kadar bekleniyor
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
        # Semafor alindiysa tekrar brakilip cikiliyor
        sem.release()
        break


    # Tekrar denemeden once kisa bekleme ekleniyor
    time.sleep(0.1)
# Bekleme tamamlaninca semafor kapatiliyor
sem.close()

# Goruntu akisi baslatiliyor

# Donanima gonderilen goruntu sayaci sifirlaniyor
count = 0

while True:


    # Yazilan byte sayisini takip eden degisken sifirlaniyor
    nwritten = 0
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Goruntu baytlari donanima yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Hangi goruntunun gonderildigi ve kac byte yazildigi yazdiriliyor
    print(count, nwritten)
    # Bir sonraki goruntu icin sayac artiriliyor
    count += 1
    
    # Donanim cikti uretirken beklemek icin gecikme ekleniyor
    time.sleep(0.5)
