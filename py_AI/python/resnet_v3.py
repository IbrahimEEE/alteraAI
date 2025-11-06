# posix_ipc kutuphanesi POSIX semafor islemleri icin kullaniliyor
import posix_ipc
# cv2 modulu goruntu isleme icin kullaniliyor
import cv2
# glob modulu dosya desenleriyle calismak icin kullaniliyor
import glob
# numpy modulu sayisal islemler icin kullaniliyor
import numpy as np
# time modulu gecikme eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu paylasimli bellek eslemeleri icin kullaniliyor
import mmap
# socket modulu UDP iletisimini saglamak icin kullaniliyor
import socket


# UIO donusum yapisinin kurulumu baslatiliyor

# Eslenecek UIO bellek boyutu 0x1000 olarak tanimlaniyor
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

# Ornek goruntu dizini tanimlaniyor
image_directory = '../resnet-50-tf/sample_images/'

# Dizin altindaki tum .bmp dosyalari listeleniyor
bmp_files = glob.glob(image_directory + '/*.bmp')

# Donanima gonderilecek goruntuler icin liste
images = []
# Izleme icin orijinal goruntuler listesi
images2view = []

# Her goruntu dosyasi icin dongu calisiyor
for bmp_file in bmp_files:
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 224x224 boyutuna yeniden boyutlandiriliyor
    im = cv2.resize(im, (224,224), interpolation= cv2.INTER_LINEAR)
    # Izleme amacli orijinal goruntu listeye ekleniyor
    images2view.append(im)
    # Donanim icin BGR -> RGB donusumu yapiliyor
    im = np.flip(im,axis=-1)
    # Lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((224,224,1),dtype=np.uint8)
    # Gerekli formatta goruntu olusturulup listeye ekleniyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))


# Paylasimli bellek dosyasi olusturularak boyut garanti ediliyor
with open('shared.mem', "wb") as f:
        # Dosyanin son byte'ina gidiliyor
        f.seek(63)  # Go to the last byte
        # Gerekli boyutu olusturmak icin bir byte yaziliyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


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


# Paylasimli bellek esleniyor
size = 40
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
# Cikti verisi unsigned integer olarak okunuyor
output_map = np.frombuffer(mem, np.uint32, size >> 2)

# Etiket listesi okunuyor
with open("imagenet-classes.txt") as file:
    lines = [line.rstrip() for line in file]


# Goruntu akisi icin hazirlik yapiliyor

# UDP uzerinden goruntu gondermek icin soket olusturuluyor
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Uzak alici adresi ve portu tanimlaniyor
server_address = ('192.168.0.33', 12345)



# Goruntu sayaci sifirlaniyor
count = 0

while True:


    # Yazilan byte sayisini takip eden degisken sifirlaniyor
    nwritten = 0
    # Goruntu izleme icin seciliyor
    image2view = images2view[count % len(images)]
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Donanima goruntu baytlari yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Gonderilen goruntu indexi ve byte miktari yazdiriliyor
    print(count, nwritten)
    # Sonraki goruntu icin sayac artiriliyor
    count += 1
    
    
    
    # Donanim cikti verirken beklemek icin gecikme ekleniyor
    time.sleep(0.5)
    # Ham cikti vektoru yazdiriliyor
    print(output_map)
    # En olasi sinif etiketi belirleniyor
    label = lines[output_map[0]-1]
    print(label)
    
    # Tahmin etiketi goruntu uzerine ekleniyor
    position = (10, 180)  # Bottom-left corner of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    color = (255, 255, 255)  # White color in BGR
    thickness = 4
    cv2.putText(image2view, label, position, font, font_scale, color, thickness)
    
    # Goruntu JPEG olarak encode ediliyor
    _, encoded_frame = cv2.imencode('.jpeg', image2view)

    # Kodlanmis goruntu UDP uzerinden gonderiliyor
    sock.sendto(encoded_frame, server_address)

# Is tamamlaninca soket kapatiliyor
sock.close()
