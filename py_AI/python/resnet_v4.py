# posix_ipc kutuphanesi POSIX semafor islemleri icin kullaniliyor
import posix_ipc
# cv2 modulu goruntu isleme ve kamera erisimi icin kullaniliyor
import cv2
# glob modulu bu senaryoda kullanilmiyor ama dosya desenleri icin bulunuyor
import glob
# numpy modulu sayisal islemler icin kullaniliyor
import numpy as np
# time modulu gecikme eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu paylasimli bellek eslemeleri icin kullaniliyor
import mmap

# socket modulu UDP iletisimini saglamak icin dahil ediliyor
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


# USB kamerayi aciyoruz
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)


# Paylasimli bellek dosyasi olusturularak boyut garanti ediliyor
with open('shared.mem', "wb") as f:
        # Dosyanin son byte'ina gidiliyor
        f.seek(63)  # Go to the last byte
        # Gerekli boyutu saglamak icin bir byte yaziliyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


# Cikarsama uygulamasina hazir oldugumuzu bildirmek icin semafor kullaniliyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Bilgilendirme mesaji icin bayrak
firstTime = True

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
sem.close()

# Paylasimli bellek esleniyor
size = 40
offset = 0
mmap_file = os.open('shared.mem', os.O_RDWR | os.O_SYNC)
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
os.close(mmap_file)
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

    # Kameradan yeni kare aliniyor
    _,frame = cap.read()
    #print(frame.shape)
    # Kare donanimin bekledigi boyuta yeniden boyutlandiriliyor
    im = cv2.resize(frame, (224,224), interpolation= cv2.INTER_LINEAR)
    # BGR -> RGB donusumu yapiliyor
    im = np.flip(im,axis=-1)
    # Lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((224,224,1),dtype=np.uint8)
    # Donanimin bekledigi formatta goruntu olusturuluyor
    im = np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8)


    # Yazilan byte sayisini takip eden degisken sifirlaniyor
    nwritten = 0
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Donanima goruntu baytlari yaziliyor
        nwritten = f.write(im.tobytes())
    # Gonderilen goruntu indexi ve byte miktari yazdiriliyor
    print(count, nwritten)
    # Sonraki goruntu icin sayac artiriliyor
    count += 1
    
    # Donanim ciktisini beklemek icin kisa gecikme ekleniyor
    time.sleep(0.05)
    # Ham cikti vektoru yazdiriliyor
    print(output_map)
    # En olasi sinif etiketi belirleniyor
    label = lines[output_map[0]-1]
    print(label)
    
    # Tahmin etiketi goruntu uzerine ekleniyor
    position = (10, 400)  # Bottom-left corner of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)  # White color in BGR
    thickness = 5
    cv2.putText(frame, label, position, font, font_scale, color, thickness)
    
    # Goruntu gonderim icin 224x224'e yenileniyor
    frame = cv2.resize(frame, (224,224), interpolation= cv2.INTER_LINEAR)
    # Goruntu JPEG olarak encode ediliyor
    _, encoded_frame = cv2.imencode('.jpeg', frame)

    # Kodlanmis goruntu UDP uzerinden gonderiliyor
    sock.sendto(encoded_frame, server_address)
    # Altyapin ciktisi icin ek gecikme ekleniyor
    time.sleep(0.1)


# Islemler bitince kamera ve soket serbest birakiliyor
cap.release()
sock.close()
