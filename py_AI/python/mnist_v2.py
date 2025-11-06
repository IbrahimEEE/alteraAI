# posix_ipc kutuphanesi POSIX semaforlarini kullanabilmek icin ekleniyor
import posix_ipc
# cv2 modulu goruntu almak ve boyutlandirmak icin kullaniliyor
import cv2
# glob modulu dosya desenleriyle calismak icin dahil ediliyor
import glob
# numpy modulu sayisal islemler icin kullaniliyor
import numpy as np
# time modulu gecikme eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu paylasimli cihaz bellegi eslemek icin kullaniliyor
import mmap

# socket modulu UDP soketleri olusturmak icin kullaniliyor
import socket

# UIO donusum yapisinin kurulumu baslatiliyor

# UIO cihazindan okunacak/yazilacak bellek boyutu 0x1000 olarak belirleniyor
size = 0x1000
# Haritalama ofseti sifira ayarlaniyor
offset = 0

# /dev/uio2 cihazi okuma-yazma ve senkron kipinde aciliyor
mmap_file = os.open('/dev/uio2', os.O_RDWR | os.O_SYNC)
# Donanim bellegi MAP_SHARED ve RW izinleriyle esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Acilan dosya tanimlayicisi kapatiliyor
os.close(mmap_file)
# Haritalanan bolge 32 bit unsigned integer olarak yorumlanarak register erisiliyor
uio_map = np.frombuffer(mem, np.uint32, size >> 2)
# Ayni bellek bolgesi 32 bit float olarak da goruntuluyor
uio_fp = np.frombuffer(mem, np.float32, size >> 2)

# uio_map[0] biti donanimi tetiklemek icin 1 yapiliyor
uio_map[0] = 1
# Tetikleme impulsunu tamamlamak icin bekleniyor
time.sleep(0.1)
# uio_map[0] sifirlanarak donanim reseti tamamlanıyor
uio_map[0] = 0
# uio_map[1] DMA stride degerini 32 olarak sakliyor
uio_map[1] = 32
# DMA cerceve genisligi uio_map[2] kaydina 28 olarak yaziliyor
uio_map[2] = 28 
# DMA cerceve yuksekligi uio_map[3] kaydina 28 olarak yaziliyor
uio_map[3] = 28

# uio_fp[16], [17], [18] kanallar icin carpma faktorlerini tutuyor
uio_fp[16] = 1.0
uio_fp[17] = 1.0
uio_fp[18] = 1.0
# uio_fp[32], [33], [34] kanal bazli ortalama cikarma degerlerini tutuyor
uio_fp[32] = -103.94
uio_fp[33] = -116.78
uio_fp[34] = -123.68


# Paylasimli bellek boyutu 40 bayt olarak ayarlaniyor
size = 40

# shared.mem dosyasi olusturularak gerekli boyut garantileniyor
with open('shared.mem', "wb") as f:
        # Son byte'a atlanip dosya boyutu olusturuluyor
        f.seek(size)  # Go to the last byte
        # Dosyanin gerekli buyuklukte olmasi icin bir byte yaziliyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


# Cikarsama uygulamasinin akisa hazir oldugunu bildirmek icin semafor kullaniliyor

# POSIX semaforu olusturuluyor veya aciliyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Bekleme mesajinin tek sefer yazilmasi icin bayrak
firstTime = True

# Semafor hazir sinyali verene kadar bekleyen dongu
while True:

    try:
        # Semafor engellemeden alinmaya calisiliyor
        sem.acquire(0)
    except:
        # Ilk hata durumunda bilgilendirici mesaj basiliyor
        if (firstTime):
          firstTime = False
          print("Waiting for streaming_inference_app to become ready.")
    else:
        # Semafor alinabildiyse tekrar brakilip cikiliyor
        sem.release()
        break


    # Tekrar denemeden once kisa bir uyku veriliyor
    time.sleep(0.1)
# Bekleme islemi bittiginde semafor kapatiliyor
sem.close()


# Paylasimli bellek haritalamasi yapiliyor
offset = 0
# shared.mem dosyasi okuma-yazma kipinde aciliyor
mmap_file = os.open('shared.mem', os.O_RDWR | os.O_SYNC)
# Dosya bellege esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor
os.close(mmap_file)
# Cikti verisi float32 olarak okunuyor
output_map = np.frombuffer(mem, np.float32, size >> 2)

# Goruntu isleme dongusu icin sayac sifirlaniyor
count = 0

# Gerekirse kullanmak uzere bir UDP soketi olusturuluyor (gonderim icin)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Uzak alicinin adresi ve portu tanimlaniyor
server_address = ('192.168.0.33', 12345)

# UDP uzerinden veri almak icin soket olusturuluyor
rxsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Yerel IP ve port baglanarak dinlemeye hazirlaniliyor
local_address = ('192.168.0.180', 12345)
rxsock.bind(local_address)

# Sonsuz dongu ile gelen kareler isleniyor
while True:

    # Her kare basinda ayirici satir yazdiriliyor
    print("\n\r#######################################################################\n\r")


    # UDP soketinden bir kare alinip ham byte verisi elde ediliyor
    frame, _ = rxsock.recvfrom(65536)
    # Gelen byte dizisi 28x28x3 uint8 goruntuye donusturuluyor
    frame = np.frombuffer(frame, dtype=np.uint8).reshape(28,28,3)
    # Goruntu izleme icin saklaniyor
    image2view = frame
    # Donanima yazilacak goruntu degiskenine atanıyor
    im = frame
    # Goruntunun boyut bilgisi yazdiriliyor
    print(im.shape)
    # Donanimin bekledigi lider kanal icin sifir matris olusturuluyor
    lead_zeros = np.zeros((28,28,1),dtype=np.uint8)
    # Lider kanal ile goruntu birlestirilip donanima uygun hale getiriliyor
    im = np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8)

    # Yazilacak byte sayisini izlemek icin degisken sifirlaniyor
    nwritten = 0
    # DMA akisi icin cihaz dosyasi aciliyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Donanima goruntu baytlari yaziliyor
        nwritten = f.write(im.tobytes())
    # Kacinci kare ve kac byte yazildigi yazdiriliyor
    print(count, nwritten)
    # Sonraki kare icin sayac artiriliyor
    count += 1
    # Donanim ciktisini beklemek icin kisa sure uyunuyor
    time.sleep(0.1)
    # Paylasimli bellekten okunan cikti degeri yazdiriliyor
    print((output_map))
    # Cikti dizisindeki en yuksek olasilikli sinif indeksini yazdiriliyor
    print(np.argmax(output_map))
    # Bir sonraki kareye gecmeden once tekrar bekleniyor
    time.sleep(0.1)
    
