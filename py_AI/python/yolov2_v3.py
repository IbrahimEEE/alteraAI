# posix_ipc kutuphanesi POSIX semafor islemleri icin kullaniliyor
import posix_ipc
# cv2 modulu goruntu okuma ve boyutlandirma icin kullaniliyor
import cv2
# glob modulu desenle eslesen dosyalari bulmak icin kullaniliyor
import glob
# numpy modulu sayisal islemler icin kullaniliyor
import numpy as np
# time modulu gecikme eklemek icin kullaniliyor
import time
# os modulu dosya ve cihaz islemleri icin kullaniliyor
import os
# mmap modulu cihaz bellegi eslemek icin kullaniliyor
import mmap

# socket modulu ag islemleri icin ekleniyor
import socket

# UIO donusum yapisinin kurulumu

# Haritalanacak UIO bellek boyutu 0x1000 olarak seciliyor
size = 0x1000
# Haritalama ofseti sifir olarak ayarlaniyor
offset = 0

# /dev/uio2 cihaz dosyasi okuma-yazma haklariyla aciliyor
mmap_file = os.open('/dev/uio2', os.O_RDWR | os.O_SYNC)
# Cihaz belleği MAP_SHARED ve RW izinleriyle uygulamaya esleniyor
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
# Dosya tanimlayicisi kapatiliyor
os.close(mmap_file)
# Haritalanan alan 32 bit unsigned integer olarak okunuyor
uio_map = np.frombuffer(mem, np.uint32, size >> 2)
# Ayni alan 32 bit float olarak da goruntuleniyor
uio_fp = np.frombuffer(mem, np.float32, size >> 2)

# uio_map[0] biti donanimi tetiklemek icin 1 yapiliyor
uio_map[0] = 1
# Tetikleme impulsu icin kisa bekleme ekleniyor
time.sleep(0.1)
# uio_map[0] sifirlanarak tetikleme tamamlanıyor
uio_map[0] = 0
# uio_map[1] DMA stride degeri olarak 32 sakliyor
uio_map[1] = 32
# uio_map[2] giris goruntu genisligini 416 olarak tutuyor
uio_map[2] = 416
# uio_map[3] giris goruntu yuksekligini 416 olarak tutuyor
uio_map[3] = 416

# uio_fp[16], [17], [18] kanal carpma faktorlerini sakliyor
uio_fp[16] = 1.0
uio_fp[17] = 1.0
uio_fp[18] = 1.0
# uio_fp[32], [33], [34] kanal ortalama cikarma degerlerini sakliyor
uio_fp[32] = -103.94
uio_fp[33] = -116.78
uio_fp[34] = -123.68

# Goruntulerin hazirlanmasi

# Goruntu dosyalarinin bulundugu dizin tanimlaniyor
image_directory = '../car_image/'

# Dizin altindaki tum .jpg dosyalari listeleniyor
bmp_files = glob.glob(image_directory + '/*.jpg')
# Bulunan dosya yolları yazdiriliyor
print(bmp_files)
# Donanima gonderilecek goruntuler icin liste
images = []
# Izleme icin orijinal goruntulerin listesi
images2view = []


# Her goruntu dosyasi icin dongu calisiyor
for bmp_file in bmp_files:
    # Dosya adi yazdiriliyor
    print(bmp_file)
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 416x416 piksele boyutlandiriliyor
    im = cv2.resize(im, (416,416), interpolation= cv2.INTER_LINEAR)
    # Orijinal goruntu izleme listesine ekleniyor
    images2view.append(im)
    # BGR->RGB donusumu icin kanallar ters cevriliyor
    im = np.flip(im,axis=-1)
    # Lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((416,416,1),dtype=np.uint8)
    # Kanallar birlestirilerek donanima gonderilecek goruntu olusturuluyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))



# Paylasimli bellek boyutu 71825*4 bayt olarak ayarlaniyor
size = 71825*4

# shared.mem dosyasi olusturulup boyutu garanti altina aliniyor
with open('shared.mem', "wb") as f:
        # Dosyanin son byte'ina gidiliyor
        f.seek(size)  # Go to the last byte
        # Dosya boyutu olusturuluyor
        f.write(b"\0")  # Write a single byte to ensure the file is the correct size


# Cikarsama uygulamasina hazir oldugumuzu bildirmek icin semafor kullaniliyor

# POSIX semaforu olusturuluyor/aciliyor
sem = posix_ipc.Semaphore("/CoreDLA_ready_for_streaming",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

# Bilgilendirme mesaji icin bayrak
firstTime = True

# Hazirlik semaforu bekleniyor
while True:

    try:
        # Semafor bloklamadan alinmaya calisiliyor
        sem.acquire(0)
    except:
        # Ilk hatada mesaj yaziliyor
        if (firstTime):
          firstTime = False
          print("Waiting for streaming_inference_app to become ready.")
    else:
        # Semafor alindiysa tekrar salinip dongu kiriliyor
        sem.release()
        break


    # Yeniden denemeden once bekleniyor
    time.sleep(0.1)
# Semafor kapatiliyor
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

# Goruntu akisi sayaci sifirlaniyor

count = 0

# YOLOv2 icin kullanilan anchor olcekleri tanimlaniyor
anchors = np.array([[0.57273, 0.677385], 
                    [1.87446, 2.06253], 
                    [3.33843, 5.47434], 
                    [7.88282, 3.52778], 
                    [9.77052, 9.16828]])

# Izgara hucre boyutu 416/13 olarak hesaplaniyor
grid_size = 416/13

# Sigmoid aktivasyon fonksiyonu tanimlaniyor
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Tum goruntuler uzerinde bir dongu calistiriliyor
for i in range(len(images2view)):

    # Ayirici satir basiliyor
    print("\n\r#######################################################################\n\r")

    # Yazilan byte sayisini takip eden degisken sifirlaniyor
    nwritten = 0
    # Goruntu izleme icin seciliyor
    image2view = images2view[count % len(images)]
    # DMA akisi icin cihaz dosyasi aciliyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Goruntu baytlari donanima yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Yazilan goruntu numarasi ve byte miktari yazdiriliyor
    print(count, nwritten)
    # Sonraki goruntu icin sayac artiriliyor
    count += 1

    # Donanim ciktisini beklemek icin kisa gecikme ekleniyor
    time.sleep(0.1)
    
    # Tum anchor ve grid konumlari icin birlesik dongu
    for x in range(5):
        for j in range (13):
            for k in range(13):
                # Guven 0.4'ten yuksekse tespiti raporla
                if output_map[0,x*85+4,j,k] > 0.4:
                    # Guven degeri aliniyor
                    conf    = output_map[0,x*85+4,j,k]
                    # Arac sinifi olasiligi aliniyor
                    car_c   = output_map[0,x*85+7,j,k]
                    # Izgara icindeki x ofseti aliniyor
                    x_pred  = output_map[0,x*85,j,k]
                    # Izgara icindeki y ofseti aliniyor
                    y_pred  = output_map[0,x*85+1,j,k]
                    # Kutu genislik logaritmasi aliniyor
                    t_w     = output_map[0,x*85+2,j,k]
                    # Kutu yukseklik logaritmasi aliniyor
                    t_h     = output_map[0,x*85+3,j,k]
                    #print(x_pred, y_pred, t_w, t_h)
                    # x ofseti sigmoid ile 0-1 araligina sikistiriliyor
                    x_sigm=sigmoid(x_pred)
                    # y ofseti sigmoid ile 0-1 araligina sikistiriliyor
                    y_sigm=sigmoid(y_pred)
                    #print(x_sigm,y_sigm)
                    # Izgara sutunu ve sigmoid sonucu ile mutlak X konumu
                    bx = (k +x_sigm)*grid_size
                    # Izgara satiri ve sigmoid sonucu ile mutlak Y konumu
                    by = (j +y_sigm)*grid_size
                    # Anchor ve exp ile mutlak genislik hesaplanıyor
                    bw = anchors[x,0] * np.exp(t_w) * grid_size
                    # Anchor ve exp ile mutlak yukseklik hesaplanıyor
                    bh = anchors[x,1] * np.exp(t_h) * grid_size

                    # Sonuclari formatlayip yazdiriyoruz
                    txt = "\n\r-> Conf:{0:1.3f} Car:{1:1.3f} X:{2:0.0f} Y:{3:0.0f} W:{4:0.0f} H:{5:0.0f} \n\r"
                    print(txt.format(conf,car_c,bx,by,bw,bh))
                    
    
    # Bir sonraki goruntuye gecmeden once daha uzun bekleniyor
    time.sleep(5)

