# posix_ipc kutuphanesi POSIX semafor islemleri icin kullaniliyor
import posix_ipc
# cv2 modulu goruntu okuma ve isleme icin kullaniliyor
import cv2
# glob modulu belirli deseni saglayan dosyalari bulmak icin kullaniliyor
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
# Tetikleme impulsunun islemesi icin kisa bekleme yapiliyor
time.sleep(0.1)
# uio_map[0] sifirlanarak tetikleme tamamlanıyor
uio_map[0] = 0
# uio_map[1] DMA stride degerini 32 olarak sakliyor
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

# Goruntuler hazirlaniyor

# Goruntu dosyalarinin bulundugu dizin tanimlaniyor
image_directory = '../car_image/'

# Dizin altindaki tum .jpg dosyalari listeleniyor
bmp_files = glob.glob(image_directory + '/*.jpg')
# Bulunan dosya yolları yazdiriliyor
print(bmp_files)
# Donanima gonderilecek goruntuler icin liste
images = []
# Izleme icin orijinal goruntuler listesi
images2view = []


# Her goruntu icin dongu calistiriliyor
for bmp_file in bmp_files:
    # Dosya adi yazdiriliyor
    print(bmp_file)
    # Goruntu OpenCV ile okunuyor
    im = cv2.imread(bmp_file)
    # Goruntu 416x416 boyutuna yeniden boyutlandiriliyor
    im = cv2.resize(im, (416,416), interpolation= cv2.INTER_LINEAR)
    # Orijinal goruntu izleme listesine ekleniyor
    images2view.append(im)
    # Donanim icin BGR -> RGB donusumu yapiliyor
    im = np.flip(im,axis=-1)
    # Lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((416,416,1),dtype=np.uint8)
    # Donanimin bekledigi formatta goruntu olusturulup listeye ekleniyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))



# Paylasimli bellek boyutu 71825*4 bayt olarak ayarlaniyor
size = 71825*4

# shared.mem dosyasi olusturularak yeterli alan garanti ediliyor
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

# YOLO ciktilari (1,425,13,13) sekline yeniden sekillendiriliyor
output_map = np.reshape(output_map, (1,425,13,13))

# Goruntu akisi baslatiliyor

# Goruntu sayaci sifirlaniyor
count = 0

# YOLO icin anchor degerleri tanimlaniyor
anchors = np.array([[0.57273, 0.677385], 
                    [1.87446, 2.06253], 
                    [3.33843, 5.47434], 
                    [7.88282, 3.52778], 
                    [9.77052, 9.16828]])

# Izgara hucre boyutu hesaplanıyor
grid_size = 416/13

# Sigmoid aktivasyon fonksiyonu tanimlaniyor
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# UDP uzerinden veri gondermek icin soket olusturuluyor
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Uzak alici IP ve portu tanimlaniyor
server_address = ('192.168.0.33', 12345)

for i in range(len(images2view)):

    # Her kare icin ayirici satir yazdiriliyor
    print("\n\r#######################################################################\n\r")

    # Yazilan byte sayisini takip eden degisken sifirlaniyor
    nwritten = 0
    # Goruntu izleme icin seciliyor
    image2view = images2view[count % len(images)]
    # DMA akisi icin cihaz dosyasi aciliyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Donanima yazilacak goruntu baytlari yaziliyor
        nwritten = f.write(images[count % len(images)].tobytes())
    # Gonderilen goruntu indexi ve byte miktari yazdiriliyor
    print(count, nwritten)
    # Sonraki goruntu icin sayac artiriliyor
    count += 1

    # Donanimin cikti uretmesi icin kisa bekleme ekleniyor
    time.sleep(0.1)
    
    # Tum anchor ve grid konumlari uzerinde geziniliyor
    for x in range(5):
        for j in range (13):
            for k in range(13):
                # Guvenlik skoru esigin uzerindeyse tespit isleniyor
                if output_map[0,x*85+4,j,k] > 0.4:
                    # Guven degeri okunuyor
                    conf    = output_map[0,x*85+4,j,k]
                    # Arac sinifi olasiligi okunuyor
                    car_c   = output_map[0,x*85+7,j,k]
                    # Izgara icindeki x ofseti aliniyor
                    x_pred  = output_map[0,x*85,j,k]
                    # Izgara icindeki y ofseti aliniyor
                    y_pred  = output_map[0,x*85+1,j,k]
                    # Kutu genisligi logaritmasi aliniyor
                    t_w     = output_map[0,x*85+2,j,k]
                    # Kutu yuksekligi logaritmasi aliniyor
                    t_h     = output_map[0,x*85+3,j,k]
                    #print(x_pred, y_pred, t_w, t_h)
                    # x koordinati sigmoid ile 0-1 araligina cekiliyor
                    x_sigm=sigmoid(x_pred)
                    # y koordinati sigmoid ile 0-1 araligina cekiliyor
                    y_sigm=sigmoid(y_pred)
                    #print(x_sigm,y_sigm)
                    # Mutlak merkez X koordinati hesaplanıyor
                    bx = (k +x_sigm)*grid_size
                    # Mutlak merkez Y koordinati hesaplanıyor
                    by = (j +y_sigm)*grid_size
                    # Anchor ve exp ile kutu genisligi elde ediliyor
                    bw = anchors[x,0] * np.exp(t_w) * grid_size
                    # Anchor ve exp ile kutu yuksekligi elde ediliyor
                    bh = anchors[x,1] * np.exp(t_h) * grid_size

                    # Ozet bilgileri formatlayip yazdiriyoruz
                    txt = "\n\r-> Conf:{0:1.3f} Car:{1:1.3f} X:{2:0.0f} Y:{3:0.0f} W:{4:0.0f} H:{5:0.0f} Anchor: {6} \n\r"
                    print(txt.format(conf,car_c,bx,by,bw,bh,x))
                    # Cizilecek kutunun sol-ust koordinati hesaplanıyor
                    top_left_x = int(bx - bw / 2)
                    top_left_y = int(by - bh / 2)
                    # Kutunun sag-alt koordinati hesaplanıyor
                    bottom_right_x = int(bx + bw / 2)
                    bottom_right_y = int(by + bh / 2)

                    # Hesaplanan kutu goruntu uzerine ciziliyor
                    cv2.rectangle(image2view, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255*(conf), 255*(1-conf)), 2)
    
    
    # Guncel goruntu JPEG olarak encode ediliyor
    _, encoded_frame = cv2.imencode('.jpeg', image2view,[cv2.IMWRITE_JPEG_QUALITY, 50])
    # Kodlanmis goruntu UDP uzerinden gonderiliyor
    sock.sendto(encoded_frame, server_address)
    
    # Sonraki goruntuye gecmeden once bekleniyor
    time.sleep(5)
