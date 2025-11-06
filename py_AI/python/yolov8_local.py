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


# UIO donusum yapisinin kurulumu baslatiliyor

# Haritalanacak UIO bellek boyutu 0x1000 olarak tanimlaniyor
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
# uio_fp[32], [33], [34] kanal offset degerlerini sakliyor
uio_fp[32] = 0
uio_fp[33] = 0
uio_fp[34] = 0

# Yerel test goruntulerini hazirliyoruz

# Ornek goruntu dizini tanimlaniyor
image_directory = '../yolov8n_test/'

# Dizin altindaki tum .jpg dosyalari listeleniyor
bmp_files = glob.glob(image_directory + '/*.jpg')

# Donanima gonderilecek goruntuler icin liste
images = []

# Goruntu orani korunarak pad eklemek icin letterbox fonksiyonu
def letterbox(image, target_size=(640, 640), color=(114, 114, 114)):
    # Giris goruntusunun orijinal boyutlari
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Orani koruyacak olcek faktoru hesaplaniyor
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Goruntu orani bozulmadan yeniden boyutlandiriliyor
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Hedef boyutta pad rengiyle dolu tuval olusturuluyor
    padded_image = np.full((target_height, target_width, 3), color, dtype=np.uint8)

    # Ortalamak icin gerekli ofsetler hesaplanıyor
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Yeniden boyutlanan goruntu pad uzerine yerlestiriliyor
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image


for bmp_file in bmp_files:
    # Goruntu dosyasi okunuyor
    im = cv2.imread(bmp_file)
    # Letterbox ile 224x224 hedef boyuta getiriliyor
    im = letterbox(im, target_size=(224,224))
    # BGR -> RGB donusumu yapiliyor
    im = np.flip(im,axis=-1)
    # Lider sifir kanali olusturuluyor
    lead_zeros = np.zeros((224,224,1),dtype=np.uint8)
    # Donanimin bekledigi formatta goruntu hazirlaniyor
    images.append(np.concatenate((lead_zeros,im),axis=-1,dtype=np.uint8))


# Paylasimli bellek dosyasi olusturularak 4000 bayt rezerve ediliyor
with open('shared.mem', "wb") as f:
        # Dosyanin son byte'ina gidiliyor
        f.seek(4000)  # Go to the last byte
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


    # Tekrar denemeden once kisa bekleme yapiliyor
    time.sleep(0.1)
sem.close()


# Paylasimli bellek esleniyor
size = 4000
offset = 0
mmap_file = os.open('shared.mem', os.O_RDWR | os.O_SYNC)
mem = mmap.mmap(mmap_file, size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=offset)
os.close(mmap_file)
output_map = np.frombuffer(mem, np.float32, size >> 2)


# Imagenet sinif adlari okunuyor
with open("imagenet-classes.txt") as file:
    lines = [line.rstrip() for line in file]

# Goruntu akisi icin sayac ve senkron semaforlar hazirlaniyor

count = 0


infer_semaphore = posix_ipc.Semaphore("/CoreDLA_infer",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);
post_semaphore = posix_ipc.Semaphore("/CoreDLA_post",flags=posix_ipc.O_CREAT, mode=0o644, initial_value=0);

for image in images:

    #timer_semaphore.release()
    # Yazilan byte sayisini takip eden degisken sifirlaniyor
    nwritten = 0
    
    start_time = time.perf_counter()
    # DMA yazim baslangic zamani aliniyor
    with open("/dev/msgdma_stream0", "wb+", buffering=0) as f:
        # Goruntu baytlari donanima yaziliyor
        nwritten = f.write(image.tobytes())
    
    dma_time = time.perf_counter()
    # Donanim DMA aktarimini tamamlayinca infer semaforu bekleniyor
    infer_semaphore.acquire()
    infer_time = time.perf_counter()
    
    # Post-process bitene kadar bekleniyor
    post_semaphore.acquire()
    # End time
    end_time = time.perf_counter()
    # Calculate elapsed time
    elapsed_time = dma_time - start_time
    print(f"DMA execution time: {elapsed_time:.6f} seconds")
    elapsed_time = infer_time - dma_time
    print(f"Inference execution time: {elapsed_time:.6f} seconds")
    elapsed_time = end_time - infer_time
    print(f"Post Process execution time: {elapsed_time:.6f} seconds")

    
    #print(count, nwritten)
    count += 1
    
    #print(output_map[0:1000])
    top_5_indices = np.argsort(output_map)[-5:][::-1]
    
    print(top_5_indices)
    for i in top_5_indices:
        print(output_map[i],lines[i])
