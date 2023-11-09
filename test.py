from vimos.pose.model import build_model
from vimos.container import Photo, Album, Skeleton


import tqdm
import glob
import time
import cv2

num_images = [1, 10, 100, 1000, 10000, 100000]
images = glob.glob("sample/*.jpeg")

for num_image in num_images:
    start_time = time.time()
    temp = []
    for image in tqdm.tqdm(images[:num_image]):
        temp.append(cv2.imread(image))
    print("OpenCV: ", time.time() - start_time)

    start_time = time.time()
    album = Album(images[:num_image])
    print("Album: ", time.time() - start_time)
