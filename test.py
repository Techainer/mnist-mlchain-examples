import glob
import time

import cv2
from mlchain.client import Client
from mlchain.workflows import Parallel, Task
from PIL import Image
from tqdm import tqdm

model = Client(api_address='127.0.0.1:9001').model()

all_samples = glob.glob('data/*.jpg')*10


def predict_single_image(sample):
    image = cv2.imread(sample)
    return model.predict(image=image)


# Sequential
start_time = time.time()
for sample in tqdm(all_samples):
    res = predict_single_image(sample)
print('Sequentail prediction tooks:', time.time() - start_time)

# Parallel
start_time = time.time()
tasks = [Task(predict_single_image, sample) for sample in all_samples]
res = Parallel(tasks, max_threads=4).run(progress_bar=True)
print('Parallel prediction tooks:', time.time() - start_time)
