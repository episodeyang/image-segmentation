import cv2
import numpy as np
from tqdm import tqdm


def validate_files(file_paths):
    for p in tqdm(file_paths):
        image=cv2.imread(p, -1)
        if type(image) is not np.ndarray:
            print(p)


