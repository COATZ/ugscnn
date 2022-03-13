import os
import sys
import shutil
import cv2

path_dataset = "/media/cartizzu/DATA/DATASETS/2D-3D-Semantics/"
path_dataset_new = "/media/cartizzu/DATA/LIN/2_CODE/4_SEGMENTATION/ugscnn/baseline/exp3_2d3ds/data2_sphe/"

mode = "pano"

for area in ["area_1","area_2","area_3","area_4","area_5a","area_5b","area_6"]:
    for feat in ["rgb","depth","semantic"]:
        path_dataset_local = os.path.join(path_dataset,str(area),str(mode),feat) # "data" for perspective, "pano" for omni
        path_dataset_new_local = os.path.join(path_dataset_new,str(area),feat)
        print("Writing to ",path_dataset_new_local)
        os.makedirs(os.path.join(path_dataset_new,str(area)), exist_ok=True)
        os.makedirs(path_dataset_new_local, exist_ok=True)
        #print(len(os.listdir(path_dataset_local)))
        for file in [f for f in os.listdir(path_dataset_local) if (os.path.isfile(os.path.join(path_dataset_local, f)) and f.endswith('.png'))]:
            if mode == "pano":
                img = cv2.imread(os.path.join(path_dataset_local,file), cv2.IMREAD_UNCHANGED)
                dim = (540, 540)
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                cv2.imwrite(os.path.join(path_dataset_new_local,file), resized)
            else:
                idx = int(file.split('frame_')[-1].split('_')[0])
                if idx <= 0: #idx == 0
                    img = cv2.imread(os.path.join(path_dataset_local,file), cv2.IMREAD_UNCHANGED)
                    dim = (540, 540)
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(path_dataset_new_local,file), resized)

