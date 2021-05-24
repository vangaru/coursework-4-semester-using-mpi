from imgpath import ImagePath
from imgproc import *
from datetime import datetime
import numpy as np
from cv2 import *
from mpi4py import MPI

IMG_SIZE = 5000
SCALE_SIZE = 2
MASTER_PROCESS = 0

if __name__ == '__main__':
    source_image = imread(ImagePath.SRC.value, IMREAD_GRAYSCALE)

    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    processes_count = communicator.Get_size()

    if rank == MASTER_PROCESS:
        scaled_image_size = get_scaled_image_size(source_image, SCALE_SIZE)
        scaled_image_mp = np.zeros([scaled_image_size["height"], scaled_image_size["width"]])
        pixel_points = {"start": 0, "end": 2000}
        data = {"img": scaled_image_mp, "points": pixel_points}
        for i in range(1, processes_count):
            communicator.send(data, dest=i, tag=i)
            data["points"]["start"] = data["points"]["end"]
            data["points"]["end"] += 2000
        pixel_points = {"start": 0, "end": 2000}
        timer_start = datetime.now()
        for i in range(1, processes_count):
            scaled_per_process = communicator.recv(source=i, tag=processes_count + i)
            scaled_image_mp[:, pixel_points["start"]:pixel_points["end"]] = \
                scaled_per_process[:, pixel_points["start"]:pixel_points["end"]]
            pixel_points["start"] = pixel_points["end"]
            pixel_points["end"] += 2000
        timer_end = datetime.now()
        timer_result = timer_end - timer_start
        print("SCALING IMAGES USING MPI")
        print("Source Image Path: {}".format(ImagePath.SRC.value))
        print("Scaled Image Path: {}".format(ImagePath.SCALED_MP.value))
        print("Numbers Of Processes: {}; 1 Master Process and 5 Worker Processes".format(processes_count))
        print("Time: {}".format(timer_result))
        imwrite(ImagePath.SCALED_MP.value, scaled_image_mp)
    else:
        data = communicator.recv(source=MASTER_PROCESS, tag=rank)
        scaled_per_process = data["img"]
        pixel_points = data["points"]
        scale_image(source_image, scaled_per_process, SCALE_SIZE, pixel_points["start"], pixel_points["end"])
        communicator.send(scaled_per_process, dest=MASTER_PROCESS, tag=processes_count + rank)
