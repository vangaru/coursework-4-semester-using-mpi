from imgpath import ImagePath
from imgproc import *
from datetime import datetime
from copy import deepcopy
from cv2 import *
from mpi4py import MPI

IMG_SIZE = 5000
MASTER_PROCESS = 0

if __name__ == '__main__':
    source_image = imread(ImagePath.SRC.value, IMREAD_GRAYSCALE)

    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    processes_count = communicator.Get_size()

    if rank == MASTER_PROCESS:
        image_with_allocated_borders_mp = deepcopy(source_image)
        pixel_points = {"start": 0, "end": 999}
        data = {"img": image_with_allocated_borders_mp, "points": pixel_points}
        for i in range(1, processes_count):
            communicator.send(data, dest=i, tag=i)
            data["points"]["start"] = data["points"]["end"]
            data["points"]["end"] += 1000
        pixel_points = {"start": 0, "end": 1000}
        timer_start = datetime.now()
        for i in range(1, processes_count):
            allocated_borders_per_process = communicator.recv(source=i, tag=processes_count + i)
            image_with_allocated_borders_mp[:, pixel_points["start"]:pixel_points["end"]] = \
                allocated_borders_per_process[:, pixel_points["start"]:pixel_points["end"]]
            pixel_points["start"] = pixel_points["end"]
            pixel_points["end"] += 1000
        timer_end = datetime.now()
        timer_result = timer_end - timer_start
        print("BORDER ALLOCATION USING MPI")
        print("Source Image Path: {}".format(ImagePath.SRC.value))
        print("Image With Allocated Borders Path: {}".format(ImagePath.ALLOCATED_BORDERS_MP.value))
        print("Numbers Of Processes: {}; 1 Master Process and 5 Worker Processes".format(processes_count))
        print("Time: {}".format(timer_result))
        imwrite(ImagePath.ALLOCATED_BORDERS_MP.value, image_with_allocated_borders_mp)
    else:
        data = communicator.recv(source=MASTER_PROCESS, tag=rank)
        allocated_borders_per_process = data["img"]
        pixel_points = data["points"]
        allocate_borders(source_image, allocated_borders_per_process, pixel_points["start"], pixel_points["end"])
        communicator.send(allocated_borders_per_process, dest=MASTER_PROCESS, tag=processes_count + rank)
