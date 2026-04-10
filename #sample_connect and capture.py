# sample_connect_and_capture.py

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import *


class camera(object):
    def __init__(self):
        self.camera = Camera()
        self.frame3d = Frame3D()

    def camera_connect(self):
        print("Discovering all available cameras...")
        camera_infos = Camera.discover_cameras()

        if not camera_infos:
            print("No cameras found. Please check the connection.")
            return False

        # Display the information of all available cameras.
        for i in range(len(camera_infos)):
            print("Camera index :", i)
            print_camera_info(camera_infos[i])

        print("Please enter the index of the camera that you want to connect: ")
        input_index = 0

        # Enter the index of the camera to be connected and check if the index is valid.
        while True:
            input_index = input()
            if input_index.isdigit() and 0 <= int(input_index) < len(camera_infos):
                input_index = int(input_index)
                break
            print("Input invalid! Please enter the index of the camera that you want to connect: ")

        error_status = self.camera.connect(camera_infos[input_index])
        if not error_status.is_ok():
            show_error(error_status)
            return False

        print("Connected to the camera successfully.")
        return True

    def capture_point_cloud(self, point_cloud_file="PointCloud.ply"):
        # 1) 3D 프레임 캡처
        print("Capturing 3D frame...")
        error_status = self.camera.capture_3d(self.frame3d)
        if not error_status.is_ok():
            show_error(error_status)
            return False

        # 2) PLY로 저장
        print(f"Saving point cloud to {point_cloud_file} ...")
        self.frame3d.save_untextured_point_cloud(FileFormat_PLY, point_cloud_file)
        print("Point cloud saved successfully.")
        return True

    def main(self):
        # 예외가 나도 disconnect는 최대한 보장
        try:
            if not self.camera_connect():
                return
            self.capture_point_cloud()
        finally:
            print("Disconnecting camera...")
            self.camera.disconnect()
            print("Done.")


if __name__ == "__main__":
    a = camera()
    a.main()