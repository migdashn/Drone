import cv2
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ArucoMarker:
    id: int
    corners: np.ndarray
    center: np.ndarray = None

    def __post_init__(self):
        self.center = self.corners.mean(axis=0)

class ArucoDetector:
    def __init__(self, marker_type: int = cv2.aruco.DICT_6X6_250):
        self.predefined_dict = cv2.aruco.getPredefinedDictionary(marker_type)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.predefined_dict, self.params)

    def detect(self, img: np.ndarray, ) -> [ArucoMarker]:
        detected_markers = []
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bboxs, ids, rejected = self.detector.detectMarkers(im_gray)
        if len(bboxs) == 0:
            return detected_markers
        for box, idx in zip(bboxs, ids):
            detected_markers.append(ArucoMarker(id=int(idx[0]), corners=box[0]))
        return detected_markers

    @staticmethod
    def draw_markers(img: np.ndarray, markers: [ArucoMarker]):
        img = img.copy()
        for marker in markers:
            cv2.aruco.drawDetectedMarkers(img, [np.expand_dims(marker.corners, 0)])
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.xticks([])
        plt.yticks([])
        plt.title('Detected Aruco Markers')
        plt.show()


def test_aruco_detector(im_path):
    detector = ArucoDetector()
    img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    detected_markers = detector.detect(img)
    print(f'Detected {len(detected_markers)} markers')
    detector.draw_markers(img, detected_markers)


if __name__ == '__main__':
    test_aruco_detector('aruco_test_image.jpg')