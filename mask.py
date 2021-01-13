from PIL import Image
import face_recognition
import numpy as np
import os


def create(image_path):
    Mask(image_path).process()


class Mask:
    MASK_PATH = "MASK.png"

    def __init__(self, image_path):
        self.image_path = image_path
        self.face_img = None
        self.mask_img = None

    def process(self):
        image = face_recognition.load_image_file(self.image_path)
        face_locations = face_recognition.face_locations(image)
        face_landmarks = face_recognition.face_landmarks(image, face_locations)
        self.face_img = Image.fromarray(image)
        self.mask_img = Image.open(Mask.MASK_PATH)
        for face_landmark in face_landmarks:
            self.mask_implementation(face_landmark)
            self.save()

    def mask_implementation(self, face_landmark: dict):
        nose_bridge = face_landmark["nose_bridge"]
        nose_len = len(nose_bridge)
        nose_point = nose_bridge[nose_len * 1 // 4]
        nose_vector = np.array(nose_point)

        chin = face_landmark["chin"]
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_vector = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        mask_wight = self.mask_img.width
        mask_height = self.mask_img.height
        wight_ratio = 1.2
        new_height = int(np.linalg.norm(nose_vector - chin_bottom_vector))

        left_mask_img = self.mask_img.crop((0, 0, mask_wight // 2, mask_height))
        left_mask_width = self.get_distance(chin_left_point, nose_point, chin_bottom_point)
        left_mask_width = int(left_mask_width * wight_ratio)
        left_mask_img = left_mask_img.resize((left_mask_width, new_height))

        right_mask_img = self.mask_img.crop((mask_wight // 2, 0, mask_wight, mask_height))
        right_mask_width = self.get_distance(chin_left_point, nose_point, chin_bottom_point)
        right_mask_width = int(right_mask_width * wight_ratio)
        right_mask_img = right_mask_img.resize((right_mask_width, new_height))

        size = (left_mask_img.width + right_mask_img.width, new_height)
        _mask_img = Image.new("RGBA", size)
        _mask_img.paste(left_mask_img, (0, 0), left_mask_img)
        _mask_img.paste(right_mask_img, (right_mask_img.width, 0), right_mask_img)

        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotate_mask = _mask_img.rotate(angle, expand=True)

        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = _mask_img.width // 2 - left_mask_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotate_mask.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotate_mask.height // 2

        self.face_img.paste(_mask_img, (box_x, box_y), _mask_img)

    def get_distance(self, point1, point2, point3):
        distance = np.abs((point3[1] - point2[1]) * point1[0] +
                          (point2[0] - point3[0]) * point1[1] +
                          (point3[0] - point2[0]) * point2[1] +
                          (point2[1] - point3[1]) * point2[0]) / \
                   np.sqrt((point3[1] - point2[1]) ** 2 +
                           (point2[0] - point3[0]) ** 2)

        return distance

    def save(self):
        name_split = os.path.splitext(self.image_path)
        name = name_split[0] + "__with_mask" + name_split[1]
        self.face_img.save(name)
