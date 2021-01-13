 import os
 from mask import create
# import shutil

 folder = ""  # path to you dataset directory
 images = [os.path.join(folder, image) for image in os.listdir(folder)]
 for i in range(len(images)):
     create(images[i])
#
# for image in os.listdir(folder):
#     if "__with_mask" in image:
#         shutil.move("part2/{}".format(image), 'D:/ML_pr/Face_detector/Dlib/face_with_mask')
#
# new_images = [os.path.join("dataset//face_with_out_mask", image) for image in
#               os.listdir("test//face_with_mask")[0:900]]
#
# for new_image in new_images:
#     if "__with_mask" in new_image:
#         shutil.move(new_image, dataset//")
