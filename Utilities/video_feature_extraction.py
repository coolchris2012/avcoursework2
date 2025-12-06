import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn.decomposition

def main(path_to_video = "Raw_Clips/video/ahmed001.MOV", filename = "ahmed001", folderName = "Visual_Features_Edge_Detection", edge_detection = True):
  # models' settings. XML files are located in the same directory as the script.
  face_detector = cv.CascadeClassifier("detectors/haarcascade_frontalface_default.xml")
  mouth_detector = cv.CascadeClassifier("detectors/haarcascade_smile.xml")

  cap = cv.VideoCapture(path_to_video)
  # fps = cap.get(cv.CAP_PROP_FPS)
  # sleep = int(1000 / fps)

  new_image_width = 640   # downsample the image

  frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  final_smile_width = 100
  final_smile_height = int(final_smile_width/2)
  features = np.empty((frame_count, 50, 10))

  # run continuously
  while cap.isOpened():
    ret, frm = cap.read()
    if not ret:
      break

    # resize image to 640 width
    new_image_height = int(new_image_width * frm.shape[0] / frm.shape[1])
    frm_resized = cv.resize(frm, (new_image_width, new_image_height))

    # apply face detection
    face_bbox = face_detector.detectMultiScale(frm_resized, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
    if len(face_bbox) > 0:
      for (face_x1, face_y1, face_width, face_height) in face_bbox:
        face_RoI = frm_resized[face_y1:face_y1 + face_height, face_x1:face_x1 + face_width]

        face_RoI_bttm_limit = int(face_height * .65) # mouth search region, bottom 35% of the face region

        # apply mouth/smile detection to the face ROI
        smile_bboxes = mouth_detector.detectMultiScale(face_RoI[face_RoI_bttm_limit:, :, :], scaleFactor=1.15,
                                                     minNeighbors=15, minSize=(15, 15))
      

        if len(smile_bboxes) > 0:
          for (smile_x1, smile_y1, smile_width, smile_height) in smile_bboxes:
            smile_RoI = face_RoI[face_RoI_bttm_limit + smile_y1:face_RoI_bttm_limit + smile_y1 + int(smile_width/2), smile_x1:smile_x1 + smile_width]

            smile_RoI = cv.cvtColor(smile_RoI, cv.COLOR_BGR2GRAY)

            smile_RoI_resized = cv.resize(smile_RoI, (final_smile_width, final_smile_height))

            smile_RoI_dct = cv.dct(np.float32(smile_RoI_resized))
            smile_RoI_dct_trunc = np.copy(smile_RoI_dct)
            for i in range(0, final_smile_height):
              for j in range(0, final_smile_width):
                if (i+j) > 100:
                  smile_RoI_dct_trunc[i, j] = 0
            
            smile_RoI_resized_new = cv.idct(smile_RoI_dct_trunc)


            if (edge_detection):
              smile_RoI_resized_blurred = cv.GaussianBlur(smile_RoI_resized_new, (3, 3), 0)
              
              sobelx = cv.Sobel(smile_RoI_resized_blurred, cv.CV_64F, 1, 0, ksize=3)
              sobely = cv.Sobel(smile_RoI_resized_blurred, cv.CV_64F, 0, 1, ksize=3)

              gradient_magnitude = cv.magnitude(sobelx, sobely)
              converted_gradient_magnitude = cv.convertScaleAbs(gradient_magnitude)

              smile_RoI_resized_final = cv.threshold(converted_gradient_magnitude, 100, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

            else:
              smile_RoI_resized_final = smile_RoI_resized_new

            pca = sklearn.decomposition.PCA(n_components=10)
            pca.fit(smile_RoI_resized_final)
            pca = pca.transform(smile_RoI_resized_final)

            features[int(cap.get(cv.CAP_PROP_POS_FRAMES)-1)] = pca

    # cv.imshow("frm_resized", smile_RoI_resized_final)
    # key = cv.waitKey(sleep)
    # if key == ord("q"):
    #   break

  # cv.destroyAllWindows()

  try:
    Path(folderName).mkdir()
  except FileExistsError:
    print("folder found")
  except PermissionError:
    print("Could not create folder")
  np.save(folderName + "/" + filename + ".npy", features)
  

# main()