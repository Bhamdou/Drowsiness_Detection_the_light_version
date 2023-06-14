import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold value for EAR below which we will consider the person's eyes as "closed"
EAR_THRESHOLD = 0.3
# Consecutive frame counter
FRAME_COUNTER = 48

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/gun/Documents/DCI-CODE/exercise/Drowsiness_Detection/shape_predictor_68_face_landmarks_GTX.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)

        EAR = (leftEAR + rightEAR) / 2.0

        if EAR < EAR_THRESHOLD:
            flag += 1
            if flag >= FRAME_COUNTER:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
