import cv2
import numpy as np
import time
import random

from data import HumanData, firstnames, lastnames
from face_recognition import detectFace, displayInfo

prevHumanData: list[HumanData] = []
humanData: list[HumanData] = []

def intersection_over_union(box1, box2):
    # Extract coordinates of box1 and box2
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection area
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Calculate areas of the individual boxes
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    return iou

# Function to check similarity between two bounding boxes
def are_boxes_similar(box1, box2, iou_threshold=0.8, distance_threshold=50):
    # Calculate IoU
    iou = intersection_over_union(box1, box2)
    
    # If Intersection over union greater than threshhold th eboxes are similar
    if iou > iou_threshold:
        # print(f"Bounding boxes are similar with IoU = {iou:.2f}")
        return True
    
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)
    
    # Euclidean distance between the centers
    distance = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
    
    # If the distance is small enough, consider the boxes similar
    if distance < distance_threshold:
        # print(f"Bounding boxes are similar with center distance = {distance:.2f}")
        return True
    else:
        # print(f"Bounding boxes are not similar. Center distance = {distance:.2f}")
        return False


def setInfo(frame):
    global humanData, prevHumanData

    i = 0
    for face, prevFace in zip(humanData, prevHumanData):
        if (are_boxes_similar(face.faceBox, prevFace.faceBox, iou_threshold=0.7, distance_threshold=400)):
            prevHumanData[i].faceBox = humanData[i].faceBox
            humanData[i] = prevHumanData[i]
        else: 
            # cv2.imwrite(f"faces/{face.firstname}{face.lastname}.jpg", face.cropped_face)
            face.reset()
        i += 1


def display(camera):
    global humanData, prevHumanData

    ret, frame = camera.read()
    if not ret:
        raise IOError("Cannot read frame")
    
    greyImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    greyImage = cv2.equalizeHist(greyImage)

    for face in detectFace(greyImage=greyImage):
        x, y, w, h = face  # Unpack directly
        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)


        cropped_face = frame[y1:y2, x1:x2]
        humanData.append(HumanData(face, cropped_face))

    if len(humanData):
        setInfo(frame)
        frame = displayInfo(frame, humanData, prevHumanData)
        prevHumanData = humanData
        humanData = []
    cv2.imshow("Image", frame)


def basicDetection():
    camera = cv2.VideoCapture(0)
    prevFaces = []

    if not camera.isOpened():
        raise IOError("Can't open webca")

    while camera.isOpened():
        display(camera)
            
        # time.sleep(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break

def main():
    print("started")
    # Load the text file with names in it
    firstnames("name.txt")
    lastnames("name.txt")
    basicDetection()


if __name__ == "__main__":
    main()