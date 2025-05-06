import cv2

from data import HumanData

# Detect Faces from an image
def detectFace(greyImage):
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )

    faces = face_classifier.detectMultiScale(
        greyImage, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    
    return faces

# Alters the image to contain HumanData and returns the image.
def displayInfo(frame, current: list[HumanData], prev: list[HumanData]):
    #Display Faces with Data.       
    for i, (x, y, w, h) in enumerate([face.faceBox for face in current]):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(
            frame, current[i].firstname + " " + current[i].lastname, (x, y-80),  cv2.FONT_HERSHEY_SIMPLEX,
            1, # font scale
            (0, 0, 0), # color
            6, # line thickness
            )
        cv2.putText(
            frame, current[i].email, (x, y-40),  cv2.FONT_HERSHEY_SIMPLEX,
            1, # font scale
            (0, 0, 0), # color
            6, # line thickness
            )
        cv2.putText(
            frame, str(current[i].id), (x, y),  cv2.FONT_HERSHEY_SIMPLEX,
            1, # font scale
            (0, 0, 0), # color
            6, # line thickness
        )

        cv2.putText(
            frame, "One Person at a time.", (700, 100),  cv2.FONT_HERSHEY_SIMPLEX,
            1, # font scale
            (55, 55, 55), # color
            6, # line thickness
            )
        return frame