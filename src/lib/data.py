import random
import cv2
import os

names = lambda file: [
    line.strip() for line in open(
        os.path.join(os.path.dirname(__file__), '..', '..', 'db', file)
        ).readlines()
]

firstname = names("name.txt")

class HumanData:
    def __init__(self, faceBox, cropped_face, id=300000000):
        self.faceBox = faceBox
        self.cropped_face = cropped_face
        self.reset()
    
    def __repr__(self):
        return f"Face: {self.faceBox}."

    def reset(self):
        self.firstname = random.choice(firstname).lower().capitalize()
        self.lastname = random.choice(firstname).lower().capitalize()
        self.email = f"{self.lastname[:6].lower()}.{self.firstname[:4].lower()}@myvuw.ac.nz"
        self.id = 300000000 + random.randint(111111,999999)

