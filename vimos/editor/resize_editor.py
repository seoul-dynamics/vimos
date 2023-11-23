import cv2

from vimos.base import Editor

class ResizeEditor(Editor):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def _process(self, image):
        data = image.data

        width, height = self.width, self.height
        if type(self.width) == float:
            width = int(data.shape[1] * self.width)
        if type(self.height) == float:
            height = int(data.shape[0] * self.height)

        data = cv2.resize(data, (width, height))
        image.data = data
        return image