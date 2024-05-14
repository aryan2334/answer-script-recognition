import cv2
import json
import numpy as np

from google.cloud import vision
from google.cloud.vision_v1 import types
from google.cloud.vision_v1 import AnnotateImageResponse
import os
import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'prefab-pixel-422805-u9-d55e427f8a02.json'

def sharpen_image(image):
    sharpening_kernel = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])

    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened_image

client = vision.ImageAnnotatorClient()


image = cv2.imread(r'Dataset_jpg/image1.jpg')
sharpened_image = sharpen_image(image)
cv2.imwrite('ImageRecog/sharpened.jpg', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

with io.open(r'ImageRecog/sharpened.jpg', 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)
request = types.AnnotateImageRequest(
        image=image,
        features=[
                types.Feature(type='TEXT_DETECTION'),
                types.Feature(type='LABEL_DETECTION')
                  ]
    )

response = client.annotate_image(request)
json_response = AnnotateImageResponse.to_json(response)

res = json.loads(json_response)

print(type(json_response))
 
with open('ImageRecog/reponse.json', 'w') as file:
    json.dump(res, file)


