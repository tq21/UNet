from model import *
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = unet()
model.load_weights("unet.hdf5")

pred_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_path = os.path.join('test')
pred_generator = pred_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    color_mode='grayscale',
    class_mode=None,
    shuffle=None
)

results = model.predict(pred_generator, workers=0)
results = results > 0.5

mask_path = os.path.join(os.getcwd(), r'test/label')
if not os.path.exists(mask_path):
    os.makedirs(mask_path)

i = 0
os.chdir(mask_path)
print(pred_generator.filenames)
for f_name in pred_generator.filenames:
    cv2.imwrite(f_name[6:], results[i] * 255)
    i += 1
