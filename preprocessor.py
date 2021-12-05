import os
import pickle
from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from tqdm import tqdm

names_actors=os.listdir('Dataset')

filenames = []
for i in names_actors:
    for j in os.listdir(os.path.join('Dataset',i)):
        filenames.append(os.path.join('Dataset',i,j))  ## for getting the exact path of the image
pickle.dump(filenames,open('filenames.pkl','wb'))


filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def feature_extractor(filepath,model):
    img = image.load_img(filepath,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0) ## For creating tehe batch of images
    preprocessed_img = preprocess_input(expanded_img) ## For creating the input image comapatible to vggface
    result = model.predict(preprocessed_img).flatten()  
    return result

features=[]

for i in tqdm(filenames):
    features.append(feature_extractor(i,model))

pickle.dump(features,open('features.pkl','wb'))
