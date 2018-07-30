# import the needed modules
import os
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img
from keras import backend as K
from keras.models import load_model
import cv2
from tqdm import tqdm
from vid_draw import vid_draw

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors,generate_colors 

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval


#Provide the name of the image that you saved in the images folder to be fed through the network
#input_video_name = "C:\\Users\\P V\\Downloads\\test2.mp4"

#Obtaining the dimensions of the input image

#Assign the shape of the input image to image_shapr variable
#image_shape = (416,416)



#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
# If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
#boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


#Preprocess the input image before feeding into the convolutional network
#image, image_data = preprocess_image(frame, model_image_size = #(416, 416))
#Run the session



video_out = 'C:\\Users\\P V\\Desktop\\GitRepos\\YOLOw-Keras\\out\\huhhh.mp4'
video_reader = cv2.VideoCapture('C:\\Users\\P V\\Desktop\\GitRepos\\YOLOw-Keras\\project_video.mp4')
nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
image_shape = (frame_h,frame_w)
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)

video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'XVID'), 
                               50.0, 
                               (frame_w, frame_h))




for i in tqdm(range(nb_frames)):
    ret, image = video_reader.read()    
    input_image = cv2.resize(image, (416, 416))
    input_image = input_image / 255.
    input_image = input_image[:,:,::-1]
    input_image = np.expand_dims(input_image, 0)
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
    out_scores, out_boxes, out_classes = sess.run([scores, boxes,classes],feed_dict={yolo_model.input:input_image,K.learning_phase(): 0})
    image = vid_draw(image,out_boxes,out_classes)       
    video_writer.write(np.uint8(image))
video_reader.release()
cv2.destroyAllWindows()
