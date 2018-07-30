import cv2
import numpy



def vid_draw(image,boxes,classes):
    image_h, image_w, _ = image.shape
    
    
    for i,c in reversed(list(enumerate(classes))):
        #predicted_class = class_names[c]
        box = boxes[i]
        top, left, bottom, right = box
        x0 = int(left*image_w)
        y0 = int(top*image_h)
        x1 = int(right*image_w)
        y1 = int(bottom*image_h)
        cv2.rectangle(image,(x0,y0),(x1,y1),(0,255,0), 3)
    
    return image