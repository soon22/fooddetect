import cv2
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow.keras

# Load Model 1
model1 = load_model('foodpresenceOrNot.h5')
# Load Model 2
model2 = load_model('fooddetect.h5')
foodlabels = ['dimsum','dumpling','fried rice','maggi curry','laksa Johor']
# Which cam ?I used a USB camera instead of my laptop's cam (In this case, Use 0)
cap = cv2.VideoCapture(1)
# Arguments for put text
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (50, 50) 
fontScale = 1 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
size = (224, 224)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Convert frame from BGR to RGB arrangement
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize, normalized and Reshape into input requirement
    img = cv2.resize(img,size)
    normalized = (img.astype(np.float32) / 127.0) - 1
    normalized = normalized.reshape((1, 224, 224, 3))
    # Prediction of Model 1
    pr = np.argmax(model1.predict(normalized))
    if pr == 1:
        label = 'unknown'
    else:
        # Prediction of Model 2
        pr2 = np.argmax(model2.predict(normalized))
        label = foodlabels[pr2]
    
    # Write text on frame
    cv2.putText(frame, label, org, font,  fontScale, color, thickness, cv2.LINE_AA) 

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
