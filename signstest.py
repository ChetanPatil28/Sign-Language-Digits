import cv2
from keras.models import load_model
new_model=load_model('sign.h5')

capt=cv2.VideoCapture(0)
while True:
    _,frame=capt.read()
    image=frame[0:300,0:300]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ya=np.resize(gray,(64,64,1))
    pred=new_model.predict(np.expand_dims(ya,axis=0))
    pred=np.argmax(pred)
    cv2.putText(frame,'The digit is '+str(pred), (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)    
    
    cv2.imshow('img',frame)
    cv2.imshow('digit',gray)
    if cv2.waitKey(3) & 0xFF==ord('q'):
        break
capt.release()
cv2.destroyAllWindows()