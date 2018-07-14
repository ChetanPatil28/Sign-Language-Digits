import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model,Input,load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

X=np.load('X.npy')
Y=np.load('Y.npy')

#SETTING THE LABELS TO THEIR ORIGINAL CLASS
k=np.zeros(Y.shape)
for i in range(Y.shape[0]):
    if np.argmax(Y[i])==1:
        k[i][0]=1
    elif np.argmax(Y[i])==4:
        k[i][1]=1    
    elif np.argmax(Y[i])==8:
        k[i][2]=1    
    elif np.argmax(Y[i])==7:
        k[i][3]=1    
    elif np.argmax(Y[i])==6:
        k[i][4]=1
    elif np.argmax(Y[i])==9:
        k[i][5]=1        
    elif np.argmax(Y[i])==3:
        k[i][6]=1        
    elif np.argmax(Y[i])==2:
        k[i][7]=1        
    elif np.argmax(Y[i])==5:
        k[i][8]=1         
    elif np.argmax(Y[i])==0:
        k[i][9]=1      
#Splitting the dataset into training,validation and test set         
X_train,Xtest,Y_train,Ytest = train_test_split(X, k, test_size=0.09, random_state=29)
Xtrain, Xval, Ytrain, Yval = train_test_split(X_train, Y_train, test_size=0.09, random_state=14)

#Reshaping the data to be compatible with a convnet.
Xtrain=Xtrain.reshape((Xtrain.shape[0],64,64,1))
Xval=Xval.reshape((Xval.shape[0],64,64,1))
Xtest=Xtest.reshape((Xtest.shape[0],64,64,1))
print(Xtrain.shape,Xval.shape,Xtest.shape)    

tb=TensorBoard()

#Calling the Keras sequential API
model=Sequential()
model.add(Conv2D(8,(3,3),activation='relu',input_shape=(64,64,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#model.summary()
Hist=model.fit(Xtrain,Ytrain,batch_size=32,epochs=2,validation_data=(Xval,Yval),callbacks=[tb])
model.save('sign.h5')

def verifier(Xtest,Ytest):
    i=np.random.randint(Xtest.shape[0])
    plt.imshow(Xtest[i].reshape(64,64),cmap='gray')
    j=np.expand_dims(Xtest[i],axis=0)
    print('Predicted digit is',np.argmax(new_model.predict(j)))
    print('Actual digit is', np.argmax(Ytest[i]))

def conf(Xtest,Ytest,correct=0,wrong=0):
    correct=0
    wrong=0
    for i in range(Xtest.shape[0]):
        a=np.argmax(Ytest[i])
        b=np.argmax(new_model.predict(np.expand_dims(Xtest[i],axis=0)))
        if a==b:
            correct+=1
        else:
            wrong+=1
    print('Out of {}, correct predictions are {} and wrong predictions are {}.'.format(Xtest.shape[0],correct,wrong))
    print('Prediction accuracy is',np.multiply(correct/(correct+wrong),100))     
    
    
    
    
    