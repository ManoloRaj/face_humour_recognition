import tensorflow as tf
from tkinter import *
import cv2
import keras
from tensorflow.keras.preprocessing import image
import matplotlib as plt
import numpy as np



#Get data before training
def get_dataset(class_name, nb_image_to_get):

    vid_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i=1
    while i<=(nb_image_to_get*10):
        ret, frame = vid_capture.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
        for (x, y, w, h ) in faces : 
            cv2.rectangle( frame , (x,y), (x+w, y+h), (0,255,0), 2)
            crp_img = frame[y:y+h, x:x+w]
            cv2.imwrite(class_name+str(i)+'.jpg', crp_img)

        if cv2.waitKey(10) & 0XFF == ord('x') : 
            break
        cv2.imshow('img',frame)
        i=i+1

#Live detection and recognition
def face_detect_predict_live():
    #Loading model

    ml= keras.models.load_model('model.h5')

    vid_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True : 
        ret, frame = vid_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3 , 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)

            crp_img = frame[y:y+h, x:x+w]
            cv2.imwrite('image_cache.jpg',crp_img)

            #Get cache image
            img = image.load_img('image_cache.jpg',target_size=(200,200))


            X = image.img_to_array(img)
            X = np.expand_dims(X,axis=0)
            images = np.vstack([X])
             
            print(ml.predict(images))
            if ml.predict(images)<0.5 : 
                cv2.putText(frame, "No smile", (x+int(x/10), y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2 )
            else :
                cv2.putText(frame, "Smile", (x+int(x/10), y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2 )
            
        if cv2.waitKey(10) & 0XFF == ord('x') : 
            break

        cv2.imshow('image', frame)


def train_own_model():
    #CNN architexture
    classifier = tf.keras.models.Sequential([ 
                        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
                        tf.keras.layers.MaxPool2D(2,2),
                        
                        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
                        tf.keras.layers.MaxPool2D(2,2),

                        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
                        tf.keras.layers.MaxPool2D(2,2),

                        tf.keras.layers.Flatten(),

                        tf.keras.layers.Dense(512, activation='relu'),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                        ])

    classifier.summary()

    from keras.preprocessing.image import ImageDataGenerator
    #Preparing data
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    training_set = train_datagen.flow_from_directory('dataset',
                                                target_size=(200,200),
                                                batch_size = 32,
                                                class_mode='binary')
                            
    classifier.compile(loss = 'binary_crossentropy',
                    optimizer='adam',
                    metrics = ['accuracy'])

    classifier.fit_generator(training_set,
                         epochs = 20)

    classifier.save('model.h5')


#Application GUI
def interfaceGUI():

    #Button 1
    def bouton1_Action(nom_images, nombre_images):
        Label(l, text=">> Enregistre la base de donnée...").pack()
        print(nom_images)
        get_dataset(nom_images, nombre_images)


    #Button 2
    def bouton2_Action():
        Label(l, text =">> Prediction live video...").pack()
        face_detect_predict_live()


    fenetre =Tk("Humour detection")

    fenetre.geometry ("500x500")
    #Panel
    l = LabelFrame(fenetre, text ="Executed command", padx = 20, pady = 20)
    l.pack( fill = "both", expand = "yes")


    nom_images = Label(fenetre, text = "Image name (with relative path)*")
    form1 = Entry(fenetre)

    nombre_images = Label (fenetre, text = "Nombre of images *")
    form2 = Entry(fenetre)

    prendre_photo_button = Button(fenetre, text="Take photo for training", command = lambda : bouton1_Action(str(form1.get()), int(form2.get())))

    train_button = Button(fenetre, text = "Train our model", command = lambda : train_own_model())

    prediction_button = Button(fenetre, text = "Video live", command = lambda : bouton2_Action())

    

    #All pack
    nom_images.pack()
    form1.pack()
    nombre_images.pack()
    form2.pack()
    prendre_photo_button.pack()
    train_button.pack()
    prediction_button.pack()

    fenetre.mainloop()


interfaceGUI()