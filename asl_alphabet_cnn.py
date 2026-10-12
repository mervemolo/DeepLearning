import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.metrics import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import MaxPooling2D, Conv2D, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#KLASÖR YOLLARI
dataset_yolu = r"C:\Users\Merve\Desktop\PythonProjects\DeepLearning\dataset"

train_dir = os.path.join(dataset_yolu, "asl_alphabet_train")
test_dir = os.path.join(dataset_yolu, "asl_alphabet_test")

class_labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
num_classes = len(class_labels)

# VERİ YÜKLEME FONKSİYONLARI
def egitim_verilerini_yukle(klasor_yolu):
    x = []
    y = []
    for label_id, class_name in enumerate(class_labels):
        harf_klasoru = os.path.join(klasor_yolu, class_name)
        if os.path.isdir(harf_klasoru):
            for resim_adi in os.listdir(harf_klasoru):
                resim_yolu = os.path.join(harf_klasoru, resim_adi)
                img = cv2.imread(resim_yolu)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    img = cv2.resize(img, (64, 64))
                    x.append(img)
                    y.append(label_id)
    return np.array(x), np.array(y).reshape(-1, 1)

def test_verilerini_yukle(klasor_yolu):
    x = []
    y = []
    if os.path.exists(klasor_yolu):
        for resim_adi in os.listdir(klasor_yolu):
            if resim_adi.endswith(('.jpg', '.jpeg', '.png')):
                sınıf_adı = resim_adi.split('_')[0] 
                if sınıf_adı in class_labels:
                    label_id = class_labels.index(sınıf_adı)
                    resim_yolu = os.path.join(klasor_yolu, resim_adi)
                    img = cv2.imread(resim_yolu)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                        img = cv2.resize(img, (64, 64))
                        x.append(img)
                        y.append(label_id)
    return np.array(x), np.array(y).reshape(-1, 1)

# Verileri bilgisayardan belleğe oku
print("Eğitim verileri yükleniyor... (Biraz sürebilir)")
x_train, y_train = egitim_verilerini_yukle(train_dir)

print("Test verileri yükleniyor...")
x_test, y_test = test_verilerini_yukle(test_dir)


# Normalizasyon ve One-Hot encoding

x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255

y_train_cat=to_categorical(y_train,num_classes=len(np.unique(y_train)))
y_test_cat=to_categorical(y_test,num_classes=len(np.unique(y_train)))

# Model Mimarisi

model=Sequential()

model.add(Conv2D(32,(3,3),padding="same",activation="relu",input_shape=x_train.shape[1:]))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(np.unique(y_train)),activation="softmax"))


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


datagen=ImageDataGenerator(
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # horizontal_flip=True,sol el ve sağ el farklı anlamlara gelebilir!
    fill_mode="nearest"
)
datagen.fit(x_train)

early_stopping=EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=2,
    restore_best_weights=True
)

checkpoint=ModelCheckpoint(
    filepath="en_iyi_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
)

history=model.fit(datagen.flow(x_train,y_train_cat,batch_size=64),epochs=10,callbacks=[early_stopping,checkpoint],validation_data=(x_test,y_test_cat))

#tahmin ve performans değerlendirme
y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,axis=1)
y_true=np.argmax(y_test_cat,axis=1)
report = classification_report(
    y_true, 
    y_pred_class, 
    labels=np.arange(len(class_labels)),  # Sınıf indeks aralığını (0-28) zorunlu tutar
    target_names=class_labels
)
print(report)

# Grafik çizme

plt.figure(figsize=(15,3))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"],label="Eğitim Başarısı(Train Accuracy):")
plt.title("Accuracy Gelişimi")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk oranı")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["loss"],label="Eğitim Kaybı (Loss):")
plt.title("Loss Gelişimi")
plt.xlabel("Epochs")
plt.ylabel("Loss Oranı")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#veri setini doğrulama kümelerine bölüyorujm çünkü yetersiz test verisi var

x_train,x_val,y_train_cat,y_val_cat=train_test_split(x_train,y_train_cat,test_size=0.1,random_state=42,stratify=y_train_cat)

history2=model.fit(datagen.flow(x_train,y_train_cat,batch_size=256),validation_data=(x_val,y_val_cat),epochs=10,callbacks=[early_stopping,checkpoint],verbose=1)

y_pred2=model.predict(x_val)
y_pred2_class=np.argmax(y_pred2,axis=1)
y_true2=np.argmax(y_val_cat,axis=1)
report2=classification_report(y_true2,y_pred2_class,target_names=class_labels,labels=np.arange(len(class_labels)))
print(report2)