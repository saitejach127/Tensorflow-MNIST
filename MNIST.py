import tensorflow as tf
from tensorflow import keras

#DATASET MNIST
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

#MODEL
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

#COMPILE( ADAM OPTIMIZER )
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

#TRAINING
model.fit(x_train,y_train, epochs=6)

#EVALUATION
score = model.evaluate(x_test, y_test)
print(" loss "+str(score[0]))
print(" accuracy "+str(score[1]))

#PREDICTING
import matplotlib.pyplot as plt
def prediction(img):
    plt.imshow(img)
    img = img.reshape(1,28,28)
    one_hot = []
    pred = model.predict(img)
    for i in pred:
        for j in i:
            one_hot.append(round(j))

    print(one_hot)

#TEST IMAGE
prediction(x_test[0])



