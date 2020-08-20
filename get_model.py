from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


base_model = MobileNetV2(include_top=False, pooling="max")

for index, layer in enumerate(base_model.layers):
    if index == len(base_model.layers) - 15:
        break
    layer.trainable = False


x = Dense(256, activation='relu')(base_model.outputs[0])
output = Dense(1, activation='softmax')(x)
model = Model(inputs=base_model.inputs, outputs=[output])


model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.001),
              metrics=["accuracy"])


model.save('models/Mobile.h5')


print("Done!")