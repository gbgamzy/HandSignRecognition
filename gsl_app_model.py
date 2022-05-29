import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




import os


ds = pd.read_csv('./dataset/database.csv')
ds = ds.sort_values('label')
ds.head()

num_samples = len(ds)
classes = pd.unique(ds.label)
print(f'Samples: {num_samples}')
print(f'Classes: {classes}')


# Shuffle and split
ds_y = ds.label.array
ds_y = pd.factorize(ds_y)[0]
ds_y = np.array(ds_y)

ds_x = ds.drop(['label'], axis=1)
ds_x = ds_x.to_numpy()


print(ds_y.shape)
print(ds_x.shape)

num_samples = ds_x.shape[0]
dim = 3
landmarks = 21
ds_x = ds_x.reshape((num_samples, landmarks, dim))
print(ds_x.shape)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ds_x, ds_y, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



# plot coordinates
# num = 10
# example = ds_x[num]
# label = ds_y[num]

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# for vec in example:
#     ax.scatter(vec[0], vec[2], vec[1], color='b')

# plt.title(f'Label: {classes[label]}')
# ax.set_xlabel('X')
# ax.set_ylabel('Z')
# ax.set_zlabel('Y')

plt.show()


# hyperparameters
epochs = 10
val_split = .2

# labels to one hot
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train.astype(int))
y_test = to_categorical(y_test.astype(int))

print(y_train.shape)
print(y_test.shape)

# create callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_acc',
                        restore_best_weights=True,
                        patience=10,
                        verbose=0)

reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                            factor=0.2,
                            min_lr=0.00001,
                            patience=5,
                            verbose=0)





# Model

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, AlphaDropout, LayerNormalization

inputs = Input(shape=x_train[0].shape, name='Landmark_Vectors')

layerNorm = LayerNormalization(name='LayerNorm')(inputs)

flatten = Flatten(name='Flatten_Vectors')(layerNorm)

dense_count = 6
dense_base = 48
out = flatten

for i in range(dense_count):
    units = (dense_count-i) * (dense_count-i) * dense_base
    dense = Dense(units, 
                kernel_initializer="lecun_normal", 
                bias_initializer="zeros", 
                activation='selu',
                name=f'Dense_{i+1}')
    a_dropout = AlphaDropout(0.05, name=f'Dropout_{i+1}')
    out = dense(out)
    out = a_dropout(out)

outputs = Dense(y_train[0].shape[0], activation='softmax', name='Output_Vector')(out)
model = Model(inputs=inputs, outputs=outputs, name="SNN_6")
model.summary()


from tensorflow.keras.optimizers import Adam

adam = Adam(learning_rate=0.001, beta_2=0.99, epsilon=0.01)
print("adam compiled")
model.compile(loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['acc'])
print("model complete")
try:

    history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=val_split,
                    callbacks=[early_stop, reduce_lr],
                    verbose=0)
except:
    print("not done")


model.predict()


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['acc'], label='train acc')
plt.plot(history.history['val_acc'], label='val acc')
plt.legend()


loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f'++++++++++++ Test data ++++++++++++\nloss={loss:.4f} acc={acc:.4f}')



from sklearn import metrics
import seaborn as sns

predictions = model.predict(x_test)
prediction_classes = np.argmax(predictions, axis=-1)

gt_classes = np.argmax(y_test, axis=-1)
confusion_matrix = metrics.confusion_matrix(gt_classes, prediction_classes)

sns.heatmap(pd.DataFrame(confusion_matrix, index=classes, columns=classes), annot=True, cmap="YlGnBu", fmt='d')
plt.tight_layout()
plt.title('confusion matrix - ' + model.name, y=1.1)
plt.ylabel('predicted')
plt.xlabel('ground truth')
plt.show()


