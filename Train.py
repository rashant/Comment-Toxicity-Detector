import os
import tensorflow as tf
import pandas as pd
from keras.layers import Dense, Embedding, TextVectorization,LSTM,Dropout,Bidirectional
from keras.models import Sequential
from keras.callbacks import TensorBoard

df=pd.read_csv('train.csv')
x=df['comment_text']
y=df[df.columns[2:]].values

log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

max_features=200000 #number of words in the vocab dictionary
vectorizer=TextVectorization(max_tokens=max_features,output_sequence_length=1800,output_mode='int')
vectorizer.adapt(x.values)
vectorized_text = vectorizer(x.values)

dataset=tf.data.Dataset.from_tensor_slices((vectorized_text,y))
dataset=dataset.cache()
dataset=dataset.shuffle(160000)
dataset=dataset.batch(16)
dataset=dataset.prefetch(8)

train=dataset.take(int(len(dataset)*.7))
val=dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test=dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

model=Sequential()
model.add(Embedding(max_features+1,32))
model.add(Bidirectional(LSTM(32,activation='tanh')))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='sigmoid'))

model.compile(loss='BinaryCrossentropy',optimizer='Adam',metrics='accuracy')
print(model.summary())
model.fit(train,epochs=1,validation_data=val, callbacks=tb_callback)
model.save("model.h5")