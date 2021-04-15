import pandas as pd
import sys

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop,Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
import os
import matplotlib.pyplot as plt
path='format_combine/train/'
feature_num = 200
RP_num = 6

def preprocessing(path=str, feature_num=int):
	csv_name_list=[x for x in os.listdir(path)]
	x_train=[]
	y_label=[]
	reg=[]
	reg1=[]
	for i in sorted(csv_name_list):
		print(i)
		x_train.append(np.array(pd.read_csv(path+i, na_filter = False, header = None))) # read_csv : read csv file; na_filter : whether detect missing value markers or not
		y_label.append(''.join([k for k in i if k.isdigit()]))
		#print(x_train)
	#print(y_label)
	for i in range(len(x_train)):
		print(len(x_train[i]))
		for j in range(len(x_train[i])):
			#print(x_train[i][jmp = str(x_train[i][j])
			tmp = str(x_train[i][j])
			#print(tmp)
			#feature_num = len(tmp)
			reg.append(tmp.replace("['","").replace("']","").replace("\\t",",").split(','))
			if j < 15056:
				reg1.append('1')
			elif j > 15055 and j < 30457 :
				reg1.append('2')
			elif j > 30456 and j < 45176 :
				reg1.append('3')
			elif j > 45175 and j < 60562 :
				reg1.append('4')
			elif j > 60561 and j < 76582 :
				reg1.append('5')
			elif j > 76581 and j < 95507 :
				reg1.append('6')
			elif j > 95506 and j < 111466 :
				reg1.append('7')
			elif j > 111465 and j < 126740 :
				reg1.append('8')
			else:
				reg1.append('9')

	N = len(reg)
	# print(reg1)
	x_n_train = np.array(reg)
	y_n_label = np.array(reg1).reshape(N,1) # N,1
	# print((x_n_train.dtype))
	# print((y_n_label.dtype))
	x_n_train = x_n_train.astype('float64') 
	y_n_label = y_n_label.astype('float64')

	onehotencoder = OneHotEncoder()
	y_n_label=onehotencoder.fit_transform(y_n_label).toarray()

	# Standardization
	save_std_mean_value = np.zeros([2,feature_num])
	tmp = np.zeros([N, feature_num])

	std_x = np.std(x_n_train, axis = 0) # axis=0，計算每一column的標準差
	mean_x = np.mean(x_n_train, axis = 0) # axis=0，計算每一column的平均值

	for j in range(feature_num):
		save_std_mean_value[0,j] = std_x[j]
		save_std_mean_value[1,j] = mean_x[j]
		for i in range(N): 
#			if std_x[j] != 0:
			tmp[i][j] = (x_n_train[i][j] - mean_x[j]) / std_x[j]

	if not os.path.exists("format_combine"):
		os.mkdir("format_combine")
	if not os.path.exists("format_combine/std_mean"):
		os.mkdir("format_combine/std_mean")
	save_std_mean_value = pd.DataFrame(save_std_mean_value)
	save_std_mean_value.to_csv("format_combine/std_mean/std_mean_value.csv",index = False,header = False)
	x_n_train = tmp
	#print(x_n_train)
	#print(y_n_label)

	x_n_train = x_n_train.reshape(int(len(x_n_train)),200,1) #B,W,H,Channel ; B*W=total num
	return x_n_train, y_n_label


def create_CNN1d():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(32, 4,strides = 4, activation='relu',input_shape=(200,1)),
            tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(9, activation='softmax')
            ])
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def save_model_plt(model=keras.models,history=keras.models,model_name=str,loss_plt_name=str,acc_plt_name=str):
	model.save(model_name)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.savefig(loss_plt_name)
	plt.clf()

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper right')
	plt.savefig(acc_plt_name)
	plt.clf()

def main():
	model = create_CNN1d()
	model.summary()
	early_stopping = EarlyStopping(monitor = 'val_loss', patience = 15, verbose = 1,mode = 'min')
	x_n_train, y_n_label = preprocessing(path, feature_num)
	history = model.fit(x_n_train, y_n_label, epochs=1500,batch_size=128, validation_split=0.01, shuffle=True, callbacks=[early_stopping])
	save_model_plt(model=model,history=history,model_name='CNN1d.h3',loss_plt_name='model_loss.pdf',acc_plt_name='model_acc.pdf')  


if __name__ == "__main__":
	main()

