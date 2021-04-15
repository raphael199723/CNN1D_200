import numpy as np
import pandas as pd
import sys
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
path='format_combine/test/'
std_mean_value = np.array(pd.read_csv('format_combine/std_mean/std_mean_value.csv', na_filter = False, header = None))
feature_num = 200
def preprocessing(path=str, feature_num=int):
	csv_name_list=[x for x in os.listdir(path)]
	x_test = []
	y_label = []
	reg = []
	reg1 = []
	for i in sorted(csv_name_list):
		x_test.append(np.array(pd.read_csv(path+i, na_filter = False, header = None))) # read_csv : read csv file; na_filter : whether detect missing value markers or not
		y_label.append(''.join([k for k in i if k.isdigit()]))
	for i in range(len(x_test)):
		#print(len(x_test[i]))
		for j in range(len(x_test[i])):
			tmp = str(x_test[i][j])
			#feature_num = len(tmp)
			reg.append(tmp.replace("['","").replace("']","").replace("\\t",",").split(','))
			if j < 4022:
				reg1.append('70,50'.split(','))
			elif j > 4021 and j < 7885 :
				reg1.append('50,170'.split(','))
			elif j > 7884 and j < 12122 :
				reg1.append('50,220'.split(','))
			elif j > 12121 and j < 16079 :
				reg1.append('150,70'.split(','))
			elif j > 16078 and j < 20135 :
				reg1.append('130,150'.split(','))
			elif j > 20134 and j < 23737 :
				reg1.append('130,250'.split(','))
			elif j > 23736 and j < 27327 :
				reg1.append('257,50'.split(','))
			elif j > 27326 and j < 30875 :
				reg1.append('320,50'.split(','))
			else:
				reg1.append('425,50'.split(','))
	N = len(reg)
	print('y_label: ',y_label[i])
	x_n_test = np.array(reg)#.reshape(N,feature_num) # N,23
	y_n_label = np.array(reg1).reshape(N,2) # N,2
	# print((x_n_test.dtype))
	# print((y_n_label.dtype))
	x_n_test = x_n_test.astype('float64') 
	y_n_label = y_n_label.astype('float64')

	# Standardization
	tmp = np.zeros([N, feature_num])

	for j in range(feature_num):
		print("std_check: ",std_mean_value[0,j])
		print("mean_check: ",std_mean_value[1,j])
		std_t = std_mean_value[0,j]
		mean_t = std_mean_value[1,j]
		for i in range(N): 
			tmp[i][j] = (x_n_test[i][j] - mean_t) / std_t

	x_n_test = tmp
	print('x_n_test[0]',x_n_test[0],';size:',len(x_n_test[0]))
	x_n_test = x_n_test.reshape(int(len(x_n_test)),200,1) #B,W,H,Channel ; B*W=total num
	return x_n_test, y_n_label

def main():
	x_n_test, y_n_label = preprocessing(path, feature_num)

	model = keras.models.load_model('CNN1d.h3')
	predict_value = model.predict(x_n_test)
	#print(predict_value, np.sum(predict_value[1300]))
	ans = np.array([])
	ans = np.array([])
	RP1 = np.array([70,50])
	RP2 = np.array([50,170])
	RP3 = np.array([50,220])
	RP4 = np.array([150,70])
	RP5 = np.array([130,150])
	RP6 = np.array([130,250])
	RP7 = np.array([257,50])
	RP8 = np.array([320,50])
	RP9 = np.array([425,50])
	print('len:',predict_value.shape[0])
	for i in range(predict_value.shape[0]):
		tmp=RP1*predict_value[i][0]+RP2*predict_value[i][1]+RP3*predict_value[i][2]+RP4*predict_value[i][3]+RP5*predict_value[i][4]+RP6*predict_value[i][5]+RP7*predict_value[i][6]+RP8*predict_value[i][7]+RP9*predict_value[i][8]
		if ans.shape[0] == 0:
			ans = np.array([tmp])
		else:
			ans = np.append(ans, tmp)
	print((ans).reshape(-1,2)[0:20])
	print(y_n_label.reshape(-1,2)[0:20])
	print((ans).reshape(-1,2)[822:842])
	print(y_n_label.reshape(-1,2)[822:842])
	out_pred_format = []
	out_label_format = []
	str1 = ','
	for i in range((ans).reshape(-1,2).shape[0]):
		out_pred_format.append(str1.join( (ans.reshape(-1,2)[i]).astype('str') ))
		out_label_format.append(str1.join( (y_n_label.reshape(-1,2)[i]).astype('str') ))
	out = pd.DataFrame([out_pred_format,out_label_format]).T
	out.columns = ["ans","label"]
	out.to_csv('CNN1d_output_predict.csv',index = False)

if __name__ == "__main__":
	main()
