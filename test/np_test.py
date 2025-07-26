import numpy as np

class1  = np.random.uniform(size=(1000, 2, 224))
class2 = np.ones(shape=(1000,2,224))+np.random.uniform(size=(1000,2,224))-0.5
label1 = np.zeros(shape=(1000))
label2 = np.ones(shape=(1000))


data = np.concatenate([class1,class2],axis=0)

x_datas = np.random.uniform(-10086, 10086, 1000).astype(np.float32)

x_mean,x_std = x_datas.mean(), x_datas.std()
print(x_std)