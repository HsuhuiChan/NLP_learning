import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
x_init = np.array(range(100))
y_init = -5 * x_init + 10
y_noise = [0*(random.random()-0.5) for _ in range(100)]
y_final = y_init + y_noise
points = np.array([x_init, y_final])
filename = 'data.csv'
dataframe = pd.DataFrame({'x':x_init,'y':y_final})##字典中的key值即为csv中列名
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv(filename,index=False,sep=',')
#pandas 读取csv文件：data = pd.read_csv('test.csv')

plt.figure(1)
plt.scatter(x_init, y_final)
plt.show()
