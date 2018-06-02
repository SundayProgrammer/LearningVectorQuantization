import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import plotly
import numpy as np
import pandas as pd

def read_data_csv(file_name):
    plot_data = pd.read_csv(file_name)
    return plot_data

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# y = np.arange(0.01,0.1,0.005)
y = np.arange(0.01,0.11,0.01)
x = np.arange(5,105,5)

X, Y = np.meshgrid(x, y)

plot_data = read_data_csv('../data/02_result_lvq1_1.csv')

temp_data = plot_data.values

ax1.set_xlabel('codebook number per class')
ax1.set_ylabel('learning rate')
ax1.set_zlabel('accuracy')

ax1.plot_surface(X, Y, temp_data)
# ax1.plot_wireframe(X, Y, temp_data)
plt.show()