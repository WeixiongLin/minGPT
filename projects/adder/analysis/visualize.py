# %%
#方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# %% [markdown]
# 定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图

z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()


# %% [markdown]
# 乘法 - 3D
fig = plt.figure()
ax1 = plt.axes(projection='3d')

N = 100
xd = np.arange(0, N, 1)
yd = np.arange(0, N, 1)
zd = xd * yd % 10

ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
plt.savefig('multi_3d.png')
# plt.show()


# %%
# 乘法 - xd=5
# plot (yd, zd)

fig = plt.figure()
# ax1 = plt.axes(projection='3d')

N = 10
xd = 7
# xd = 5
yd = np.arange(0, N, 1)
zd = xd + yd % 10
# zd = xd * yd % 10

# ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
plt.scatter(yd, zd, marker='o')
plt.savefig('add.png')
# plt.savefig('multi.png')
# plt.show()


# %%
fig = plt.figure()
ax1 = plt.axes(projection='3d')

N = 100
xd = np.arange(0, N, 1)
yd = np.arange(0, N, 1)
zd = xd + yd % 10

ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
plt.savefig('add_3d.png')
# plt.show()


# %%
#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义图像和三维格式坐标轴
fig=plt.figure()
ax2 = Axes3D(fig)
