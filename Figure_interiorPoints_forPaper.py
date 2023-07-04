import numpy as np
import matplotlib.pyplot as plt
from IPython import embed


def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

"""
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
R = np.linspace(0, 2*np.pi, 500)
theta = np.sin(10*R)+ 2
# ax.plot(R, theta, '-k')
# ax.plot(0.09, 2.47, 'om')
# ax.axis('off')
#
# # rlist = np.linspace(0, 3.0, 500)
# # thetalist = np.linspace(0, 2*np.pi, 500)
# # X, Y = np.meshgrid(thetalist, rlist)
# # Z = 1/np.cos(X - np.pi/6)
# # cp = ax.contourf(X, Y, Z)
#
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.savefig("polar_flower.png")
# plt.show()



fig, ax = plt.subplots(1, 1)

x, y = pol2cart(theta, R)
plt.plot(x, y, '-k', linewidth=1)

x_grad = np.linspace(1, 1.48, 100)
y_grad = -1*x_grad + 1.4
plt.plot(x_grad, y_grad, '-w', linewidth=2)

index = np.where(np.logical_and(np.logical_and(x > 1.46, x < 2.9), np.logical_and(y > -0.09, y < 0.35)))
x_constraint = x[:10] # and y< ...
y_constraint = y[:10]
plt.plot(x_constraint, y_constraint, '-w', linewidth=2)
x_constraint = x[495:500] # and y< ...
y_constraint = y[495:500]
plt.plot(x_constraint, y_constraint, '-w', linewidth=2)

plt.plot(1, 0.4, 'ow', markersize=5)  # fillstyle='none'
plt.plot(2.88, 0.35, 'om', markersize=5)

xlist = np.linspace(-3.5, 3.5, 500)
ylist = np.linspace(-3.5, 3.5, 500)
X, Y = np.meshgrid(xlist, ylist)
Z = X - Y
cp = ax.contourf(X, Y, Z, 20)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
ax.set_aspect('equal', 'box')
plt.savefig("gradient.png", dpi=300)
plt.show()
"""




fig, ax = plt.subplots(1, 1)

xlist = np.linspace(-3.5, 3.5, 500)
ylist = np.linspace(-3.5, 3.5, 500)
X, Y = np.meshgrid(xlist, ylist)
Z = (X/0.5)**2 + Y**2 + 10*(X+0.6-Y)**2
cp = ax.contourf(X, Y, Z, 20)

plt.plot(xlist, -xlist+0.6, "k", linestyle=(0, (5, 10)), linewidth=0.5)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
ax.set_aspect('equal', 'box')
plt.savefig("Unconstrainted_version.png", dpi=300)
plt.show()


fig, ax = plt.subplots(1, 1)

xlist = np.linspace(-3.5, 3.5, 500)
ylist = np.linspace(-3.5, 3.5, 500)
X, Y = np.meshgrid(xlist, ylist)
Z = (X/0.5)**2 + Y**2
cp = ax.contourf(X, Y, Z, 20)

plt.plot(xlist, -xlist+0.6, '-k', linewidth=2)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
ax.set_aspect('equal', 'box')
plt.savefig("Constrainted_version.png", dpi=300)
plt.show()




fig, ax = plt.subplots(1, 1)

xlist = np.linspace(-7.5, 7.5, 500)
ylist = np.linspace(-7.5, 7.5, 500)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sin(X) + np.sin(Y)
cp = ax.contourf(X, Y, Z, 20)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.axis('off')
ax.set_aspect('equal', 'box')
plt.savefig("Multiple_minimum.png", dpi=300)
plt.show()





