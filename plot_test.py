import matplotlib.pyplot as plt
import numpy as np

num = 6000
plt.style.use('ggplot')
x = np.linspace(0, num-1, num)
y1 = np.sin(x+3.14/2)
y2 = np.sin(x)


fig = plt.figure(figsize = (300, 10), dpi = 100)
ax = fig.add_subplot(111)

ax.plot(x, y1)
ax.plot(x, y2)
ratio = 1
xleft, xright = ax.get_xlim()
ybottom, ytop = ax.get_ylim()
# the abs method is used to make sure that all numbers are positive
# because x and y axis of an axes maybe inversed.
#ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
plt.tight_layout()
plt.savefig('plot_test.png')

plt.show()