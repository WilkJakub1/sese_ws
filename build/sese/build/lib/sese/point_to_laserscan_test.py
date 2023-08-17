import matplotlib.pyplot as plt
import numpy as np


angle_min = 0
angle_max = np.pi

resolution = 200
angle_increment = (angle_max - angle_min) / resolution



X = float(input('\nX:'))
Y = float(input('\nY:'))

R = np.sqrt(pow(X, 2) + pow(Y, 2))

range = (angle_max - angle_min) * R

section = range / resolution

if Y >= 0:
    alfa = np.arctan(Y/X)
    arc = alfa * R + (int(resolution/2)) * section
    # result = int(resolution/2)
else:
    alfa = np.arctan(np.abs(Y)/X)
    arc = (int(resolution/2))*section-alfa * R

print(f'alfa:{alfa}')
print(f'arc: {arc}')
print(f'section: {section}')

circle1 = plt.Circle((0, 0), R, color='r', fill=False)


result = 0

for i in np.arange(resolution):
    sekcja = section*(i+1)
    print(sekcja)
    if  sekcja < arc:
        result += 1
    else:
        break

print(f'R: {R}')
print(f'range: {range}')
print(f'result: {result}')

fig = plt.figure()
ax = fig.add_subplot(111)

line1, = ax.plot(Y, X, 'b.')
ax.add_patch(circle1)
fig.canvas.draw()
plt.show()
# fig.canvas.flush_events()