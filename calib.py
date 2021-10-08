import matplotlib.pyplot as plt
import numpy as np

filename = "/home/kluger/Downloads/4igvte9nfhf61.jpg"
image = plt.imread(filename)
save = True

if save:
    fig, ax = plt.subplots()
    ax.imshow(image)

    current_points = []
    vanishing_points = []


    def vp_from_line_segments(coordinates):

        lines = []

        for idx in range(0,len(coordinates),2):
            p1 = coordinates[idx]
            p2 = coordinates[idx+1]

            line = np.cross(p1, p2)

            lines += [line]

        if len(lines) < 2:
            assert False
        elif len(lines) > 2:
            assert False
        else:
            vp = np.cross(lines[0], lines[1])
            vp = vp / vp[2]

        return vp


    def onclick(event):
        global current_points
        global vanishing_points
        if event.name == 'button_press_event' and event.button == 3:
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            current_points += [np.array([event.xdata, event.ydata, 1])]
        elif event.name == 'key_press_event':
            print(event.key)
            if event.key == 'n':
                vp = vp_from_line_segments(current_points)
                vanishing_points += [vp]
                print(vp)
                current_points = []


    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onclick)

    plt.show()

    vanishing_points = np.vstack(vanishing_points)

    np.save('vps.npy', vanishing_points)

else:
    vanishing_points = np.load('vps.npy')

v1 = vanishing_points[0]
v2 = vanishing_points[1]
v3 = vanishing_points[2]

M = np.array([[v1[0] * v2[0] + v1[1] * v2[1], v1[0] + v2[0], v1[1] + v2[1], 1],
              [v3[0] * v2[0] + v3[1] * v2[1], v3[0] + v2[0], v3[1] + v2[1], 1],
              [v1[0] * v3[0] + v1[1] * v3[1], v1[0] + v3[0], v1[1] + v3[1], 1]])

u, s, vh = np.linalg.svd(M)
w = vh[-1]
print(np.dot(M, w))

W = np.array([[w[0], 0, w[1]], [0, w[0], w[2]], [w[1], w[2], w[3]]])
print("W: \n", W)

Kinv = np.linalg.cholesky(W).T
print(Kinv)
K = np.linalg.inv(Kinv)
print(K)
K = K/K[2,2]
print(K)