import math

import cv2
import numpy as np


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)


def draw3b(f, width=64):
    width=width//2
    x0, y0, x1, y1, x2, y2, x3, y3, z0, z2, w0, w2 = f
    # x1 = x0 + (x2 - x0) * x1
    # y1 = y0 + (y2 - y0) * y1
    # x2 = x0 + (x3 - x1) * x2
    # y2 = y0 + (y3 - y1) * y2
    x0 = normal(x0, width*2 )
    x1 = normal(x1, width*2 )
    x2 = normal(x2, width*2 )
    x3 = normal(x3, width*2 )
    y0 = normal(y0, width*2 )
    y1 = normal(y1, width*2 )
    y2 = normal(y2, width*2 )
    y3 = normal(y3, width*2 )
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width*2 , width*2 ]).astype('float32')
    tmp = 1. / 20
    data=[]
    for i in range(20):# 10 means the circle will move 100 times along the line
        t = i * tmp
        x = (int)((1-t) * (1-t) * (1-t) * x0 + 3 * t * (1-t) * (1-t) * x1 + 3 * t * t * (1-t) * x2 + t * t * t * x3)
        y = (int)((1-t) * (1-t) * (1-t) * y0 + 3 * t * (1-t) * (1-t) * y1 + 3 * t * t * (1-t) * y2 + t * t * t * y3)
        z = (int)((1-t) * z0 + t * z2)
        w = (1,1,1)
        cv2.circle(canvas, (x, y), z, w, -1)
        data.append([x,y,z])
        # z = (int)(z / 2)
        # cv2.rectangle(canvas, (y-z, x-z), (y+z, x+z), w, -1)
    # return cv2.resize(canvas, dsize=(width, width))
    return cv2.blur(canvas, (5, 5)) , data
    # return cv2.resize(canvas, dsize=(width, width))

def redraw3b(list, width=64):
    canvas = np.zeros([width , width ]).astype('float32')
    i = 0
    while i < len(list)-3:# 100 means the circle will move 100 times along the line
        cv2.circle(canvas, (int(list[i]), int(list[i+1])), int(list[i+2]), (1,1,1), -1)
        i = i + 3
    return cv2.blur(canvas, (5, 5))