import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
cv2.line(img, (0, 0), (511, 511), (0, 255, 255), 6)

cv2.imshow("demo",img)

cv2.waitKey(0)
cv2.destroyAllwindows()