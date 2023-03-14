import cv2
import numpy
from pyinpaint import Inpaint


img = cv2.imread('original_1.png')


# Load the mask.
mask = cv2.imread('yellow_1.png', 0)

mask = 255 - mask
# cv2.imwrite('new_mask.png', mask)

  
# Inpaint.
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)



# inpaint = Inpaint('original_1.png', 'yellow_1.png')
# dst = inpaint()
  
# Write the output.
cv2.imwrite('cat_inpainted.png', dst)