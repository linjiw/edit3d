import cv2
import numpy as np
# Load the image
f_name = "mask_11_43.png"
image = cv2.imread(f_name)

# Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image[np.where(image!=0)] = 255
# # Threshold the image to create a binary mask
# mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

# # Invert the mask to convert black pixels to white
# mask = cv2.bitwise_not(mask)

# # Create a white background image
# white = np.full(image.shape, 255, dtype=np.uint8)

# # Copy the original image to the white background using the mask
# result = cv2.copyTo(image, white, mask)

# Save the result
cv2.imwrite(f'result_{f_name}', image)

# Show the result
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
