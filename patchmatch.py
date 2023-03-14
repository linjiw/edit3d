import numpy as np
import cv2

# Set the number of iterations and the patch size
NUM_ITER = 5
PATCH_SIZE = 7

# Load the damaged image and the mask
img = cv2.imread('original_1.png')
mask = cv2.imread('new_mask.png', cv2.IMREAD_GRAYSCALE)

# Convert the mask to a binary mask
mask = mask > 0

# Initialize the filled image with the damaged image
filled_img = img.copy()

# Loop over the iterations
for i in range(NUM_ITER):
    print(f"Iteration {i+1}")
    
    # Compute the size of the search window
    search_size = np.array(img.shape[:2]) - PATCH_SIZE
    
    # Initialize the random nearest neighbor field
    nnf = np.zeros(img.shape[:2] + (2,), dtype=np.int32)
    nnf[:, :, 0] = np.random.randint(PATCH_SIZE, img.shape[0] - PATCH_SIZE, img.shape[:2])
    nnf[:, :, 1] = np.random.randint(PATCH_SIZE, img.shape[1] - PATCH_SIZE, img.shape[:2])
    
    # Loop over the pixels in the image
    for y in range(PATCH_SIZE, img.shape[0] - PATCH_SIZE):
        for x in range(PATCH_SIZE, img.shape[1] - PATCH_SIZE):
            
            # Skip the pixel if it's in the mask
            if mask[y, x]:
                continue
            
            # Extract the patch from the filled image
            patch = filled_img[y - PATCH_SIZE:y + PATCH_SIZE + 1, x - PATCH_SIZE:x + PATCH_SIZE + 1]
            
            # Extract the patch from the damaged image
            ref_patch = img[y - PATCH_SIZE:y + PATCH_SIZE + 1, x - PATCH_SIZE:x + PATCH_SIZE + 1]
            
            # Compute the distance between the patches
            dist = np.sum(np.square(patch - ref_patch))
            
            # Initialize the best match
            best_match = nnf[y, x]
            
            # Loop over the pixels in the search window
            for dy in range(-search_size[0], search_size[0] + 1):
                for dx in range(-search_size[1], search_size[1] + 1):
                    
                    # Compute the position of the neighbor
                    ny = y + dy
                    nx = x + dx
                    
                    # Skip the neighbor if it's outside the image or in the mask
                    if ny < PATCH_SIZE or ny >= img.shape[0] - PATCH_SIZE or nx < PATCH_SIZE or nx >= img.shape[1] - PATCH_SIZE or mask[ny, nx]:
                        continue
                    
                    # Extract the patch from the filled image
                    nn_patch = filled_img[ny - PATCH_SIZE:ny + PATCH_SIZE + 1, nx - PATCH_SIZE:nx + PATCH_SIZE + 1]
                    
                    # Compute the distance between the patches
                    nn_dist = np.sum(np.square(nn_patch - ref_patch))
                    
                    # Update the best match if the neighbor is closer
                    if nn_dist < dist:
                        dist = nn_dist
                        best_match = (ny, nx)
            
            # Update the nearest neighbor field
            nnf[y, x] = best_match
            
            # Fill in the pixel with the best match
            filled_img[y, x] = img[best_match[0], best_match[1]]
            
    # Show the filled image
    cv2.imshow("Filled Image", filled_img)
    cv2.waitKey(0)

