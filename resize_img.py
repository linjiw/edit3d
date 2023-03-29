import cv2

def resize_image(image_path, output_path, width, height):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Save the resized image
    cv2.imwrite(output_path, resized_image)

# Example usage
input_image_path = "origin_14_20.png"
output_image_path = f"resized_{input_image_path}"
desired_width = 1920
desired_height = 1080

resize_image(input_image_path, output_image_path, desired_width, desired_height)
