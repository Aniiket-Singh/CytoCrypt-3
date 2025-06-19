import cv2

def convert_to_grayscale(input_path, output_path):
    """
    Convert an image to grayscale and save it
    
    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the grayscale image
    """
    # Read the image
    img = cv2.imread(input_path)
    
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        return
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Save the grayscale image
    cv2.imwrite(output_path, gray_img)
    print(f"Grayscale image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_image = "images/peppers.png"  # Replace with your input image path
    output_image = "images/peppers_grayscale.png"  # Replace with your output path
    
    convert_to_grayscale(input_image, output_image)