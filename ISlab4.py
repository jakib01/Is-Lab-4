import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, display=False):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize the image to create a binary format (black and white)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Optional: Display the binarized image for verification
    if display:
        cv2.imshow("Binarized Image", binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return the processed binary image
    return binary_image

# Class for Radial Basis Function Network (RBFN)
class RBFN:
    def __init__(self, num_neurons):
        # Initialize the number of neurons for the RBFN
        self.num_neurons = num_neurons

    # Method to train the RBFN
    def fit(self, X, y):
        # Use KMeans clustering to find the centers of RBF neurons
        kmeans = KMeans(n_clusters=self.num_neurons)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Compute RBF features using the radial basis function kernel
        rbf_features = rbf_kernel(X, self.centers)

        # Train a logistic regression model using the RBF features
        self.model = LogisticRegression()
        self.model.fit(rbf_features, y)

    # Method to predict labels for new input data
    def predict(self, X):
        # Compute RBF features for the input data
        rbf_features = rbf_kernel(X, self.centers)

        # Predict labels using the trained logistic regression model
        return self.model.predict(rbf_features)

# Function to segment digits from a binary image
def segment_digits(binary_image, h_=20):
    # Find contours in the binarized image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours from left to right based on their x-coordinates
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Initialize a list to store segmented digits
    digits = []

    # Iterate over each contour to extract individual digits
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box for the contour

        # Filter contours based on size constraints
        if w > 10 and h > h_:
            # Crop the digit and resize it to a fixed size (50x70 pixels)
            digit = binary_image[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (50, 70))
            digits.append(resized_digit)  # Append the digit to the list

    # Return the list of segmented digits
    return digits

# Function to visualize segmented digits
def visualize_digits(digits, predictions=None):
    # Iterate over each segmented digit
    for i, digit in enumerate(digits):
        # Resize the digit for better visibility in the display
        enlarged_digit = cv2.resize(digit, (200, 280), interpolation=cv2.INTER_LINEAR)

        # Set the window name for each digit
        window_name = f"Digit {i}"
        if predictions is not None:
            # Append the predicted label to the window name
            window_name += f": Predicted {predictions[i]}"

        # Display the digit in a window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, enlarged_digit)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to extract features from a list of images
def extract_features(images):
    # Flatten each image into a 1D vector and return as a numpy array
    return np.array([image.flatten() for image in images])

# Main script for training and testing
if __name__ == "__main__":
    try:
        # Path to the training image
        train_image_path = r"/Users/julkerakib/Documents/Master's Docs/First Semester/Intelligent System Lab/IS-Lab34/training.jpeg"

        # Load and preprocess the training image
        binary_train_image = load_and_preprocess_image(train_image_path)

        # Segment the digits from the training image
        train_digits = segment_digits(binary_train_image)
        num_train_digits = len(train_digits)  # Count the number of segmented digits
        print(f"Number of segmented digits from training image: {num_train_digits}")

        # Check if the correct number of digits is segmented for training
        if num_train_digits != 10:
            print(f"Error: found {num_train_digits} digits, but expected 10 for training.")
            visualize_digits(train_digits)  # Visualize the segmented digits for debugging
            exit(1)  # Exit the script if the number of digits is incorrect

        # Extract features from the training digits
        train_features = extract_features(train_digits)

        # Assign labels to the training digits (0-9)
        train_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Initialize and train the RBFN
        rbfn = RBFN(num_neurons=7)
        rbfn.fit(train_features, train_labels)

        # Path to the test image
        test_image_path = r"/Users/julkerakib/Documents/Master's Docs/First Semester/Intelligent System Lab/IS-Lab34/test.jpeg"

        # Load and preprocess the test image
        binary_test_image = load_and_preprocess_image(test_image_path)

        # Segment the digits from the test image
        test_digits = segment_digits(binary_test_image)
        num_test_digits = len(test_digits)  # Count the number of segmented digits
        print(f"Number of segmented digits from test image: {num_test_digits}")

        # Check if any digits are found in the test image
        if num_test_digits == 0:
            print("Error: No digits were found in the test image.")
            visualize_digits(test_digits)  # Visualize the test image for debugging
            exit(1)  # Exit the script if no digits are found

        # Extract features from the test digits
        test_features = extract_features(test_digits)

        # Predict labels for the test digits
        test_predictions = rbfn.predict(test_features)

        # Visualize the segmented and predicted test digits
        visualize_digits(test_digits, test_predictions)

    except FileNotFoundError as e:
        # Handle errors when the file is not found
        print(e)
