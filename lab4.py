import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel 
from sklearn.linear_model import LogisticRegression


#Creating a function to load and preprocess the image
def load_and_preprocess_image(image_path, display=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV) #Binarizing the number
    if display:#Displaying the image
        cv2.imshow("Binarized Image", binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return binary_image

#Creating a function for RBNF
class RBFN:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
#Training the model using the cluster and logistic regression model
    def fit(self, X, y):  
        # Clustering to find centers
        kmeans = KMeans(n_clusters=self.num_neurons)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Compute RBF features and train logistic regression model
        rbf_features = rbf_kernel(X, self.centers)
        self.model = LogisticRegression()
        self.model.fit(rbf_features, y)

    def predict(self, X): #Predicting the labels for new data.
        rbf_features = rbf_kernel(X, self.centers)
        return self.model.predict(rbf_features)

#Segmenting the  digits from the image and returns them as a list of images.
def segment_digits(binary_image, h_ = 20): 
    
    # Finding contours on the binarized image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contours from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # Digit segmentation with size filtering
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > h_:  # Filtering contours based on minimum and maximum sizes
            digit = binary_image[y:y + h, x:x + w]
            resized_digit = cv2.resize(digit, (50, 70)) 
            digits.append(resized_digit)

    return digits
#Enlarging the image for better viewing
def visualize_digits(digits, predictions=None):
    for i, digit in enumerate(digits):
        # Enlarge the digit image for display
        enlarged_digit = cv2.resize(digit, (200, 280), interpolation=cv2.INTER_LINEAR)

        # Window name with predicted digit
        window_name = f"Digit {i}"
        if predictions is not None:
            window_name += f": Predicted {predictions[i]}"

        # Creating a larger window and displaying the image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, enlarged_digit)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#Extracting features from a list of images by flattening them into vectors.
def extract_features(images):
    return np.array([image.flatten() for image in images])

if __name__ == "__main__":
    
    #Training
    try:
        # train_image_path = r"C:\Users\sohai\OneDrive\Desktop\Intelligent System Lab\Lab4\digits.jpg"
        train_image_path = r"/Users/julkerakib/Documents/Master's Docs/First Semester/Intelligent System Lab/IS-Lab34/training.jpeg"
        binary_train_image = load_and_preprocess_image(train_image_path) #Loading and preprocessing the image
        # Segmenting the digits from the training image
        train_digits = segment_digits(binary_train_image)
        num_train_digits = len(train_digits)
        print(f"Number of segmented digits from training image: {num_train_digits}")

        # Checking if the number of segmented digits is correct for training
        if num_train_digits != 10:
            print(
                f"Error: found {num_train_digits} digits, but expected 10 for training."
            )
            visualize_digits(train_digits)
            exit(1)

        
        train_features = extract_features(train_digits)# Extracting the features from training digits

        train_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Training the RBFN
        rbfn = RBFN(num_neurons=7)
        rbfn.fit(train_features, train_labels)

        # Testing
        # test_image_path = r"C:\Users\sohai\OneDrive\Desktop\Intelligent System Lab\Lab4\test1.jpg"
        test_image_path = r"/Users/julkerakib/Documents/Master's Docs/First Semester/Intelligent System Lab/IS-Lab34/test.jpeg"
        binary_test_image = load_and_preprocess_image(test_image_path)

        # Segment digits from the test image
        test_digits = segment_digits(binary_test_image)
        num_test_digits = len(test_digits)
        print(f"Number of segmented digits from test image: {num_test_digits}")

        # Check if there are any segmented digits
        if num_test_digits == 0:
            print("Error: No digits were found in the test image.")
            visualize_digits(test_digits)
            exit(1)

        # Extract features from test digits
        test_features = extract_features(test_digits)

        # Predictions on the test set
        test_predictions = rbfn.predict(test_features)

        # Visualize segmented and predicted test digits
        visualize_digits(test_digits, test_predictions)

    except FileNotFoundError as e:
        print(e)