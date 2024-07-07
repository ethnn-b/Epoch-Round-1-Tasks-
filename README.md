



# Task 1 - Classification Model

User-Defined Functions Used:

## Function to clean latitude and longitude 
clean_coordinate(coord_str)

Explanation: Extracts the coordinates in the CSV file as float

## Function to check if a point is within Kerala
is_within_kerala(row, kerala_geom)
    
## Class to implement the K-means clustering algorithm
class KM_alg

Functions inside:
  i) __init__(self,k)
  
  ii)euclidean_distance(point,centroids)
  
  Returns the Euclidean distance between a point and the centroids of the clusters
  
  iii)fit(self, X ,max_iterations=5000)

# Overview:
The clustering_data.csv file containing location details is loaded into a pandas data frame.
The shapefile for Kerala used for creative plotting can be found at: https://www.kaggle.com/datasets/nehaprabhavalkar/india-gis-data
The points outside the state are filtered out.

The K-means clustering algorithm is defined and used to plot the elbow curve.

We can observe the elbow falls at k=4.

We use this value to plot the final clusters.

We can infer that the locations and pin codes of BO offices provided in Kerala can be grouped into 4 major regions.
This division can be used to create a more efficient administrative system.

#Refences links used:
https://www.youtube.com/watch?v=iNlZ3IU5Ffw&pp=ygUYa21lYW5zIGNsdXN0ZXJpbmcgcHl0aG9u
https://www.youtube.com/watch?v=5w5iUbTlpMQ
https://www.youtube.com/watch?v=tJ4A6QBdEQs

# Task 2 - CV + NLP
Sentiment Analysis of Handwritten Text Using Custom-built OCR and Sentiment Classification Models

The models for OCR and Sentiment Analysis are both implemented using TensorFlow

User-Defined Functions Used:
## create_dataset(images, labels, batch_size=32, shuffle=True)

To create a TensorFlow dataset with the labels and image data.

## preprocess_test_image(img_path)

This is used to load and preprocess a test image.

## slice_image_into_characters(img, char_width=28, char_height=28)

Divides the image into portions of 28 x 28.


# Overview:
## Load and Preprocess Data:
The alphabet_labels.csv, and images inalphabet_images datasets are loaded.

 We load images, preprocess them (convert to grayscale and normalize), and store them with their corresponding labels and encode the labels as integers.

## OCR Model Building and Training:

We define and compile a Convolutional Neural Network (CNN) model to recognize characters from images.
Then we train the model on the preprocessed images and evaluate the model's accuracy on test data.

## Character Segmentation and Prediction:

We load test images, preprocess them, and slice them into individual characters.
Then we use the trained CNN model to predict the characters in the segmented images.
We reconstruct the original statement from the predicted characters and spaces.

## Load and Preprocess Sentiment Data:
We read a CSV file containing sentences and their corresponding sentiment labels.
Then, tokenize the sentences and convert them to padded sequences.
We encode the sentiment labels as integers.

## Model Building and Training:
We define and compile a Bidirectional LSTM model for sentiment analysis.
And train the model on the tokenized sentences.

## Sentiment Prediction:
Use the trained sentiment analysis model to predict the sentiment of the reconstructed statement from the character recognition part.
Then output the predicted sentiment (Angry, Happy, or Neutral).


Reference links used:
https://www.geeksforgeeks.org/sentiment-analysis-with-an-recurrent-neural-networks-rnn/
https://www.youtube.com/watch?v=JgnbwKnHMZQ&pp=ygUoc2VudGltZW50IGFuYWx5c2lzIHB5dGhvbiB0ZW5zb3JmbG93IHJubg%3D%3D
https://www.youtube.com/watch?v=eMPQw7Xbjd0&pp=ygUoc2VudGltZW50IGFuYWx5c2lzIHB5dGhvbiB0ZW5zb3JmbG93IHJubg%3D%3D
https://www.youtube.com/watch?v=Zi4i7Q0zrBs&pp=ygU5aGFuZHdyaXR0ZW4gY2hhcmFjdGVyIHJlY29nbml0aW9uIHVzaW5nIHB5dGhvbiB0ZW5zb3JmbG93
https://www.youtube.com/watch?v=Ixm_yQYK09E&pp=ygU5aGFuZHdyaXR0ZW4gY2hhcmFjdGVyIHJlY29nbml0aW9uIHVzaW5nIHB5dGhvbiB0ZW5zb3JmbG93








  Basic overview:
  
  
