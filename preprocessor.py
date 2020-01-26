from PIL import Image
import os
import imghdr
import numpy
import csv

# Constants
TRAINING_DATA_DIRECTORY = 'TrainingDataSelected/'
TRAIN_CSV_FILE = 'train.csv'
TRAIN_CSV_WRITE_MODE = 'w'
TRAIN_DATA_IMAGE_TYPE = 'jpeg'

FACIAL_EXPRESSION_LABELS = {
    'Happy': '0',
    'Sad': '1',
    'Surprise': '2'
}


class Preprocessor:

    def preprocess_data(self):
        """Preprocess facial expression dataset.
        This function will save training data to train.csv in the directory.
        train.csv contains two columns:
        - Facial expression label
        - Numpy array of facial expression image file (48 x 48)
        """
        with open(TRAIN_CSV_FILE, mode=TRAIN_CSV_WRITE_MODE) as train:

            # Create train writer
            train_writer = csv.writer(train)

            # Iterate through training image data
            with os.scandir(TRAINING_DATA_DIRECTORY) as entries:

                for entry in entries:
                    path = entry.path
                    file_type = imghdr.what(path)

                    if file_type == TRAIN_DATA_IMAGE_TYPE:

                        # Label
                        name = entry.name
                        facial_expression_name = name.split('_')[0]
                        label = FACIAL_EXPRESSION_LABELS[facial_expression_name]

                        # Image file
                        image_file = Image.open(path)
                        image = numpy.array(image_file)
                        pixels = '-'.join(map(str, image.flat))

                        # Write to CSV file
                        row = [label, pixels]
                        train_writer.writerow(row)


# Preprocess data
preprocessor = Preprocessor()
preprocessor.preprocess_data()
