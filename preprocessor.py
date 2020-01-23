from PIL import Image
import os
import imghdr
import numpy

TRAINING_DATA_DIRECTORY = 'TrainingDataSelected/'

facial_expression_labels = {
    'Happy': '0',
    'Sad': '1',
    'Surprise': '2'
}


def load_data():
    """Load facial expression dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train)`.
    """

    images = []
    labels = []

    # Iterate through training image data
    with os.scandir(TRAINING_DATA_DIRECTORY) as entries:

        for entry in entries:
            path = entry.path
            file_type = imghdr.what(path)

            if file_type == 'jpeg':
                
                image_file = Image.open(path)
                image = numpy.array(image_file)
                images.append(image)

                name = entry.name
                facial_expression_name = name.split('_')[0]
                label = facial_expression_labels[facial_expression_name]
                labels.append(label)
    
    x_train = numpy.array(images)
    y_train = numpy.array(labels)

    return (x_train, y_train)