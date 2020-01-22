import numpy as np
import csv
from PIL import Image

facial_expression_labels = {
    "0": 'Angry',
    "1": 'Disgust',
    "2": 'Fear',
    "3": 'Happy',
    "4": 'Sad',
    "5": 'Surprise',
    "6": 'Neutral'
}

with open('original_train.csv') as csv_file:
    counter = {}
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header

    for row in csv_reader:

        facial_expression_index = row[0]
        facial_expression_label = facial_expression_labels.get(facial_expression_index)
        facial_expression_dir = 'TrainingData/' + facial_expression_label + '/'

        pixels_str = row[1]
        pixels_list = [int(i) for i in pixels_str.split(' ')]
        pixels_list = np.array(pixels_list, dtype='uint8')
        pixels_list = pixels_list.reshape((48, 48))

        print(pixels_list)
        image = Image.fromarray(pixels_list)

        if facial_expression_label not in counter:
            counter[facial_expression_label] = 0
        else:
            counter[facial_expression_label] += 1

        image_name = facial_expression_dir + facial_expression_label + '_' + \
                     str(counter[facial_expression_label]) + '.jpg'
        image.save(image_name, 'JPEG')
