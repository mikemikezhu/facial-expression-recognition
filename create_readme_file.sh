 # Install nbconvert package
 pip install --upgrade nbconvert

 # Remove previously generated README.md file
 rm -rf facial_expression_recognition_files/
 rm -rf README.md

 # Convert jupyter notebook to markdown
 jupyter nbconvert --to markdown facial_expression_recognition.ipynb

 # Rename README.md
 mv facial_expression_recognition.md README.md