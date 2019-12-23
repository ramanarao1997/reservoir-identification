# reservoir-identification
Using Convolutional Neural Networks to identify if an image contains a reservoir.

## Data
The dataset used is USDA NAIP imagery. Pre-processed the data to get a sample of images having reservoirs and terrains. Feel free to download my sample from [here](https://drive.google.com/drive/u/0/folders/1m-aVwKA24N2vbI6-0rhnuXJvsbuVIoPR).

USDA - United States Department of Agriculture
NAIP - National Agriculture Imagery Program

# How to Run
Once you download the images from the link. Run the 'augment_data.py' file to generate additional images using data augmentation.
Open ResNet or VGG folder to run the model of your choice and execute the 'train.py' file.
When the code finishes execution it will generate a model.h5 file.
Run the 'test.py' file to test the model.

## Techniques used
Two Convolutional Neural Networks namely ResNet and VGG were used for this project.
Observed that ResNet performs better than VGG because of its 'Skip Connections'. 

