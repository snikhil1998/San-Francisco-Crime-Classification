# San-Francisco-Crime-Classification
A machine learning model to predict the category of past crime incidents given time and location from San Francisco crime report dataset.

## Usage
Extract the csv files to the working directory. The csv files and the python file should be in the same directory. To run on GPU, an Nvidia GPU with CUDA support is required. CUDA should be installed and then Python modules cupy, cudf, and cuml should be installed.

## Building docker image
RAPIDS.ai provides a docker image which has all the necessary software and this will be used to build our docker image. The csv files are automatically extracted during the image build process. Any additional Python modules which should be installed can be added to `requirements.txt`
