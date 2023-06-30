# Application Setup
1. Make sure Docker is installed has daemon running on the machine in use.
2. Move to the root folder of this application on Terminal and run `docker-compose up`.
3. Open browser to http://localhost:3000 (http://localhost:3000/). The application should be up and ready for image uploads.

# Training Notebook Setup
The source code for model training and evaluation is included in the notebook server/ml/training.ipynb
1. Download and unzip the pill dataset: https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz
2. Move the unzipped pill folder inside the folder `server/ml/dataset`.
3. Make sure the notebook is run on the virtual environment kernel.
