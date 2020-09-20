# Vehicle Type Recognition
This repo contains code for training models and predicting vehicle classes.

# Run training
Download competition's dataset https://www.kaggle.com/c/vehicle/data

Next, for example, call: `python train.py --batch_size 50 --epochs 15`

# Run prediction
Call `python predict.py`

One may want to change IMG_HEIGHT and IMG_WIDTH in arguments to choose appropriate image dimensions.

# Ensembling/Blending

`ensembling.py` takes several .csv Kaggle submission files and produces ensembled submission based on weights of the individual submissions.

`blending.py` merges .csv files with probabilities of several models and makes .csv Kaggle submission file
