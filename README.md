# How to Use

## Download dataset

You can download our resources in [Google Drive]() or via Terminal.
```
wget 
```
After downloading, you need to unzip it into the root directory of your project.

## Install requirements

Create virtualenv:
```
python -m venv BatikEnv
BatikEnv/Scripts/activate
```
Install all libraries in .txt file
```
pip install -r requirements.txt
```

## Run model

Before you run, you need to make some directories, such as "Checkpoint" to save the .h5 ModelCheckpoint.
```
mkdir Checkpoint
```
After that, you can easily run the .ipynb file.

## Convert Model to JSON

You can convert the model either one of the .h5 file in Checkpoint folder or "vgg19_model.h5".
```
tensorflowjs_converter --input_format=keras ./vgg19_model.h5 ./
```
Finally, you can zip all of weights files (.bin files) and "model.json"
```
zip batik_model_vgg19.zip *.bin model.json
```
Congratulations, now you can deploy the model into Browser-based.s