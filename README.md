# DBP391 Project 
This is the implementation of our project for subject DBP391 - FPT University:

[An Improvement of Graph Neural Network for Multi-
behavior Recommendation](https://drive.google.com/file/d/16ri78dOC1AdplyAHyw7FqkIelfmEJWPY/view)

## Project member:
- Nguyen Bao Phuoc
- Duong Thuy Trang
- Instructor: Phan Duy Hung

## Requirement

Create virtual enviroment and download necessary library list in requirements.txt:
```
python -m venv venv
cd venv/Scripts
activate
cd ../..
pip install requirements.txt
```

## Create and preprocess data

To be able to run code, create a directory in Dataset folder name 'taobao'
and a dataframe name 'taobao.csv' in that folder and then run preprocess_data.py to prepare data for training: 

```
cd Dataset
python preprocess_data.py
```

## Training

To training MF model, run:

```
python main.py --model_name MF
```

To training graph-based model, run:

```
python main.py
```

*Note: See more hyperparameter in main.py to tuning model if you want*

## Reference
[List in our project report](https://drive.google.com/file/d/16ri78dOC1AdplyAHyw7FqkIelfmEJWPY/view)






