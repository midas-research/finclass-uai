# FinCLASS
Code release for the paper titled **Modeling Financial Uncertainty with Multivariate Temporal Entropy-based Curriculums** [(link)](https://www.auai.org/uai2021/pdf/uai2021.638.preliminary.pdf), accepted at the UAI 2021 conference as a full paper.

## Environment & Installation Steps

Create an environment having Python 3.6 and install the following dependencies

```
pip3 install -r requirements.txt
```

## Contents

- `data_preprocessing_code` - Contains all the scripts for preprocessing the data, and steps to run, information about preprocessing, etc.
- `difficulty_score_code` - Contains the code and the steps to run for computing the difficulty score defined in FinCLASS framework.
- `model_training` - Contains the code for training THA-Net both with and without using the curriculum generated using FinCLASS.

## Data

Find the US S&P 500 data [here](https://github.com/yumoxu/stocknet-dataset), and the China & Hong Kong data [here](https://pan.baidu.com/s/1mhCLJJi).

## Steps
1. Preprocess the data using the scripts in `data_preprocessing_code`, following the instructions mentioned there
2. Calculate the different complexities using the scripts in `difficulty_score_code`
3. Train the model using the scripts in `model_training`, following the instructions there
