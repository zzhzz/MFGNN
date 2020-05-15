# MFGNN

## Introduction
This repository contains the codes and pretrained model for tasks mentioned in MFGNN.


## Usage
For generating datas, please clone and run [Preparation](https://github.com/zzhzz/MFGNNPreparation) first.
Take CodeChef experiment as the example,
Clone this repository with `git clone https://github.com/zzhzz/MFGNN`
`cd CodeChef`
Adjust the settings in `config.py`. If you want to run `SUB`, change the value of `prob` into `subinc`. If you want to train a new model, set `test_mode=False`.
`python model.py`
