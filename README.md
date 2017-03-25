## ann_bmi203

Chris Mathy
For UCSF Course BMI 203, Winter 2017
Final Project
Implementation of an artificial neural network for predicting TF binding sites.


[![Build
Status](https://travis-ci.org/cjmathy/ann_bmi203.svg?branch=master)](https://travis-ci.org/cjmathy/ann_bmi203)

## usage

To use the package, first run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `ann/__main__.py`) can be run as follows

```
python -m ann
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
