# aidtep


![Python](https://github.com/gitpython-developers/GitPython/workflows/Python%20package/badge.svg)
[![Release](https://img.shields.io/github/v/release/zeromicro/go-zero.svg?style=flat-square)](https://github.com/YannLee1208/aidtep)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Description
AI enhanced Digital Twin Engineering Platform (AIDTEP)

## Project Structure
```shell
aidtep/
│
├── README.md              # Project Description
├── environment.yaml       # Conda environment dependency file 
├── aidtep/                
│   ├── data_processed     # data preprocess 
│   ├── extract_basis      # reduce basis methods
│   ├── forward_problem    # forward problem methods
│   ├── inverse_problem    # inverse problem methods  
│   ├── launcher           # entry points
│   ├── utils              # utils  
├── bin/                   # shell scripts
├── config/                # configuration
├── data/                   
│   ├── raw/               # raw data
│   └── processed/         # processed data
│   └── model_weights/     # offline training model weights   
└── docs/                  # documentations
├── service/               # web service
├── simulation/            # call simulator
├── notebooks/             # Jupyter Notebooks
└── tests/                 # unit tests and ensemble tests
```

## Installation

### Using Conda
* Create and activate conda enviroment
    ```sh
    conda env create -f environment.yaml
    conda activate my_env
    ```


## Usage


```python
# example
```





## License

This project is based on MIT License. See [LICENSE](https://github.com/YannLee1208/aidtep/blob/master/LICENSE) file.