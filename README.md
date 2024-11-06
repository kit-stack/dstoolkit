# ML Pipeline
This is a pipeline designed to simulate a machine learning workflow. This pipeline runs on make automation, where it taps on:

```bash
make data
make features
make train
make predict
```


## Project Structure
This is how the project structure looks like:

```
├── data
│   ├── raw         		<- input data
│   │	 
│   │	
│   │	 
│   └── processed   <- Data used to develop models
│
├── models               <- Where model building is stored
│
├── submission          <-.Submission of output
│
├── test         <- Data dictionaries, manuals, etc. 
│
├── .gitignore         <- Avoids uploading data, credentials, 
│                         outputs, system files etc
│
├── config.toml               <- global config file, in this case, it is a config file for model parameters
│
├── environment.yaml          <- This is similar to requirements.txt file
│
├── Makefile            <- Master control file to run/test the pipeline
│
├── README.md          <- The top-level README for developers.
│
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data       <- Scripts to reading and writing data etc
    │   └── make_data.py
    │
    ├── explore       <- Scripts to reading and writing data etc
    │   └── exploration.py
    │
    ├── features<- Scripts to transform data from raw to intermediate
    │   └── build_features.py
    │   └── transformations.py
    │
    │
    ├── models  <- Scripts to train models and then use 
    |   |                  trained models to make predictions. 
    │   └── classifier.py
    │   └── predict_model.py
    │   └── train_model.py
    │
    │
    ├── utils  <- utility files with pre-made functions that can be called in submodules  
    |   |                   
    │   └── config.py
```

