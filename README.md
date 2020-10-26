# Handwriting Determination
Code for the Paper "Synthetic Data for the Analysis of Archival Documents: Handwriting Determination" - [DICTA 2020](http://www.dicta2020.org/wp-content/uploads/2020/09/9_CameraReady.pdf).

## Installation

1. make sure you have Python (>= 3.6) installed
1. clone the Repository
1. create a virtual environment
1. install all requirements with `pip install -r requirements.txt`
1. You should be ready to go!

## Data Preparation

We provide our data generation tool (in the `data_generation` directory) and also information on how to get all
base data that we used for generation of our train dataset.
You can find more information on that [here](https://bartzi.de/research/handwriting_determination).

Before you can generate data, you first have to adapt the file `data_generation/generation_profiles.py`
and set the base path to all auxiliary files (`line 7`).
Generation can then be started by running the script `data_generation/data_generator.py`.
You can check possible option by running 
```shell script
python data_generator.py -h
```

You can also download the training data that we used for the training of our model [here](https://bartzi.de/research/handwriting_determination).

## Training of a Model

You can train a model by getting/preparing the training data.

Before training you'll have to configure everything.
In the section `[PATHS]` you have to set the paths to your `train.json` and `validation.json`.

You can then use the script `train_handwriting_determination.py` to train a model.
It could look like this:
```shell script
python train_handwriting_determination.py training config.cfg --gpu 0
```
This runs the training, saves the logs in the directory `test/training`, while setting the config to
`config.cfg` and running the training on GPU `0`.

Further options are available:
- `--log-dir`: change the directory where logs are saved
- `--gpu`: set the gpu id to use (negative valuse indicate CPU)
- `--resume`: provide path to a log dir and allow training to resume from last saved trainer snapshot.
- `--save-gradient`: plots gradients into tensorboard.

You can also inspect the train logs, using tensorboard.

## Evaluation

After you've trained a model, you can evaluate the model using the script `evaluate.py`.
You can use it like this (assuming that our model was logged to `test/training` and your data is in `/data/val.json`):
```shell script
python evaluate.py test/training HandwritingNet_ /data/val.json -g 0
```
This evaluates all trained model files that start with `HandwritingNet_` in the directory
`test/training` using the validation groundtruth in `/data/val.json`.

Further options are possible:
- `--log-name`: if you changed the default name of the log json, you need to set this here
- `-g`, `--gpu`: GPU id to use (same semantics as with training)
- `-b`, `--batch-size`: batch size to use for evaluation (batch size of 1 is recommended)
- `-e`, `--evaluation-name`: name of the resulting json file, where evaluation results are saved. Useful if you want to 
run evaluation on multiple datasets.
- `--force-reset`: force the script to run the full dataset, in case a result file already exists
- `-r`, `--render-regions`: plot decisions of the model for each inout image. Choices allow you to specify whether to plot, e.g.,
false positives, true positives, or all results.
- `--render-negatives`: render regions only renders the confidence for positive decisions, if you want to render 
confidence for negative decisions, use this flag.

# Citation

If you think our work is useful, please consider citing it:

```bibtex
@article{bartzsynthetic,
  title={Synthetic Data for the Analysis of Archival Documents: Handwriting Determination},
  author={Bartz, Christian and Seidel, Laurenz and Nguyen, Duy-Hung and Bethge, Joseph and Yang, Haojin and Meinel, Christoph}
  booktitle={2020 Digital Image Computing: Techniques and Applications (DICTA)}
}
```

# Questions and Enhancements

If you have any questions or you have something nice to contribute, please let us know by opening an
issue or pull request.
