# Music categorization project (APS360)

The project aims at training a model for music categorization. It will accept a music track given in input and will return the category of the music (pop, rock, ...) among the ones the model learnt.

Below is the list of all categories taken into account in this project:

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

## Repository structure

- `requirements.txt` allows you to install all dependencies to the project. See 'Set up' section in order to install them.

- `baseline_model` is the directory containing all python files and notebooks regarding the baseline model.

- `Data` folder contains 2 `csv` files used for training baseline model.

## Set up

Whether you want to use the notebooks or the python files, you need to resolve all dependencies. You can install them either in your virtual environment or Python base environment with the following command: `pip install -r requirements.txt`. 

If you want to use notebooks defined in this repository, you can upload them on *Google Collab* or you can run a Jupyter notebook server locally. You can install Jupyter by following [the official documentation](https://docs.jupyter.org/en/latest/install/notebook-classic.html).
