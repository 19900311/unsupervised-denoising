# Unsupervised Denoising
## Structure
* main.ipynb - entry point, this is where the data is loaded and the training loop executed.
* datasets.py - defines the pytorch dataset for Spineweb.
* train_adn.py - defines the training loop for the model.
* adn.py - defines the ADN model.
* radon_adn.py - defines the Radon enhanced ADN model.
* components.py - defines the components that are used to build the models. The discriminator is defined here too.
* blocks.py - defines the blocks that are used to build the components.

## Datasets
* [Spineweb](https://drive.google.com/drive/folders/1MsNWQgAqLMBWsmUU4EklbMzc2uMhGOZ3)

## Original Papers
* [ADN](https://arxiv.org/abs/1908.01104)
