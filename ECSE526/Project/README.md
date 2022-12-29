# NLP Disaster Tweets Classifier

This project is about training a model capable of classifying a tweet as being about a distater or not. This project was conducted as final project of the ECSE526 course on Artificial Intelligence at McGill University

# Setup

To install the depencies needed for this project, you'll need to install conda, then run :
```bash
conda env create -f environment.yml
```

# Important files

Most of our work can be found in the notebook, here is a brief summary of what you can find in them :
* data_exploration.ipynb explores the data set and the pre-processing pipeline
* BERT.ipynb is about the fine-tuning of BERT, its optimization and how to create a simple ensemble prediction
* LSTM.ipynb is about the training of a LSTM network
