Character-level language modelling with RNN
========

A RNN character-level language model, implemented with Tensorflow in Python. [This blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is a great introduction to RNN character-level language model. [Here](https://github.com/jcjohnson/torch-rnn) is a torch implementation of the same model.

## Dependency
- python2.6+
- numpy
- tensorflow

## Usage

#### Training
Train the model with default hyperparameters:

    python train.py

Or, train with self-specified hyperparameters:

    python train.py --model gru --batch_size 60 --hidden_size 100 --init_lr 0.05 --num_epochs 25 --dropout 0.5

By default, a 2-layer LSTM model will be used, and will be unrolled for 50 steps during gradient descent.

For a complete list of parameters, please consult `model.py` and `train.py`.

#### Sampling
Sampling is the process to generate sampled text from a trained model. To perform sampling:

    python sample.py --max_length 1000

## License

MIT