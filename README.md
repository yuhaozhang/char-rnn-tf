Character-level language modelling with RNN
========

A RNN character-level language model, implemented with Tensorflow in Python. [This blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) is a great introduction to RNN character-level language model. [Here](https://github.com/jcjohnson/torch-rnn) is a torch implementation of the same model.

## Dependency
- python2.6+
- numpy
- tensorflow 0.7+

## Usage

#### Small Shakespeare Data

The data is provided in this [repository](https://github.com/jcjohnson/torch-rnn).

##### Training
Train the model with default hyperparameters:

    python train.py

Or, train with self-specified hyperparameters:

    python train.py --model gru --batch_size 60 --hidden_size 100 --init_lr 0.05 --num_epochs 25 --dropout 0.5

By default, a 2-layer LSTM model will be used, and will be unrolled for 50 steps during gradient descent.

For a complete list of parameters, please consult `model.py` and `train.py`.

##### Sampling
Sampling is the process to generate sampled text from a trained model. To perform sampling:

    python sample.py --max_length 1000


#### Wikipedia Article Samples

The data is from [wikipedia dump](https://dumps.wikimedia.org/backup-index.html), and is cleaned by myself.

To train the model on the sample wikipedia article data, we need slightly different parameters:

    python train.py --data_dir data/wikipedia/ --data_file enwiki-articles-sample.xml --vocab_size 203 --hidden_size 256 --batch_size 128

To sample from the trained model, we need to set the model to the same parameter setting:
    
    python sample.py --data_dir data/wikipedia/ --vocab_size 203 --hidden_size 256 --batch_size 128 --start_with "<ref" --max_length 2000


## Example Sampling Output:

After training on the data for 30 epochs, the models generate the following sampling results:

#### Shakespeare

    ROMEO:
    Saint me: disperature so none.

    CAMILLO:
    Al you so beating they amen thy truthfring man
    And far own fear.
    Go, my leave revet you, live me noble from juster!
    I have that give these men joy as are in defubject.

    HENRY PERCY:
    That he all ask'd court? who, being costing. Stather's corrant more
    with reputy to a vastory soft unherity.
    Our with kins from his na talk not! is Busfort:
    From death's lady! hath kneal,-trains hot peace,
    Be love the kinglies.

    PETER:
    Who hath Garuems were, but the birth,
    Than good lord!

#### Wikipedia
    [[Category:Tideranus]]
    [[Category:Scales at the 0940 in Portigo]]
    [[Category:Pubself in sponsors for, for handled the based trainager|Fighter Ulmation Court of Couge]]</text>
          <sha1>bjcn02zyb9hzgps22fbrvqfmiwzdy27l</sha1>
        </revision>
      </page>
      <page>
        <title>Bandelleha Director Commons of Pat Rubyaportfland</title>
        <ns>0</ns>
        <id>7797276</id>
        <revision>
          <id>647031804</id>
          <parentid>656267067</parentid>
          <timestamp>2013-07-09T14:59:11Z</timestamp>
          <contributor>
            <username>Revilities</username>
            <id>16562568</id>
          </contributor>
          <minor />
          <comment>[[Wikipedia:Wikidata|Wikidata|Wikidata]] leaded delined to 11 &ndacre discussionship groud in the [[Barlomawa]], and wons, Marino Regionals, Source. ([[Green]].

We can see that the RNN actually learns to generate all the indentations and learns to open and close XML tags and brackets in most cases!

## License

MIT