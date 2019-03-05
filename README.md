# INTRUS
Anonymized supplementary code for NeurIPS submission. This code trains and applies a machine translation model that can generate sequences in arbitrary order

![orders](https://i.imgur.com/Bvxeqv2.png)

# What do i need to run it?
* A machine with some CPU (preferably 4+) and at least one GPU
* The optimal performance is reached when running on 8 GPUs
* Some popular Linux x64 distribution
  * Tested on Ubuntu16.04, should work fine on any popular linux64 and even MacOS;
  * Windows and x32 systems may require heavy wizardry to run;
  * When in doubt, use Docker, preferably GPU-enabled (i.e. nvidia-docker)

# How do I run it?
1. __Setup environment__
 * Clone or download this repo. `cd` yourself to it's root directory.
 * Get a python distribution. [Anaconda](https://www.anaconda.com/) works fine.
 * Install packages from `requirements.txt`
 
2. __Prepare data__
 * Grab the WMT English-Russian dataset from http://statmt.org/ (or another language of your choosing)
 * Tokenize it with [mosestokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)  or any other reasonable tokenizer. It is also recommended that you lowercase the data.
 * Learn and apply BPE with [subword-nmt](https://github.com/rsennrich/subword-nmt)
 * You can find example preprocessing pipelines [here](https://github.com/pytorch/fairseq/tree/c778a31e2b6ae4d089d9a213ba023140438725b2/examples/translation).
 
3. __Run jupyter notebook__
 * All the training notebooks are in the `./notebooks/` folder
 * Before you run the first cell, optionally set `%env CUDA_VISIBLE_DEVICES=###` to devices that you plan to use.
 * Follow the code as it loads data, trains model and reports training progress.
 * __NOTE:__ The BLEU metric measured in the notebook is not the one used for evaluation. See [sacrebleu](https://pypi.org/project/sacrebleu/).
