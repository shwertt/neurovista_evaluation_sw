# neurovista_evaluation_sw
Code for evaluation on full neurovista trial data following the [instructions](https://github.com/epilepsyecosystem/CodeEvaluationDocs) (commit 20e6f0f, dated 16/06/2020). 

This project has been cloned from
https://github.com/MatthiasEb/neurovista_evaluation (a colleague of mine, both
working at the Dresden University of Technology, Germany) and since adapted to
feature networks consisting of 1D-Convolutions on the raw time series.

## Settings
Settings can be adjusted in SETTINGS.json

# ## Using a virtual environment
# ### requirements
# 1. python3
# 2. cuda toolbox 10, nvidia-drivers
#  
# ### installation
# Install requirements by running:
# 
# `pip install -r requirements.txt`
# 
# 
# ### execution
# Run training by executing:
# 
# `python3 run.py`
# 
# ## Using Docker
# ### requirements
# Tested with Docker version 19.03.6, build 369ce74a3c on Ubuntu 18.04
# 
# ### Build Image
# Build Docker Image by running:
# 
# `docker build --tag nv1x16_eval .`
# 
# ### Execution
# Specify the directory of your data segments by
# 
# `export DATA_DIR=YOUR_DATA_DIRECTORY`,
# 
# replacing `YOUR_DATA_DIRECTORY` with your specific directory.
# 
# Run training by executing
# 
# `docker run --gpus 1 -v $PWD:/code -v /$DATA_DIR:/$DATA_DIR:ro nv1x16_eval python ./run.py`

## Using Singularity

Singularity recipe is included. SingularityHub URI of the image is shwertt/neurovista_evaluation_sw:nv_eval.

## Remarks

You should use a GPU for training. I used an RTX 2080 Ti.
If you use a GPU with much less RAM, you might have to reduce the batch size, I did not try that though.
I ran the code with run_on_contest_data=1, the results seemed to be comparable
to the results from MatthiasEb
(https://github.com/MatthiasEb/neurovista_evaluation).
I did try to run it within a singularity container.
Do not hesitate to contact me if you run into problems, have any questions or remarks.

### Algorithm
This is a pretty naive approach on a 1D-Convolution Deep Neural Network, applied to the raw time series. 
The network
expects standardized 15 s segments, sampled at 200 Hz. 
tensorflow.keras (2.0.1) was used as Deep Learning API. 
I did a few testruns with different models on the contest data, see below:

```
{'run_on_contest_data': 1, 'mode': 3, 'pat': '1-3', 'subtract_mean': 1, 'model': './trains', 'feat': './features', 'CSV': './CSV', 'solutions': './solutions'}
./solutions/contest_data_solution_shwertt_mode3_20201001_resnet1d.csv
Global roc auc: 0.6555
Global Public roc auc: 0.6627
Global Private roc auc: 0.6684
Patient 1 roc auc: 0.2143
Patient 2 roc auc: 0.6700
Patient 3 roc auc: 0.7326
Patient 1 Public roc auc: 0.2111
Patient 1 Private roc auc: 0.2182
Patient 2 Public roc auc: 0.6695
Patient 2 Private roc auc: 0.6766
Patient 3 Public roc auc: 0.7618
Patient 3 Private roc auc: 0.7208
```

However, considerable variations are conceivable.


### Implementation
Loading the original (~ 400 Hz) .mat files, resampling to 200 Hz,
standardizing (optionally, if `subtract_mean==1`), splitting them in 15 s
segments is done asynchronously on the fly by the dataloader in 5 different
threads. The 15s Segments are enqueued in a buffer with the size of 400
10-min-sequences, implemented as a tf.queue.RandomShuffleQueue. The data is
therefore dequeued in random order, although not perfectly uniformly
shuffeled, depending on the buffer size and the size of the data set. The
intention was to ensure a reasonably shuffeled training set of 15 s segments
while minimizing IO, working on the .mat files and having the possibility for
standardization. If the IO-Bandwidth of the filesystem is reasonably high,
this should not slow down the training too much. 

If run_on_contest_data==1, 3 networks (one for each patient) are trained and evaluated individually. 
Subsequently, the solution file is concatenated.
