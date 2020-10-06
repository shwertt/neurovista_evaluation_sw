# neurovista_evaluation_sw
Code for evaluation on full neurovista trial data following the [instructions](https://github.com/epilepsyecosystem/CodeEvaluationDocs) (commit 20e6f0f, dated 16/06/2020). 

This project has been cloned from
https://github.com/MatthiasEb/neurovista_evaluation (a colleague of mine, both
working at Dresden University of Technology, Germany) and since adapted to
feature networks consisting of 1D-Convolutions on the raw time series.

## Settings
Settings can be adjusted in SETTINGS.json

## Using Singularity

This project has been thoroughly testey with the Singularity recipe
`Singularity.nv_eval` that is included in the repository.
The SingularityHub URI of my image is `shwertt/neurovista_evaluation_sw:nv_eval`.

### Installing Singularity

If you need to install singularity on your work station, it did not suffice
for me to just do a `sudo apt install singularity-container` because this
install a singularity version 2. But https://singularity-hub.org/ deploys
singularity version 3 images.

Trying to pull the shub image with singularity version 2 generates the
following error:

```
ERROR  : Unknown image format/type: /mnt/ieecad/s9759051/Downloads/tmp/shwertt-neurovista_evaluation_sw-main-nv_eval
ABORT  : Retval = 255
```

In order to install singularity version 3 locally, I followed the installation
instructions from https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/Container

This involved:

1. Check if go is installed by executing `go version`. If it is not installed, get it with:

    `wget https://storage.googleapis.com/golang/getgo/installer_linux && chmod +x installer_linux && ./installer_linux && source $HOME/.bash_profile`

2. Install Singularity by cloning the singularity repo

    `mkdir -p ${GOPATH}/src/github.com/sylabs && cd ${GOPATH}/src/github.com/sylabs && git clone https://github.com/sylabs/singularity.git && cd singularity`

3. Checkout the singularity version you want (see the Github Releases page for available releases), e.g.

    `git checkout v3.6.3`

4. Check the environment variables for `go` in `~/.bash_profile`

    I had to change all references from `/home/s9759051` to `/mnt/ieecad/s9759051`
    otherwise the installer could not find the correct environment for the needed
    modules.

5. Build and install

    `cd ${GOPATH}/src/github.com/sylabs/singularity && ./mconfig && cd ./builddir && make && sudo make install`


## Interacting with the Singularity container

Pull the container from Singularity Hub (once)

    singularity pull --name nv_eval_shwertt.sif shub://shwertt/neurovista_evaluation_sw:nv_eval

Set `mode=1` in `SETTINGS.json`, then start a shell in the container with:

    singularity shell --contain -B /PATH/TO/CLONE/OF/THIS/GITHUB/PROJECT/neurovista_evaluation_sw:$HOME -B /scratch:/scratch --nv nv_eval_shwertt.sif

This mounts this github project to the `Home` folder inside singularity and
binds the local scratch folder to the scratch folder inside singularity. If
the data files are not inside `/scratch`, please modify this bind statement.
CUDA support and access to the GPU is achieved with the `--nv` flag.

Now while inside the Singularity container, execute:

    export CUDA_VISIBLE_DEVICES=0 && python3 run.py

Please specify your `CUDA_VISIBLE_DEVICES` according to the available resources
of the supercomputer.

After training is complete, you can test the model by closing the container
with `exit`, modify `SETTINGS.json` to switch to `mode=3` and start the container again with:

    singularity shell --contain -B /PATH/TO/CLONE/OF/THIS/GITHUB/PROJECT/neurovista_evaluation_sw:$HOME -B /scratch:/scratch --nv nv_eval_shwertt.sif

then execute again:

    export CUDA_VISIBLE_DEVICES=0 && python3 run.py

Please find the solution file under
`solutions/contest_data_solution_shwertt_mode3.csv`.


## Remarks

You should use a GPU for training. I did use an RTX 2080 Ti.
If you use a GPU with much less RAM, you might have to reduce the batch size, I did not try that though.
When I ran the code with `run_on_contest_data=1`, the results seemed to be comparable
to the results from [MatthiasEb](https://github.com/MatthiasEb/neurovista_evaluation).
The singularity container works just fine, in case you run into problems, have any questions or remarks,
please do not hesitate to contact me.


### Algorithm
This approach uses a Residual Network (Resnet) based on 1D-Convolutions, applied to the raw time series. 
The residual block consist of this stack:

```
def residual_block(X, kernels, size):
    out = tf.keras.layers.Conv1D(kernels, size, padding='same')(X)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv1D(kernels, size, padding='same')(out)
    out = tf.keras.layers.add([X, out])
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool1D(pool_size=5, strides=3)(out)
    return out
```

The deep neural network then consists of six residual blocks followed by a
global 1D-MaxPooling layer and a dense layer. The model is optimized with 
Adam, which uses a learning rate of 0.001.

The model expects standardized 15 s segments of data, sampled at 200 Hz. 
tensorflow.keras (2.0.1) was used as Deep Learning API. 
After several testruns with different models on the contest data, I chose the
resnet1d architecture that has been described above.

With it I have achieved the following results:

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
10-min-sequences, implemented as a `tf.queue.RandomShuffleQueue`. The data is
therefore dequeued in random order, although not perfectly uniformly
shuffeled, depending on the buffer size and the size of the data set. The
intention was to ensure a reasonably shuffeled training set of 15 s segments
while minimizing IO, working on the .mat files and having the possibility for
standardization. If the IO-Bandwidth of the filesystem is reasonably high,
this should not slow down the training too much. 

If `run_on_contest_data==1`, 3 networks (one for each patient) are trained and evaluated individually. 
Subsequently, the solution file is concatenated.
