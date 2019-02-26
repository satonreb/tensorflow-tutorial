# Python environment set-up

This chapter will describe how to prepare your computer in order to run provided scripts.

First of all, ensure that you have installed **conda** package manager. If you have not done so, please follows the steps described [here](https://conda.io/docs/user-guide/install/download.html).

> Note: Python version required to run the provided scripts is **3.6.2** and for TensorFlow it is **1.3.0**.

Further, git clone all the scripts that are available in [this repository](https://github.com/satonreb/machine-learning-using-tensorflow) and then using the command line to switch to the location of the cloned files. In `Scripts` directory you should see `environment.yml` file that specifies the name of the conda environment and packages that will be installed. To create the environment run the following command while in `Scripts` directory:

```bash
conda env create -f environment.yml
```

Additional information on **conda** package manager and commands can be found [here](https://conda.io/docs/).

This tutorial does not require GPU and for that reason, only CPU version of TensorFlow is installed. If you wish to install the GPU version, first, follow the steps regarding CUDA set up on [TensorFlow website](https://www.tensorflow.org/install/). After you have successfully installed CUDA and all associated programs, modify `environment.yml` by replacing `tensorflow` with `tensorflow-gpu` and then run the command shown above.

> On the date of writing the [website](https://www.tensorflow.org/install/) has a misleading version number for **cuDNN** that should be **v6** rather than **v5.1**.

After the environment setup is complete, to activate it on _Unix_ systems run the following:

```bash
source activate tf_tutorial
```

and if you running _Windows_ use:

```bash
activate tf_tutorial
```

Here **tf\_tutorial** is the default environment name that is specified in `environment.yml` file. Hence, if you have replaced it, replace it in the commands above.

While in the environment, you can run any Python command or/and script but it will use only packages that are available in the envirment. This tutorial is using the following packages:

* [python](https://www.python.org/)
* [matplotlib](https://matplotlib.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [numpy](http://www.numpy.org/)
* [pandas](http://pandas.pydata.org/)
* [pip](https://pip.pypa.io/en/stable/)
* [tensorflow](https://www.tensorflow.org/)

> Note: It is advisable to create a separate environment and `environment.yml` files for each project. For more information on **conda** environment management see [here](https://conda.io/docs/commands.html#conda-environment-commands).

[Next chapter](introduction-to-tensorflow.md) is going give a very brief introduction into TensorFlow, if you wish to return to previous chapter press [here](./).

## Code

* [environment.yml](https://github.com/satonreb/machine-learning-using-tensorflow/blob/master/scripts/environment.yml)

