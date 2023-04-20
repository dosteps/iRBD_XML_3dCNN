# iRBD_XML_3dCNN

This repository contains the code used in the paper titled "**Spatiotemporal characteristics of abnormal cortical activities of isolated REM sleep behavior disorder revealed by an explainable machine learning approach using three-dimensional convolutional neural network**". The paper presents a novel approach to to identify the spatiotemporal characteristics of brain activity related to visuospatial attentional impairment in iRBD patients using three-dimensional convolutional neural network.

## Requirements

To run the code in this repository, you will need the following:

- Python 3.7 or later
- PyTorch
- WandB account (create [here](https://wandb.ai/)) and connection to use logging feature

Recommended computer specifications:

- CPU: Intel Core i5 or higher
- RAM: 64GB or more (256GB for machine learning (SVM))
- GPU: Four GPU graphic cards, NVIDIA GeForce GTX 1080 or higher with at least 8GB of VRAM
- Disk Space: 64GB or more (for storing datasets and models)
- Operating System: Linux Ubuntu 20.04 (recommended), Windows 10

## Installation

To install the required Python libraries, simply run:
```
pip install -r requirements.txt
```
To create a virtual environment for this project, we recommend using Anaconda as the code was developed and tested in an Anaconda environment. You can create a new environment using the command `conda create --name envname python=3.9`, replacing envname with a name of your choice. Once you have created and activated your virtual environment, you can run the command `pip install -r requirements.txt` to install the required libraries in the virtual environment. Alternatively, you can use other virtual environment tools such as 'venv', 'virtualenv', or 'pipenv', depending on your preference and project requirements.

By creating a virtual environment, you can ensure that your project dependencies are isolated and version-controlled, which can help avoid conflicts and ensure reproducibility. It also allows you to easily share your project with others without worrying about compatibility issues or dependency conflicts

## Usage

### Data Description and Access Information
We utilized the WandB platform to log our training results and provide more detailed information. You can view our training results, including loss and accuracy graphs, as well as trained models by accessing the link to our WandB project:
([Main Results](https://wandb.ai/dosteps/Posner_iRBD_XML_3dCNN), [Supplementary Results](https://wandb.ai/nelab/Posner_iRBD_XML_3dCNN_verification))

The data presented in this study are not publicly available because they contain information that can compromise the privacy of the research participants. Some of the data may be available from the corresponding authors upon request.

Yet, we have included options in our code ('-dt "demo"' in the 'main.py' file) so that you can still test the functionality of the code using the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. When executing the code presented in this repository, the dataset is modified to classify images into two superclasses: terrestrial animals [3, 4, 5, 6, 7], and others [0, 1, 2, 8, 9]. you can use the dataset by running the code with the option '-dt "demo"'.

### Python

The code in this repository can be used to reproduce the results presented in the paper. To do so, simply run the Python scripts provided.

We highly recommend using a Linux environment to run the code. Although it can also be executed in a Windows environment, we cannot guarantee its full functionality.

The entry point for running the different stages of the pipeline is the main.py file.
To run the Python code, navigate to the `python` directory and run.
Here are the available stages and the corresponding script file names:
- Data preparation:
    - `!python Methods/1_split_dataset.py`
    - `!python Methods/2_data_preparation.py`
- Pretraining: `!python Methods/train_ercd_artifacts.py -pipe "init_train"`
- Fine-tuning: `!python Methods/finetune_ercd_artifacts.py -pipe "init_train"`
- Evaluation: `!python Methods/*_ercd_artifacts.py -pipe "eval"`
- Interpretation: `!python Methods/*_ercd_artifacts.py -pipe "explain"`
- Visualization: `!python Methods/*_ercd_artifacts.py -pipe "plot"`

Optional arguments for each stage can be found by running '*.py -h' command.

You can run the script for a specific stage by providing the corresponding command-line arguments. For example, to run the training stage with the demo dataset, you can use the following command:
```
python Methods/train_ercd_artifacts.py -pipe "init_train" -dt "demo" -entity "username"
```


## Citation

If you find this code useful in your own work, please consider citing our paper:

Thank you!

## License

The code is available for use under the GNU General Public License version 3 (GPL-3), which allows for free use, modification, and distribution of the code as long as any modifications or derivative works are also shared under the same terms.


