# LLMxCPG - Baselines

## VulBERTa

The baseline is adopted from the implementation of the paper: VulBERTa: Simplified Source Code Pre-Training for Vulnerability Detection, which can be found [here](https://github.com/ICL-ml4csec/VulBERTa.git)


**Model weight**

- Please download the model weights [here](https://drive.google.com/file/d/1nElR1n_YMzCbGGgrJsVXn5G0juF1sJeA/view?usp=sharing)
- Then, unzip the file and put all model weight inside the following directory: `./VulBERTa/models`

**Setup**

In general, we used this version of packages when running the experiments:

 - Python 3.8.5
 - Pytorch 1.7.0
 - Transformers 4.4.1
 - Tokenizers 0.10.1
 - Libclang (any version > 12.0 should work. https://pypi.org/project/libclang/)

For an exhaustive list of all the packages, please refer to [requirements.txt](https://github.com/ICL-ml4csec/VulBERTa/blob/main/requirements.txt "requirements.txt") file.

**Running the baselines**

- To replicate the results for `VulBERTA-MLP`, execute the code in the following file: `./VulBERTa/Evaluation_VulBERTa-MLP.ipynb`
- To replicate the results for `VulBERTA-CNN`, execute the code in the following file: `./VulBERTa/Finetuning+evaluation_VulBERTa-CNN.ipynb`
- For `VulBERTA-CNN`, training the model is optional, just need to run the `TEST_ONLY` setting.