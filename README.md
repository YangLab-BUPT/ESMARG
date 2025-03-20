# ESMARG

ESMARG is a new model designed based on ESM-2, consisting of four modules: ESMARG-1, ESMARG-2, ESMARG-3, and ESMARG-4. 

- **ESMARG-1**: This module identifies whether a sequence is classified as an Antibiotic Resistance Gene (ARG) or non-ARG.

- **ESMARG-2**: This module categorizes the identified sequences into specific groups.

- **ESMARG-3**: This module assesses and determines the resistance mechanisms of the ARGs.

- **ESMARG-4**：This module uses DIAMOND to align functions of the ARGs.

## Preparation

ESMARG uses the ESM-2 model for feature extraction. Please visit [ESM-2 GitHub Repository](https://github.com/facebookresearch/esm) to obtain the ESM-2 model. Use the following command to perform feature extraction:

```bash
   python scripts/extract.py
```
## Installation
ESMARG using python3.11 can install the environment using the following instructions, which contain the configuration of the esm2 environment
```bash
   conda env create -f environment.yml
```
## test
You can use the prediction files in the test folder to perform testing:
  ### For ESMARG-1, to identify ARG sequences, use:
```bash
     python predictp.py
```
  ### For ESMARG-2, to categorize ARG sequences, use:
```bash
     python predictmult.py
```
  ### For ESMARG-3, to determine the resistance mechanisms of ARG sequences, use:
```bash
     python predictmachism.py
```

# train
You can retrain ESMARG using the files in the model directory. Please note that the current files do not contain the feature vectors extracted using ESM-2. Therefore, you need to perform feature extraction on the corresponding .fasta files in the data folder using ESM-2 first.

## About
If you use ESMARG in published research, please cite:

## License
ESMARG is under the MIT licence. However, please take a look at te comercial restrictions of the databases used during the mining process (CARD, SARG, DeepARG and UniProt).

## Contact
If need any asistance please contact: lihm@bupt.edu.cn
