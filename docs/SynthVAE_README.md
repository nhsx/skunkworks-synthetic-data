# Synthetic Data Exploration: Variational Autoencoders
## NHSX Analytics Unit - PhD Internship Project

This part of the repository holds code which is derived from the [NHSX Analytics Unit PhD internship project (previously known as Synthetic Data Generation - VAE)](https://nhsx.github.io/nhsx-internship-projects/synthetic-data-exploration-vae/) which seeks to contextualise and investigate the potential use of Variational AutoEncoders (VAEs) for synthetic health data generation undertaken by Dominic Danks.

The code in this folder is a derivative of this work, with additional scripts replacing some elements on the original functionality. This implementation of the code has moved in a more functional direction, using the SynthVAE code to produce scripts for generating synthetic data with a range of architectures.


_**Note:** No data, public or private are shared in this repository._

**N.B.** A modified copy of [Opacus](https://github.com/pytorch/opacus) (v0.14.0), a library for training PyTorch models with differential privacy, is contained within the repository. See the [model card](./src/synthetic_data_generation/SynthVAE/model_card.md) for more details.

### Built With

[![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
- [PyTorch](https://github.com/pytorch)
- [SDV](https://github.com/sdv-dev/SDV)
- [Opacus](https://github.com/pytorch/opacus)

### Getting started

Getting started with SynthVAE is included as part of the setup of the wider skunkworks-synthetic-data project.

### Usage

SynthVAE functionality has been reproduced in [train_generate_synthetic](../src/synthetic_data_generation/train_generate_synthetic.py), using the framework outlined in the origin repo, with some key changes:
* The script has been updated to take MIMIC-III dataset, namely to take datetime fields. This included ensuring that fields are casting fields to the correct data type
* Some additional data processing has been included, such as adding a ROW_ID and rounding SUBJECT_ID

#### Dataset
The primary dataset used is the the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) accessed via Physio.net, where an access request is needed.

The dataset used for the test pipeline is the [Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) dataset](https://biostat.app.vumc.org/wiki/Main/SupportDesc) accessed via the [pycox](https://github.com/havakv/pycox) python library.


#### Caveats

During development of this repo, it was identified that SynthVAE appears to have issues converging when several categorical columns are used in conjunction with continuous and date problems. This manifested as synthetic categorical fields containing the same value/nearly always the same value for all rows of the synthetic dataset. This problem was mitigating by experimenting with different combinations and proportions of categorical and continuous variables, however if this work was extended this might be an issue to consider.

### Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [analytics-unit@nhsx.nhs.uk](mailto:analytics-unit@nhsx.nhs.uk).

<!-- ### Acknowledgements -->
