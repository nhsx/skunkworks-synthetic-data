# MIMIC-III Preprocessing

This projec utilises data from the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/), more information is available at the link.

In order to produce a input dataset which is consistent with the requirements of this project, preprocessing was undertaken to:
- Produce a dataset of suitable size
- Update some field values and ranges for readability.

## Preprocessing steps
The preprocessing undertaken is outlined in `MIMIC_eda.ipynb` but as a simple outline:
- The OUTPUTEVENTS, ICUSTAYS, PATIENTS, ADMISSIONS, CHARTEVENTS and D_ITEMS datasets are loaded.
- The PATIENTS dataset is manipulated to bring Date of Birth (DOB) into a more realistic range so that the dataset appears more logical. This is done by adding or substracting a random percentage of the years that the original date if ahead or behind present day
- The loaded files are joined together to create a single, wide dataset. Depending on the required size of the input dataset, different numbers of entries from OUTPUTEVENTS are joined
    - For the smallest size (approx. 11k rows), a single output event is loaded per patient
    - For the middle size (approx 80k rows), 8 output events are joined per patient
    - For the largest size (approx 225k rows), a mix of 1, 2 and up to 100 events are joined per patient. This is implemented so that most patients (90%) have 1 or 2 events, while the minority have up to 100.
- From the joined dataset, ADMITTIME, DISCHTIME and CHARTTIME are adjusted in a similar manner to DOB to ensure that they are logical and make sense. This is achieved again using random proportions of time, and adheres to the following:
    - ADMITTIME must occur after DOB
    - DISCHTIME must occur after ADMITTIME, and within a sensible time horizon (implemented as 50 days)
    - CHARTTIME must occur between ADMITTIME and DISCHTIME
- Finally Age is calculated
- The file is saved down

This concludes the preprocessing performed, and the files generated here go on to become the training material for the synthetic data generation models.