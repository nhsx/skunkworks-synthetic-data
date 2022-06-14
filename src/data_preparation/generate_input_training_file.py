import pandas as pd
import numpy as np
from random import randrange, seed

def load_mimic_data():
    admissions = pd.read_csv('data/01_raw/admissions.csv.gz')
    chartevents = pd.read_csv('data/01_raw/CHARTEVENTS.csv.gz',nrows=10000000)
    icustays = pd.read_csv('data/01_raw/ICUSTAYS.csv.gz')
    items = pd.read_csv('data/01_raw/D_ITEMS.csv.gz')
    outputevents = pd.read_csv('data/01_raw/OUTPUTEVENTS.csv.gz')
    patients = pd.read_csv('data/01_raw/PATIENTS.csv.gz')   

    return admissions, chartevents, icustays, items, outputevents, patients


def preproc_patients(patients: pd.DataFrame):
    seed(2021)
    ## Augment patient DOBs
    new_dobs = []
    dob_offset = []
    for index, row in patients.iterrows():
        years_diff_behind = len(pd.date_range(start=pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime('2021-12-01 00:00:00',format='%Y-%m-%d %H:%M:%S'),freq='Y'))
        years_diff_ahead = len(pd.date_range(start=pd.to_datetime('2021-12-01 00:00:00',format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S'),freq='Y'))
        
        if (years_diff_behind != 0) and (pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S') < pd.to_datetime('1930-01-01 00:00:00',format='%Y-%m-%d %H:%M:%S')):
            num_years = randrange(years_diff_behind - 80, years_diff_behind - 40)
            new_dobs.append(pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S') + pd.DateOffset(years=num_years))     
            dob_offset.append(num_years)
        elif (years_diff_ahead != 0) and (pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S') > pd.to_datetime('2021-12-01 00:00:00',format='%Y-%m-%d %H:%M:%S')):
            num_years = randrange(years_diff_ahead + 30, years_diff_ahead + 50)
            new_dobs.append(pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S') - pd.DateOffset(years=num_years))
            dob_offset.append(-num_years)
        else:
            new_dobs.append(pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S'))
            dob_offset.append(0)
    patients['DOB'] = new_dobs
    patients['DOB_offset'] = dob_offset
    return patients

def join_table(admissions: pd.DataFrame, patients: pd.DataFrame, icustays: pd.DataFrame):
    table_one = admissions[['ROW_ID','SUBJECT_ID','ETHNICITY','ADMITTIME','DISCHTIME','DISCHARGE_LOCATION']]
    table_one = table_one.merge(patients[['SUBJECT_ID','GENDER','DOB']])
    table_one = table_one.merge(icustays[['SUBJECT_ID','ICUSTAY_ID','FIRST_CAREUNIT']])

    return table_one

def generate_11k_dataset(admissions: pd.DataFrame, patients: pd.DataFrame, icustays: pd.DataFrame, chartevents: pd.DataFrame, items: pd.DataFrame):
    seed(2021)
    # Generate small input data file
    table_one = join_table(admissions, patients, icustays)
    one_per_pat = chartevents.drop_duplicates(subset=['SUBJECT_ID','ICUSTAY_ID'])
    table_one = table_one.merge(one_per_pat[['SUBJECT_ID','ICUSTAY_ID','CHARTTIME','ITEMID','VALUE','VALUEUOM']],on=['SUBJECT_ID','ICUSTAY_ID'])

    new_admits = []
    new_dischs = []
    new_chart = []

    for index, row in table_one.iterrows():
        admit_min = len(pd.date_range(start=pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime('2021-12-01 00:00:00',format='%Y-%m-%d %H:%M:%S'),freq='D'))
        stay_len = len(pd.date_range(start=pd.to_datetime(row['ADMITTIME'],format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime(row['DISCHTIME'],format='%Y-%m-%d %H:%M:%S'),freq='S'))

        num_days_admit = randrange(np.round(admit_min*0.25,0),np.round(admit_min*0.9,0)+5)
        num_days_disch = randrange(0,50)
        num_secs_chart = randrange(np.round(stay_len*0.01,0)+1,np.round(stay_len*0.99,0)+10)
        new_admit_date = pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S') + pd.DateOffset(days=num_days_admit)
        new_admits.append(new_admit_date)
        new_dischs.append(new_admit_date + pd.DateOffset(days=num_days_disch))
        new_chart.append(new_admit_date + pd.DateOffset(seconds=num_secs_chart))

    table_one['ADMITTIME'] = new_admits
    table_one['DISCHTIME'] = new_dischs
    table_one['CHARTTIME'] = new_chart

    table_one = table_one[(pd.to_datetime(table_one.ADMITTIME) < pd.to_datetime(table_one.CHARTTIME)) & (pd.to_datetime(table_one.DISCHTIME) > pd.to_datetime(table_one.CHARTTIME))]
    table_one = table_one.merge(items[['ITEMID','LABEL']],on=['ITEMID'])
    table_one.drop(['ICUSTAY_ID','ITEMID'],axis=1,inplace=True)
    age_calc = pd.Timestamp('2021-12-01 00:00:00')
    table_one['DOB'] = pd.to_datetime(table_one['DOB'], format='%Y-%m-%d %H:%M:%S')
    table_one['age'] = (age_calc - table_one['DOB']).astype('<m8[Y]')
    table_one = table_one.groupby('SUBJECT_ID').head(4)
    print(f"Small input table saved, number of columns:  {table_one.shape[1]}, number of rows: {table_one.shape[0]}")
    #table_one.to_csv(f'data/05_model_input/table_one_{table_one.shape[0]}.csv')
    return table_one

def generate_81k_dataset(admissions: pd.DataFrame, patients: pd.DataFrame, icustays: pd.DataFrame, chartevents: pd.DataFrame, items: pd.DataFrame):
    seed(2021)
    # Generate mid-sized input file
    table_one = join_table(admissions, patients, icustays)

    total_subjects = list(chartevents.SUBJECT_ID.unique())

    df_list = []

    split_one = int(np.round(len(total_subjects)*0.3,0))
    split_two = int(np.round(len(total_subjects)*0.9,0))

    for sub_index in range(0,split_one):
        search_id = total_subjects[sub_index]
        df_list.append(chartevents[chartevents.SUBJECT_ID == int(search_id)].head(1))
        
    for sub_index in range(split_one,split_two):
        search_id = total_subjects[sub_index]
        df_list.append(chartevents[chartevents.SUBJECT_ID == int(search_id)].head(2))
        
    for sub_index in range(split_two,len(total_subjects)):
        search_id = total_subjects[sub_index]
        df_list.append(chartevents[chartevents.SUBJECT_ID == int(search_id)].head(100))

    one_per_pat = pd.concat(df_list)
    one_per_pat.shape

    table_one = table_one.merge(one_per_pat[['SUBJECT_ID','ICUSTAY_ID','CHARTTIME','ITEMID','VALUE','VALUEUOM']],on=['SUBJECT_ID','ICUSTAY_ID'])

    new_admits = []
    new_dischs = []
    new_chart = []

    patient_stays = table_one[['SUBJECT_ID','ICUSTAY_ID','DOB','ADMITTIME']]
    patient_stays.drop_duplicates(inplace=True)

    for index, row in table_one.iterrows():
        admit_min = len(pd.date_range(start=pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime('2021-12-01 00:00:00',format='%Y-%m-%d %H:%M:%S'),freq='D'))
        stay_len = len(pd.date_range(start=pd.to_datetime(row['ADMITTIME'],format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime(row['DISCHTIME'],format='%Y-%m-%d %H:%M:%S'),freq='S'))

        num_days_admit = randrange(np.round(admit_min*0.25,0),np.round(admit_min*0.9,0)+5)
        num_days_disch = randrange(0,50)
        num_secs_chart = randrange(np.round(stay_len*0.01,0)+1,np.round(stay_len*0.99,0)+10)
        new_admit_date = pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S') + pd.DateOffset(days=num_days_admit)
        new_admits.append(new_admit_date)
        new_dischs.append(new_admit_date + pd.DateOffset(days=num_days_disch))
        new_chart.append(new_admit_date + pd.DateOffset(seconds=num_secs_chart))

    table_one['ADMITTIME'] = new_admits
    table_one['DISCHTIME'] = new_dischs
    table_one['CHARTTIME'] = new_chart

    table_one = table_one[(pd.to_datetime(table_one.ADMITTIME) < pd.to_datetime(table_one.CHARTTIME)) & (pd.to_datetime(table_one.DISCHTIME) > pd.to_datetime(table_one.CHARTTIME))]
    table_one = table_one.merge(items[['ITEMID','LABEL']],on=['ITEMID'])
    table_one.drop(['ICUSTAY_ID','ITEMID'],axis=1,inplace=True)
    age_calc = pd.Timestamp('2021-12-01 00:00:00')
    table_one['DOB'] = pd.to_datetime(table_one['DOB'], format='%Y-%m-%d %H:%M:%S')
    table_one['age'] = (age_calc - table_one['DOB']).astype('<m8[Y]')
    print(f"Mid sized input table saved, number of columns:  {table_one.shape[1]}, number of rows: {table_one.shape[0]}")
    #table_one.to_csv(f'data/05_model_input/table_one_imbalanced_{table_one.shape[0]}.csv')
    return table_one

def generate_217k_dataset(admissions: pd.DataFrame, patients: pd.DataFrame, icustays: pd.DataFrame, chartevents: pd.DataFrame, items: pd.DataFrame):
    seed(2021)
    # Generate large-sized input file
    table_one = join_table(admissions, patients, icustays)

    total_subjects = list(chartevents.SUBJECT_ID.unique())

    df_list = []

    split_one = int(np.round(len(total_subjects)*0.3,0))
    split_two = int(np.round(len(total_subjects)*0.7,0))

    for sub_index in range(0,split_one):
        search_id = total_subjects[sub_index]
        df_list.append(chartevents[chartevents.SUBJECT_ID == int(search_id)].head(1))
        
    for sub_index in range(split_one,split_two):
        search_id = total_subjects[sub_index]
        df_list.append(chartevents[chartevents.SUBJECT_ID == int(search_id)].head(2))
        
    for sub_index in range(split_two,len(total_subjects)):
        search_id = total_subjects[sub_index]
        df_list.append(chartevents[chartevents.SUBJECT_ID == int(search_id)].head(100))

    one_per_pat = pd.concat(df_list)
    one_per_pat.shape

    table_one = table_one.merge(one_per_pat[['SUBJECT_ID','ICUSTAY_ID','CHARTTIME','ITEMID','VALUE','VALUEUOM']],on=['SUBJECT_ID','ICUSTAY_ID'])

    new_admits = []
    new_dischs = []
    new_chart = []

    patient_stays = table_one[['SUBJECT_ID','ICUSTAY_ID','DOB','ADMITTIME']]
    patient_stays.drop_duplicates(inplace=True)

    for index, row in table_one.iterrows():
        admit_min = len(pd.date_range(start=pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime('2021-12-01 00:00:00',format='%Y-%m-%d %H:%M:%S'),freq='D'))
        stay_len = len(pd.date_range(start=pd.to_datetime(row['ADMITTIME'],format='%Y-%m-%d %H:%M:%S'),end=pd.to_datetime(row['DISCHTIME'],format='%Y-%m-%d %H:%M:%S'),freq='S'))

        num_days_admit = randrange(np.round(admit_min*0.25,0),np.round(admit_min*0.9,0)+5)
        num_days_disch = randrange(0,50)
        num_secs_chart = randrange(np.round(stay_len*0.01,0)+1,np.round(stay_len*0.99,0)+10)
        new_admit_date = pd.to_datetime(row['DOB'],format='%Y-%m-%d %H:%M:%S') + pd.DateOffset(days=num_days_admit)
        new_admits.append(new_admit_date)
        new_dischs.append(new_admit_date + pd.DateOffset(days=num_days_disch))
        new_chart.append(new_admit_date + pd.DateOffset(seconds=num_secs_chart))

    table_one['ADMITTIME'] = new_admits
    table_one['DISCHTIME'] = new_dischs
    table_one['CHARTTIME'] = new_chart

    table_one = table_one[(pd.to_datetime(table_one.ADMITTIME) < pd.to_datetime(table_one.CHARTTIME)) & (pd.to_datetime(table_one.DISCHTIME) > pd.to_datetime(table_one.CHARTTIME))]
    table_one = table_one.merge(items[['ITEMID','LABEL']],on=['ITEMID'])
    table_one.drop(['ICUSTAY_ID','ITEMID'],axis=1,inplace=True)
    age_calc = pd.Timestamp('2021-12-01 00:00:00')
    table_one['DOB'] = pd.to_datetime(table_one['DOB'], format='%Y-%m-%d %H:%M:%S')
    table_one['age'] = (age_calc - table_one['DOB']).astype('<m8[Y]')
    print(f"Large input table saved, number of columns:  {table_one.shape[1]}, number of rows: {table_one.shape[0]}")
    #table_one.to_csv(f'data/05_model_input/table_one_imbalanced_{table_one.shape[0]}.csv')
    return table_one








