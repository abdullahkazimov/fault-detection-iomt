import numpy as np
import pandas as pd

dataset_path = "raw_data.csv"
df = pd.read_csv(dataset_path, encoding="utf-8")
df['datetime'] = pd.to_datetime(df['DATE'] + " " + df['Time'], format="%d/%m/%Y %H:%M", dayfirst=True)
df = df.drop(['DATE', 'Time'], axis=1)
df['cooling pump'] = df['cooling pump'].fillna(0)
counts = df['labeling'].value_counts()
to_remove = counts[counts < 10].index
df = df[~df['labeling'].isin(to_remove)]
df["heating season"] = (
    (~df["Heat Exchanger"].isna()) |
    (~df["Heating supply temperature"].isna()) |
    (df["cooling pump"] == 0)
).astype(int)
def to_float_safe(col):
    return (col.astype(str)
            .str.replace(',', '.')                   # Handle Korean/Excel commas
            .str.replace(r'[^\d.-]', '', regex=True) # Remove non-numeric except .-
            .replace(['', '.', '..'], np.nan)
            .astype(float))
df['Return temperature'] = to_float_safe(df['Return temperature'])
df['Supply air temperature'] = to_float_safe(df['Supply air temperature'])
df = df.rename(columns={'AHU name': 'ahu_id', 'labeling': 'label'})
df = pd.get_dummies(df, columns=['ahu_id'], prefix='ahu', dtype=int)
df['label'] = df['label'].str.strip().replace({
    'Normal condition': 'Normal',
    'Return air temperature fault': 'RATSF',
    'Supply fan fault': 'SFF',
    'Valve position fault': 'VPF'
})
df['label'] = df['label'].map({'Normal': 0, 'RATSF': 1, 'SFF': 2, 'VPF': 3})
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["day_of_month"] = df["datetime"].dt.day
ahu_mapping = {
    "ahu_AHU-N01": "ahu_1",
    "ahu_AHU-N02": "ahu_2",
    "ahu_AHU-N03": "ahu_3",
    "ahu_AHU-N04": "ahu_4",
    "ahu_AHU-N05": "ahu_5",
    "ahu_AHU-N06": "ahu_6",
    "ahu_AHU-N07": "ahu_7",
    "ahu_AHU-N08": "ahu_8"
}
df = df.rename(columns=ahu_mapping)
df = df.rename(columns={
    "Set point temperature": "set_point_temperature",
    "Return temperature": "return_temperature",
    "Supply air temperature": "supply_air_temperature",
    "Supply fan": "supply_fan",
    "Valve position": "valve_position",
    "Heat Exchanger": "heat_exchanger",
    "Heating supply temperature": "heating_supply_temperature",
    "Cooling supply temperature": "cooling_supply_temperature",
    "cooling pump": "cooling_pump",
    "label": "label",
    "heating season": "heating_season",
    "ahu_1": "ahu_1",
    "ahu_2": "ahu_2",
    "ahu_3": "ahu_3",
    "ahu_4": "ahu_4",
    "ahu_5": "ahu_5",
    "ahu_6": "ahu_6",
    "ahu_7": "ahu_7",
    "ahu_8": "ahu_8",
    "hour": "hour",
    "dayofweek": "day_of_week",
    "month": "month",
    "is_weekend": "is_weekend",
    "day_of_month": "day_of_month"
})
df['cooling_pump'] = df['cooling_pump'].astype(int)
df['supply_fan'] = df['supply_fan'].astype(int)
datetime_cols = ['datetime', 'hour', 'day_of_week', 'month', 'is_weekend', 'day_of_month']
ahu_cols = [col for col in df.columns if col.startswith('ahu_')]
other_cols = [col for col in df.columns if col not in datetime_cols + ahu_cols]
new_order = datetime_cols + other_cols + ahu_cols
df = df[new_order]
df = df.sort_values(by='datetime')
df = df.drop(['datetime'], axis=1)
ahu_cols = [f'ahu_{i}' for i in range(1, 9)]
ahu_label_counts = {}
for ahu in ahu_cols:
    counts = df[df[ahu] == 1]['label'].value_counts().sort_index()
    ahu_label_counts[ahu] = counts
ahu_label_counts_df = pd.DataFrame(ahu_label_counts).fillna(0).astype(int)
ahu_label_counts_df.to_csv("label_ahu_distribution.csv")
# full descriptive mapping
full_label_mapping = {
    0: 'Normal condition',
    1: 'Return air temperature fault',
    2: 'Supply fan fault',
    3: 'Valve position fault'
}
# map numeric labels to full names
df['label_full'] = df['label'].map(full_label_mapping)
# create a separate DataFrame with counts
label_counts_df = df['label_full'].value_counts().rename_axis('label').reset_index(name='count')
label_counts_df.to_csv("label_distribution.csv")
df.to_csv("preprocessed_data.csv")
