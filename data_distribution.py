# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
orig_df = pd.read_csv('30_Training Dataset_V2/training_data.csv', header=0, keep_default_na=False, na_values=['None', '', '其他'])

# %%
dtype = np.float64
numeric_col_name_dict = {
    '土地面積': 'land_area',
    '移轉層次': 'floor',
    '總樓層數': 'total_floor',
    '屋齡': 'house_age', 
    '建物面積': 'total_building_area', 
    '車位面積': 'parking_area', 
    '車位個數': 'n_parking', 
    '橫坐標': 'x', 
    '縱坐標': 'y', 
    '主建物面積': 'main_building_area', 
    '陽台面積': 'balcony_area', 
    '附屬建物面積': 'aux_building_area', 
    '單價': 'unit_price'
}


# %%
numeric_df = orig_df[numeric_col_name_dict.keys()].astype(dtype)
eng_df = numeric_df.rename(columns=numeric_col_name_dict, inplace=False)


# %%
total_data_count = len(eng_df)
for col_name, col_vals in eng_df.items():

    min_val = col_vals.min()
    val_counts = col_vals.value_counts()
    min_val_infer_zero = min_val == val_counts.index[0]

    uniq_val_count = len(val_counts)
    col_val_count = total_data_count
    # if min_val_infer_zero:
    #     uniq_val_count -= 1
    #     col_val_count -= val_counts.values[0]

    print(col_name)
    print(f'min_val_infer_zero = {min_val_infer_zero}')
    print(f'uniq_val_count/col_val_count = {uniq_val_count}/{col_val_count}') 
    plt.figure(figsize=(10, 2))
    sns.histplot(data=eng_df, x=col_name, kde=True)
    plt.show()
    print('\n')


# %%
for col_name, col_vals in eng_df.items():
    col_value_counts = col_vals.value_counts()
    col_value_counts = col_value_counts.sort_index()
    plt.figure(figsize=(10, 2))
    plt.title(col_name, fontsize=20)
    plt.plot(col_value_counts.index, col_value_counts, '.-', color=(0, 0, 1, 0.5))
    plt.xscale('log')
    plt.show()
    print('\n')
