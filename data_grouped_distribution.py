# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


# %%
orig_df = pd.read_csv('30_Training Dataset_V2/training_data.csv', header=0, keep_default_na=False)


df = orig_df.copy()



# %%
catorgory_name = '縣市'
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index
plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')
plt.show()



# %%
catorgory_name = '鄉鎮市區'
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index
plt.figure(figsize=(10, 50))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')
plt.show()



# %%
catorgory_name = '使用分區'
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index
plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')
plt.show()


# %%
catorgory_name = '移轉層次'
df[catorgory_name] = df[catorgory_name].astype(str)
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index

plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')

plt.show()


# %%
catorgory_name = '移轉層次'
df[catorgory_name] = df[catorgory_name].astype(str)
order = np.arange(orig_df[catorgory_name].min(), orig_df[catorgory_name].max()+1).astype(str)

plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}_ordered.png')

plt.show()



# %%
catorgory_name = '總樓層數'
df[catorgory_name] = df[catorgory_name].astype(str)
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index

plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')

plt.show()


# %%
catorgory_name = '總樓層數'
df[catorgory_name] = df[catorgory_name].astype(str)
order = np.arange(orig_df[catorgory_name].min(), orig_df[catorgory_name].max()+1).astype(str)

plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}_ordered.png')

plt.show()



# %%
catorgory_name = '主要用途'
df[catorgory_name] = df[catorgory_name].astype(str)
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index

plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')

plt.show()


# %%
catorgory_name = '主要建材'
df[catorgory_name] = df[catorgory_name].astype(str)
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index

plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')

plt.show()



# %%
catorgory_name = '建物型態'
df[catorgory_name] = df[catorgory_name].astype(str)
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index

plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')

plt.show()



# %%
catorgory_name = '屋齡'
df[catorgory_name] = df[catorgory_name].astype(str)
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index
plt.figure(figsize=(10, 100))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')

plt.show()


# %%
catorgory_name = '屋齡'
df[catorgory_name] = df[catorgory_name].astype(str)
order = np.arange(orig_df[catorgory_name].min(), orig_df[catorgory_name].max()+1, 1/12).astype(str)
plt.figure(figsize=(10, 100))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}_ordered.png')

plt.show()


# %%
catorgory_name = '車位個數'
df[catorgory_name] = df[catorgory_name].astype(str)
order = df.groupby(by=catorgory_name)['單價'].median().sort_values().iloc[::-1].index
plt.figure(figsize=(10, 10))
sns.violinplot(data=df, x='單價', y=catorgory_name, split=False, palette='Spectral', inner='quart', order=order)
plt.grid(axis='x')
plt.savefig(f'grouped_data_distribution_plot/{catorgory_name}.png')
plt.show()

# %%
