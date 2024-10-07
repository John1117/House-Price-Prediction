# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
def twd97_to_wgs84(city_E_N_df):
    dtype = np.float64

    # some preliminary values
    a = 6378.137 # 地球半徑
    f = 1 / 298.257222101 # 地球扁率的倒數

    n = f / (2-f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)

    b1 = n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96
    b2 = n ** 2 / 48 + n ** 3 / 15
    b3 = 17 * n ** 3 / 480

    d1 = 2 * n - 2 * n ** 2 / 3 - 2 * n ** 3
    d2 = 7 * n ** 2 / 3 - 8 * n ** 3 / 5
    d3 = 56 * n ** 3 / 15

    # TWD97 params
    TW_lng0 = np.radians(121) # 臺灣中央經線(弧度)
    PKM_lng0 = np.radians(119) # 澎湖、金門、馬祖中央經線(弧度)
    k0 = 0.9999 # 中央經線尺度
    N0 = 0 # 縱座標橫移量(公里)
    TW_E0 = 250.000 # 臺灣橫坐標橫移量(公里)
    PKM_E0 = 0.0 # 澎金馬橫坐標橫移量(公里)

    # m to Km
    is_PKM_series = city_E_N_df['縣市'].apply(lambda x: x in ['澎湖縣', '金門縣', '連江縣'])
    lng0 = is_PKM_series.apply(lambda is_PKM: PKM_lng0 if is_PKM else TW_lng0).to_numpy(dtype)
    E0 = is_PKM_series.apply(lambda is_PKM: PKM_E0 if is_PKM else TW_E0).to_numpy(dtype)

    E = city_E_N_df['橫坐標'].to_numpy(dtype) / 1000.0
    N = city_E_N_df['縱坐標'].to_numpy(dtype) / 1000.0

    xi = (N - N0) / (k0 * A)
    eta = (E - E0) / (k0 * A)

    xip = xi \
        - b1 * np.sin(2 * 1 * xi) * np.cosh(2 * 1 * eta) \
        - b2 * np.sin(2 * 2 * xi) * np.cosh(2 * 2 * eta) \
        - b3 * np.sin(2 * 3 * xi) * np.cosh(2 * 3 * eta)
        
    etap = eta \
        - b1 * np.cos(2 * 1 * xi) * np.sinh(2 * 1 * eta) \
        - b2 * np.cos(2 * 2 * xi) * np.sinh(2 * 2 * eta) \
        - b3 * np.cos(2 * 3 * xi) * np.sinh(2 * 3 * eta)

    chi = np.arcsin(np.sin(xip) / np.cosh(etap))

    lat = chi \
        + d1 * np.sin(2 * 1 * chi) \
        + d2 * np.sin(2 * 2 * chi) \
        + d3 * np.sin(2 * 3 * chi)

    lng = lng0 + np.arctan(np.sinh(etap) / np.cos(xip))

    return pd.DataFrame({'lat': np.degrees(lat), 'lng': np.degrees(lng)})


# orig_df = pd.read_csv('30_Private Dataset _Private and Publict Submission Template_v2/private_dataset.csv', header=0, keep_default_na=False, na_values=['None', '', '其他'])
# city_E_N_df = orig_df[['縣市', '橫坐標', '縱坐標']]

# choose_PKM = False
# if choose_PKM:
#     city_E_N_df = city_E_N_df[city_E_N_df['縣市'] == '金門縣']
# else:
#     city_E_N_df = city_E_N_df[city_E_N_df['縣市'] != '金門縣']

# E_N_df = city_E_N_df[['橫坐標', '縱坐標']]

# lat_lng_df = twd97_to_wgs84(city_E_N_df)

# fig, ax1 = plt.subplots(figsize=(7, 14))

# ax1.plot(E_N_df['橫坐標'], E_N_df['縱坐標'], 'c.', alpha=0.5)
# ax1.set_xlim(E_N_df['橫坐標'].min(), E_N_df['橫坐標'].max())
# ax1.set_ylim(E_N_df['縱坐標'].min(), E_N_df['縱坐標'].max())
# ax1.set_xlabel('TWD97 X', color='c')
# ax1.set_ylabel('TWD97 Y', color='c')
# ax1.tick_params(axis='x', colors='c')
# ax1.tick_params(axis='y', colors='c')

# ax2 = fig.add_subplot(111, frame_on=False)
# ax2.xaxis.tick_top()
# ax2.yaxis.tick_right()

# ax2.plot(lat_lng_df['lng'], lat_lng_df['lat'], 'b.', alpha=0.5)
# ax2.set_xlim(lat_lng_df['lng'].min(), lat_lng_df['lng'].max())
# ax2.set_ylim(lat_lng_df['lat'].min(), lat_lng_df['lat'].max())
# ax2.xaxis.set_label_position('top') 
# ax2.yaxis.set_label_position('right')
# ax2.set_xlabel('WGS84 Lng', color='b')
# ax2.set_ylabel('WGS84 Lat', color='b')
# ax2.tick_params(axis='x', colors='b')
# ax2.tick_params(axis='y', colors='b')
# plt.savefig('tw.png', transparent=True)

# plt.show()

