import pandas as pd
import numpy as np

dates = pd.date_range('2023-01-01', periods=100)
hours = [f'{h}时' for h in range(1, 25)]

price_data = np.random.uniform(200, 800, size=(100, 24))
wind_data = np.random.uniform(0, 100, size=(100, 24))
pv_data = np.random.uniform(0, 100, size=(100, 24))
load_data = np.random.uniform(1000, 5000, size=(100, 24))
thermal_data = np.random.uniform(500, 4000, size=(100, 24))

with pd.ExcelWriter('汇总.xlsx') as writer:
    pd.DataFrame(price_data, index=dates, columns=hours).to_excel(writer, sheet_name='日前电价')
    pd.DataFrame(wind_data, index=dates, columns=hours).to_excel(writer, sheet_name='风电24')
    pd.DataFrame(pv_data, index=dates, columns=hours).to_excel(writer, sheet_name='光伏24')
    pd.DataFrame(load_data, index=dates, columns=hours).to_excel(writer, sheet_name='负荷24')
    pd.DataFrame(thermal_data, index=dates, columns=hours).to_excel(writer, sheet_name='负载24')
print("Mock data generated: 汇总.xlsx")
