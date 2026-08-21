import pandas as pd

from data_profile import create_profile


data = {
    "Product": ["Laptop", "Phone", "Tablet"],
    "Revenue": [50000, 30000, 20000],
    "Units": [10, 20, 15],
}

df = pd.DataFrame(data)

profile = create_profile(df)

print(profile)