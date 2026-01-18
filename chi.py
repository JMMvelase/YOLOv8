
import pandas as pd

df = pd.DataFrame({
    'age': [2, 4, 6, 8],
    'height': [85, 100, 115, 125]
})

correlation = df.corr()  # Pearson by default
print(correlation)
