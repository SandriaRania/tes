import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#Baca Dataset
base_path = os.path.dirname(__file__)  
file_name = "Filedata Data Jumlah Penduduk Provinsi DKI Jakarta Berdasarkan Agama.csv"
file_path = os.path.join(base_path, file_name)
df = pd.read_csv(file_path)


#Kolom dan Baris Dataset
print(df.shape,"\t")


#Jumlah Class Tiap Fitur
for i in df.columns:
    print(i, "dengan jumlah unik", df[i].nunique())


#Missing Values
col_na = df.isnull().sum().sort_values(ascending=True)
percent = col_na / len(df)
missing_data = pd.concat([col_na, percent], axis=1, keys=['Total', 'Percent'])

if (missing_data[missing_data['Total'] > 0].shape[0] == 0):
    print("Tidak ditemukan missing value pada dataset")
else:
    print(missing_data[missing_data['Total'] > 0])


#Outlier
Q1 = df["jumlah"].quantile(0.25)
Q3 = df["jumlah"].quantile(0.75)

IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR

outliers = (df["jumlah"] < lower_limit) | (df["jumlah"] > upper_limit)
print ("Outlier pada atribut:")
print(outliers.sum())


#Barchart Agama
agama_counts = df["agama"].value_counts()
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=agama_counts.index, y=agama_counts.values, palette="viridis")
plt.xlabel("Agama")
plt.ylabel("Total")
plt.title("Distribusi Agama")
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=10, color="black", weight="bold")
plt.show()