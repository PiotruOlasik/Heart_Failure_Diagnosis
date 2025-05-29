import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Klasa do analizy danych. Wejściowy dataset musi być w postaci liczbowej. Można wywołać summarize albo pojedyncze metody
class Data_exploration():
    def __init__(self,dataset):    #input=pandas.dataFrame
        self.data=dataset

    def info(self):
        print("Information about database:\n")
        print(self.data.info())

    def mising_values(self):
        print("Missing data:\n", self.data.isna().sum())

    def head(self):
        print("First 5 rows:\n", self.data.head())

    def describtion(self):
        print("Database description:\n", self.data.describe())

    def unique_values(self):
        for col in self.data.columns:
            print(f'Unique values in "{col}" column:\n', self.data[col].unique())

    def num_of_samples_per_class(self):
        print("Number of samples in each class:\n", self.data['HeartDisease'].value_counts())

    def histograms(self):
        self.data.hist(figsize=(14, 8), bins=15)
        plt.suptitle("Histograms for all features", fontsize=16)
        plt.tight_layout()
        plt.show()

    def boxplots(self):
        for col in self.data.columns:
            if col != 'HeartDisease':
                plt.figure(figsize=(4, 2))
                sns.boxplot(data=self.data, x='HeartDisease', y=col)
                plt.title(f"{col} vs HeartDisease")
                plt.show()

    def distributions(self):
        for col in self.data.columns:
            if col != 'HeartDisease':
                plt.figure(figsize=(8, 6))
                sns.histplot(data=self.data, x=col, hue='HeartDisease', multiple='stack', kde=True)
                plt.title(f'Distribution of HeartDisease by {col}')
                plt.xlabel(f'{col}')
                plt.ylabel('Count')
                plt.show()

    def pair_plots(self):
        sns.pairplot(self.data, hue="HeartDisease", corner=True)
        plt.suptitle("Scatterplots between features by HeartDisease")
        plt.show()

    def correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    def mean_values_per_class(self):
        means = self.data.groupby("HeartDisease").mean()
        means.T.plot(kind='bar', figsize=(14, 6))
        plt.title("Mean values of features depending on the HeartDisease")
        plt.ylabel("Mean")
        plt.xlabel("Features")
        plt.xticks(rotation=45)
        plt.legend(title="HeartDisease")
        plt.tight_layout()
        plt.show()


    def summarize(self):
        self.info()
        self.mising_values()
        self.head()
        self.describtion()
        self.unique_values()
        self.num_of_samples_per_class()
        self.histograms()
        self.distributions()
        self.pair_plots()
        self.correlation_matrix()
        self.mean_values_per_class()
