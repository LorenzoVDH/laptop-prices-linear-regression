from dataclasses import dataclass, field 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

@dataclass
class DataPreparation:
    csv_file: str  # Path to the CSV file
    df: pd.DataFrame = field(init=False)  # DataFrame to store the CSV data
    label_encoder: LabelEncoder = field(init=False, default_factory=LabelEncoder)
    minmax_scaler: MinMaxScaler = field(init=False, default_factory=MinMaxScaler)

    def __post_init__(self):
        """ Load the CSV file into a DataFrame and initialize encoders/scalers """
        try:
            csv = pd.read_csv(self.csv_file)
            self.df = csv[['Processor', 'GPU', 'RAM (GB)', 'Price ($)']]
        except FileNotFoundError:
            print(f"ERROR: File {self.csv_file} not found.")
            raise
        except Exception as e:
            print("An error occurred while reading the CSV file:", e)
            raise

    def prepare_data(self):
        """ Prepare the data by encoding categories and normalizing numerical values """
        try:
            self.df['Processor'] = self.label_encoder.fit_transform(self.df['Processor'])
            self.df['GPU'] = self.label_encoder.fit_transform(self.df['GPU'])
            self.df['RAM (GB)'] = self.minmax_scaler.fit_transform(self.df[['RAM (GB)']])
            self.df['Price ($)'] = self.minmax_scaler.fit_transform(self.df[['Price ($)']])
        except Exception as e:
            print("An error occurred during data preparation:", e)
            raise

    def get_data(self):
        """ Return the prepared DataFrame """
        return self.df
