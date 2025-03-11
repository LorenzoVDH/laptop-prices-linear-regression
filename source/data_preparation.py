from dataclasses import dataclass, field 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd

@dataclass
class DataPreparation:
    csv_file: str  # Path to the CSV file
    df: pd.DataFrame = field(init=False)  # DataFrame to store the CSV data
    processor_encoder: LabelEncoder = field(init=False)
    gpu_encoder: LabelEncoder = field(init=False)
    processor_scaler: MinMaxScaler = field(init=False)
    gpu_scaler: MinMaxScaler = field(init=False)
    ram_scaler: MinMaxScaler = field(init=False)
    price_scaler: MinMaxScaler = field(init=False)

    def __post_init__(self) -> None:
        ''' Load the CSV file into a DataFrame and initialize encoders/scalers '''
        csv = pd.read_csv(self.csv_file)
        self.df = csv[['Processor', 'GPU', 'RAM (GB)', 'Price ($)']]
        self.processor_encoder = LabelEncoder()
        self.gpu_encoder = LabelEncoder()
        self.processor_scaler = MinMaxScaler()
        self.gpu_scaler = MinMaxScaler()
        self.ram_scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()

    def prepare_data(self) -> None:
        ''' Prepare the data by encoding categories and normalizing numerical values '''
        #encode the values into numerical values 
        self.df['Processor'] = self.processor_encoder.fit_transform(self.df['Processor'])
        self.df['GPU'] = self.gpu_encoder.fit_transform(self.df['GPU'])
        #normalize the numerical values between 0 and 1 
        self.df['Processor'] = self.processor_scaler.fit_transform(self.df[['Processor']])
        self.df['GPU'] = self.gpu_scaler.fit_transform(self.df[['GPU']])
        self.df['RAM (GB)'] = self.ram_scaler.fit_transform(self.df[['RAM (GB)']])
        self.df['Price ($)'] = self.price_scaler.fit_transform(self.df[['Price ($)']])

    def get_data(self) -> pd.DataFrame:
        ''' Return the prepared DataFrame '''
        return self.df

    def inverse_transform_price(self, normalized_price: float) -> float:
        '''
            This function takes a normalized price and returns the actual price. 
            It's the inverse of the price normalization function and uses the inverse_transform method of the price_scaler.
        '''
        return self.price_scaler.inverse_transform([[normalized_price]])[0][0]
