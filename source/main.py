import os 
from data_preparation import DataPreparation 
from sklearn.model_selection import train_test_split
import pandas as pd  # for type hints
from laptop_price_predictor import LaptopPricePredictor  # new import

if __name__ == '__main__':
    # append the directory of the current file to the csv file 
    csv_path: str = os.path.join(os.path.dirname(__file__), 'data/laptop_prices.csv') 

    # Use the DataPreparation class to prepare the data     
    dp = DataPreparation(csv_file=csv_path)
    try:
        dp.prepare_data()
    except Exception as e:
        print("Error preparing data:", e)
        exit(1)
    priceDataFull = dp.get_data() 

    # Split the inputs and output data into two arrays  
    X: pd.DataFrame = priceDataFull[['Processor', 'GPU', 'RAM (GB)']] # inputs (matrix)
    y: pd.Series = priceDataFull['Price ($)']                         # output (vector)

    # Use sklearn train_test_split to split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Create an instance of our LaptopPricePredictor class 
    predictor = LaptopPricePredictor()
    predictor.train(X_train, y_train)
    
    # Define legal choices for test inputs
    valid_processors = {1: "AMD Ryzen 9", 2: "AMD Ryzen 7", 3: "Intel i9", 4: "Intel i7", 5: "Intel i5"}
    valid_gpus = {1: "AMD Radeon RX 6600", 2: "Nvidia RTX 3060", 3: "AMD Radeon RX 6600"}
    valid_ram = [8, 16, 32]

    # Unified prompt function for numeric selection.
    # 'options' can be a dict (for processor and GPU) or a list (for RAM).
    def prompt_choice(prompt_text: str, options):
        while True:
            print(f"{prompt_text} (Options: {options})")
            choice = input().strip()
            try:
                choice_as_int = int(choice)
            except ValueError:
                print("Please enter a valid numeric value.")
                continue
            if isinstance(options, dict) and choice_as_int in options:
                return options[choice_as_int]
            elif isinstance(options, list) and choice_as_int in options:
                return choice_as_int
            print("Invalid option. Try again.")

    user_processor = prompt_choice("Enter processor", valid_processors)
    user_gpu = prompt_choice("Enter GPU", valid_gpus)
    user_ram = prompt_choice("Enter RAM (GB)", valid_ram)
    
    # Encode categorical inputs using the fitted encoders,
    # then normalize using the fitted scalers.
    proc_encoded = dp.processor_encoder.transform([user_processor])
    proc_normalized = dp.processor_scaler.transform(pd.DataFrame({'Processor': proc_encoded}))[0][0]
    
    gpu_encoded = dp.gpu_encoder.transform([user_gpu])
    gpu_normalized = dp.gpu_scaler.transform(pd.DataFrame({'GPU': gpu_encoded}))[0][0]
    
    ram_normalized = dp.ram_scaler.transform(pd.DataFrame({'RAM (GB)': [user_ram]}))[0][0]
    
    test_prediction = predictor.predict(proc_normalized, gpu_normalized, ram_normalized)
    print(f"Test Prediction (Normalized Price): {test_prediction}")
    
    actual_price = dp.inverse_transform_price(test_prediction)
    print(f"Test Prediction (Actual Price): ${actual_price}")
