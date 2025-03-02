import os 
from data_preparation import DataPreparation 
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    #append the directory of the current file to the csv file 
    csv_path = os.path.join(os.path.dirname(__file__), 'data/laptop_prices.csv') 

    #Use the DataPreparation class to prepare the data     
    dp = DataPreparation(csv_file = csv_path)
    dp.prepare_data() 
    priceDataFull = dp.get_data() 

    #Split the inputs and output data into two arrays  
    X = priceDataFull[['Processor', 'GPU', 'RAM (GB)']] #Capitalized because it's a matrix 
    y = priceDataFull['Price ($)']                      #Not capitalized because it's a vector

    #Use sklearn train_test_split to split the data into training and testing sets
    #test_size is the proportion of the dataset to include in the test split (train_size is inferred)
    #random_state is the seed used by the random number generator
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    print("\n\nTraining Input Features Head:\n\n",X_train.head()) 
    print("\n\nTraining Output Head:\n\n",y_train.head()) 
