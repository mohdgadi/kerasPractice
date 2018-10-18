import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("./dataset.csv")
new_dict = dict()

iteration = 1
for i,x in enumerate(data["Balance"]):
    if x not in new_dict:
        new_dict[x] = iteration
        data["Balance"][i] = iteration
        iteration = iteration+1
    else:
         data["Balance"][i] = new_dict.get(x)

train, test = train_test_split(data, test_size=0.2)

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_training = scaler.fit_transform(train)
scaled_testing = scaler.transform(test)

scaled_training_df = pd.DataFrame(scaled_training, columns=train.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("sales_data_testing_scaled.csv", index=False)
