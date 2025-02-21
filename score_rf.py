import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization

def preprocess_data(file_path):

    with open(file_path, 'r') as f:
        data = json.load(f)
    
    records = []
    instance_metadata = {} 
    for region in data.values():
        for instance in region:
            record = {
                'cpu': float(instance['cpu']),
                'memory': float(instance['memory']),
                'spot_price': float(instance['spot_price']),
                'Price_per_CPU': float(instance['Price_per_CPU']),
                'Price_per_memory': float(instance['Price_per_memory']),
            }
            records.append(record)

            instance_metadata[(instance['typeName'], instance['region'])] = instance  # Store instance metadata
    
    df = pd.DataFrame(records)
    return df, instance_metadata

def prepare_data(AWS_data_file, Azure_data_file=None):

    aws_data, aws_metadata = preprocess_data(AWS_data_file)
    
    if Azure_data_file:
        azure_data, azure_metadata = preprocess_data(Azure_data_file)
        combined_data = pd.concat([aws_data, azure_data], ignore_index=True)
        combined_metadata = {**aws_metadata, **azure_metadata}
    else:
        combined_data = aws_data
        combined_metadata = aws_metadata
    
    X = combined_data.drop(columns=['spot_price']) 
    y = combined_data['spot_price']  
    
    return X, y, combined_metadata

def optimize_hyperparameters(X_train, y_train):

    def rf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        params = {
            'n_estimators': int(n_estimators),
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf),
            'max_features': max_features,
            'random_state': 42
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        return -rmse  

    pbounds = {
        'max_depth': (3, 10),
        'min_samples_split': (5, 20),
        'min_samples_leaf': (1, 20),
        'max_features': (0.3, 0.9),
        'n_estimators': (50, 500),
    }
    
    optimizer = BayesianOptimization(
        f=rf_cv,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=10)
    
    return optimizer.max['params']

def train_random_forest_model(X_train, y_train, best_params):

    model = RandomForestRegressor(
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        max_features=best_params['max_features'],
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_predictions(predictions, instance_metadata, file_path):
    result = []
    
    for (instance_type, region), prediction in zip(instance_metadata.keys(), predictions):
        instance_info = instance_metadata[(instance_type, region)]
        
        components = instance_info.get('components', [])
        
        instance_data = {
            "price": prediction, 
            "EC2 Type": "onDemand", 
            "region": instance_info['region'],
            "instances": [
                {
                    "onDemandPrice": instance_info['onDemandPrice'], 
                    "region": instance_info['region'],
                    "cpu": str(instance_info['cpu']), 
                    "ebsOnly": True if instance_info['storage'] == 'EBS only' else False,
                    "family": instance_info['family'],
                    "memory": str(instance_info['memory']),  
                    "network": instance_info.get('network', 'Moderate'), 
                    "os": instance_info['os'],
                    "typeMajor": instance_info['typeMajor'],
                    "typeMinor": instance_info['typeMinor'],
                    "storage": instance_info['storage'],
                    "typeName": instance_info['typeName'],
                    "physicalProcessor": instance_info.get('physicalProcessor', 'Intel Xeon Family'),
                    "processorArchitecture": "64-bit",  
                    "Architecture": "x86_64",  
                    "discount": instance_info.get('discount', 0),  
                    "interruption_frequency": instance_info.get('interruption_frequency', "5%-10%"),
                    "interruption_frequency_filter": 1.0,  
                    "spot_price": instance_info['spot_price'], 
                    "Price_per_CPU": instance_info['Price_per_CPU'],  
                    "Price_per_memory": instance_info['Price_per_memory'],  
                    "components": components
                }
            ]
        }
        result.append(instance_data)

    # result.sort(key=lambda x: x['price'])

    with open(file_path, 'w') as f:
        json.dump(result, f, indent=4)


def main():
    AWS_data_file = 'AWSData/ec2_data_Linux.json'
    X, y, instance_metadata = prepare_data(AWS_data_file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = optimize_hyperparameters(X_train, y_train)

    model = train_random_forest_model(X_train, y_train, best_params)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")

    if predictions is not None:
        predictions = predictions.astype(float)
        save_predictions(predictions, instance_metadata, "rf_predictions.json")
        print("Predictions have been saved to rf_predictions.json")
    else:
        print("Predictions are None. Please check your model and data.")

if __name__ == "__main__":
    main()
