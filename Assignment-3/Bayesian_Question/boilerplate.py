#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model
import time
######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    trainData = pd.read_csv("train_data.csv")
    validationData = pd.read_csv("validation_data.csv")
    # print(len(trainData.columns))
    return trainData , validationData
    

def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    # startTime = time.time()    
    storedValues = [(df.columns[i] , df.columns[j]) for i in range(len(df.columns)) for j in range(i + 1 , len(df.columns))]
    dagGraph = bn.make_DAG(storedValues)
    # cpds = bn.print_CPD(dagGraph)
    bn.plot(dagGraph)    
    dagmodel = bn.parameter_learning.fit(dagGraph , df)
    # cpds = bn.print_CPD(dagGraph)
    # print(cpds)
    # endTime = time.time()
    # print("Time taken to train the first model: " , endTime - startTime , "seconds")
    return dagmodel
    
    
def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    # startTime = time.time()
    prunemodel = bn.structure_learning.fit(df , methodtype='cs')
    g = bn.plot(prunemodel)
    prunemodel = bn.independence_test(prunemodel , df , alpha=0.05 , prune=True)
    bn.plot(prunemodel , g['pos']) 
    prunemodel = bn.parameter_learning.fit(prunemodel , df)
    # endTime = time.time()
    # print("Time taken to train the pruned model: " , endTime - startTime , "seconds")
    return prunemodel    
    


def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    # startTime = time.time()
    optimizedmodel = bn.structure_learning.fit(df)
    bn.plot(optimizedmodel)
    optimizedmodel = bn.parameter_learning.fit(optimizedmodel , df)
    # endTime = time.time()   
    # print("Time taken to train the optimized model: " , endTime - startTime , "seconds")    
    return optimizedmodel       
    
    
    
    

def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    # base_model = make_network(train_df)
    # save_model("base_model.pkl", base_model)

    # Create and save pruned model
    # pruned_network = make_pruned_network(train_df)
    # save_model("pruned_model.pkl", pruned_network)

    # # Create and save optimized model
    # optimized_network = make_optimized_network(train_df)
    # save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    # evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    # evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

