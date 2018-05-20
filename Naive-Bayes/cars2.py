import pandas as pd
import numpy as np
import nltk

nltk.NaiveBayesClassifier



def load_dataset(trainFile, testFile, trainingSet=[], y=[], testSet=[]):
    # train.txt
    with open(trainFile, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        trainDataset = list(lines)
        for x in range(len(trainDataset)):
            trainingSet.append(trainDataset[x][:-1])
            y.append(trainDataset[x][-1])

    # test.txt
    with open(testFile, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        testDataset = list(lines)
        for x in range(len(testDataset)):
            # for y in range(6):
            #     testDataset[x][y] = float(testDataset[x][y])
            testSet.append(testDataset[x])



def main():
    # given data
    train = pd.read_csv('car_bayes/training')
    test = pd.read_csv('car_bayes/test')
    # print(test)

    # sample
    car = pd.DataFrame()
    car['buying_price'] = ['vhigh']
    car['price_maintenance'] = ['vhigh']
    car['number_doors'] = ['4']
    car['capacity_people'] = ['4']
    car['luggage_boot'] = ['big']
    car['safety'] = ['high']

    # Number of acc
    n_acc = train['acceptability'][train['acceptability'] == 'acc'].count()
    # Number of unacc
    n_unacc = train['acceptability'][train['acceptability'] == 'unacc'].count()
    # Number of vgood
    n_vgood = train['acceptability'][train['acceptability'] == 'vgood'].count()
    # Number of vgood
    n_vgood = train['acceptability'][train['acceptability'] == 'good'].count()
    # Total rows
    total_cars = train['acceptability'].count()

    # Number of acc divided by the total rows
    P_acc = n_acc / total_cars
    # Number of unacc divided by the total rows
    P_unacc = n_unacc / total_cars
    # Number of vgood divided by the total rows
    P_vgood = n_vgood / total_cars
    # Number of acc divided by the total rows
    P_good = n_vgood / total_cars

    #Likelihood
    # Group the data by gender and calculate the means of each feature
    # train['acceptability'].astype(str).astype(int)
    data = pd.DataFrame()
    data['Gender'] = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female']
    # data_means = train['acceptability'].data.groupby('acceptability').mean()
    # data_means = data.groupby('Gender').mean()
    print(train.dtypes)



main()