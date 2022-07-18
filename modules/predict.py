import os
import json
import dill
from datetime import datetime
import pandas as pd
from pandas import DataFrame, read_csv, isna

path = os.path.expanduser('~/airflow_hw')


def predict():
    folder = f'{path}/data/models/'
    last_pick = [f'{folder}{el}' for el in list(sorted(os.listdir(folder), reverse=True))][0]
    with open(last_pick, 'rb') as file:
        model = dill.load(file)

    upload_list = {}
    folder1 = f'{path}/data/test/'
    n = [f'{folder1}{el}' for el in list(sorted(os.listdir(folder1)))]

    for files in n:
        with open(files, 'rb') as data:
            data1 = json.load(data)
            df = pd.DataFrame([data1])
            y = model.predict(df)
            upload_list[df.iloc[0]['id']] = y[0]
    result = pd.DataFrame(list(upload_list.items()), columns=['car_id', 'pred'])
    result = result.set_index('car_id')
    model_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    result.to_csv(model_filename)


if __name__ == '__main__':
    predict()
