import json
import pandas as pd


def get_value_from_log(path, key):
    with open(path, 'rb') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df[key][len(df[key]) - 1]
