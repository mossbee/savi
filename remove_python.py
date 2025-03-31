import pandas as pd
import json

df = pd.read_csv('b6_test_data.csv')

# return the list of task_id that has question that has "following Python code" in it
task_id_list = df[df['question'].str.contains("following Python code")]['task_id'].tolist()

# iterate through the task 