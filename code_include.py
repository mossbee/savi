import pandas as pd

df = pd.read_csv('b6_test_data.csv')

# create a new column 'code', and all values are None
df['code'] = None

yes_num = 0
no_num = 0

for index, row in df.iterrows():
    if "the output of" in row['question'] or "which behavior most" in row['question'] or "which option is the most likely" in row['question']:
        df.at[index, 'code'] = "Yes"
        yes_num += 1
    else:
        df.at[index, 'code'] = "No"
        no_num += 1

print(f"Yes: {yes_num}, No: {no_num}")

# save to a new csv file
df.to_csv('b6_test_data_modified.csv', index=False)

# generate two csv files, one with code = Yes, one with code = No
df_yes = df[df['code'] == "Yes"]
df_no = df[df['code'] == "No"]

df_yes.to_csv('b6_test_data_yes.csv', index=False)
df_no.to_csv('b6_test_data_no.csv', index=False)