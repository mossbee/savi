with open('code.txt', 'r') as file:
    code = file.read()



content = code.split('/it]')
print(len(content), content[-5:])

import pandas as pd

code_extracted = pd.read_csv('b6_code_extracted.csv')
task_ids = code_extracted['task_id'].tolist()

codes = []
for c in content[1:]:
    cc = c.split('\n')[:-1]
    codes.append('\n'.join(cc).strip())

print(len(codes), codes[-5:])
df = pd.DataFrame(codes, columns=['code'])
df.insert(0, 'task_id', task_ids)
df.to_json('code.json', orient='records')
