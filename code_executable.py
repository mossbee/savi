import pandas as pd
import re

def extract_code(question):
    patterns = [
        r"^Question: What will be output of the following c code\?",
        r"^Question: What is the output of this program\?",
        r"^Question: What will be the output of the following C# code\?",
        r"^Question: What will be the output of the following C\+\+ code\?",
        r"^Question: What will be the output of the following Java code\?",
        r"^Question: What is the output of the following code\?",
        r"^Question: What will be the output of the following Java code snippet\?",
        r"^Question: What would be the output of the following code \(in editor window\)\?",
        r"^Question: What will be the output of the following Python expression\?",
    ]
    
    for pattern in patterns:
        match = re.match(pattern, question)
        if match:
            return question[match.end():].strip()
    return None

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    
    df['code'] = df['question'].apply(extract_code)
    
    filtered_df = df.dropna(subset=['code'])[['task_id', 'code']]
    
    filtered_df.to_csv(output_file, index=False)

# Usage
process_csv('b6_test_data.csv', 'b6_test_code.csv')