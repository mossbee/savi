import pandas as pd
import ast

# Read CSV file
df = pd.read_csv("b6_train_data.csv")

# Prepare list for storing extracted data
extracted_data = []

for _, row in df.iterrows():
    task_id = row["task_id"]
    question = row["question"]
    answer = str(row["answer"]).strip()
    choices = row["choices"]
    
    # Extract valid answer character (e.g., 'C' from 'ANSWER: C')
    if "ANSWER:" in answer:
        answer = answer.split(":")[-1].strip()
    
    # Skip invalid answers (empty or numeric values)
    if not answer.isalpha() or len(answer) != 1:
        continue
    
    # Convert choices from string representation of list to actual list
    try:
        choices_list = ast.literal_eval(choices)
        if not isinstance(choices_list, list):
            continue
    except (SyntaxError, ValueError):
        continue
    
    # Ensure the answer index is within the valid range
    answer_index = ord(answer) - ord('A')
    if 0 <= answer_index < len(choices_list):
        extracted_data.append([task_id, question, str(choices_list[answer_index])])

# Save extracted data to a new CSV file
output_df = pd.DataFrame(extracted_data, columns=["task_id", "question", "correct_choice"])
output_df.to_csv("output.csv", index=False)
