import pandas as pd

# read b6_test_data.csv
df = pd.read_csv('b6_test_data.csv')

def format_choices_with_letters(choices_str):
    choices = eval(choices_str)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    return "\n".join([f"{letters[i]}. {choice}" for i, choice in enumerate(choices)])

for _, row in tqdm(questions_df.iterrows()):
    task_id = row['task_id']
    question = row['question']
    choices_str = row['choices']
    
    formatted_choices = format_choices_with_letters(choices_str)
    
    # Format the prompt for this specific question
    current_prompt = prompt.format(
        question=question,
        formatted_choices=formatted_choices
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers multiple-choice programming questions. For each question, respond with only the letter corresponding to the correct option: A, B, C, or D. Do not include explanations, code, or restate the question. Do not add quotation marks or punctuation—just the letter."},
        {"role": "user", "content": current_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Store the result
    results.append({"task_id": task_id, "answer": extract_first_uppercase_with_dot(response)})