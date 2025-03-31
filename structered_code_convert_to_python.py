from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('b6_test_data.csv')

def format_choices_with_letters(choices_str):
    choices = eval(choices_str)
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    return "\n".join([f"{letters[i]}. {choice}" for i, choice in enumerate(choices)])


# Set your vLLM endpoint as the OpenAI-compatible API
os.environ["OPENAI_API_BASE"] = "https://glad-bass-wrongly.ngrok-free.app/v1"
os.environ["OPENAI_API_KEY"] = "sk-fake-key"  # Not used but required by LangChain

# Define structured output using Pydantic
class ExtractedInfo(BaseModel):
    answer: str = Field(description="The output of the given code as one of A, B, C, D, or E")
    # reason: str = Field(description="A short reason explaining why the output is correct")

# Initialize the vLLM-backed ChatOpenAI model
llm = ChatOpenAI(model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
                    temperature=0,
                    openai_api_base=os.environ["OPENAI_API_BASE"],
                    
                )

llm = llm.with_structured_output(ExtractedInfo, method="json_mode")

# Function to analyze code and return structured output
def analyze_code(question, formated_choices):
    prompt = [ ("system", "You are a helpful assistant that excellent in answering multiple-choice programming questions. Let's explain step by step."), ("human",
        "Analyze the following question and choices."
        "Choose the correct answer from the following choices and return JSON with 'answer'.\n\n"
        f"Question:\n{question}\n\n"
        "Which choice is correct?\n\n"
        f"Choices:\n{formated_choices}\n\n"
    )]
    return llm.invoke(prompt)

# Example usage
if __name__ == "__main__":

    for _, row in tqdm(df.iterrows()):
        results = []
        task_id = row['task_id']
        question = row['question']
        choices_str = row['choices']
        
        formatted_choices = format_choices_with_letters(choices_str)

        analyze_code(question, formatted_choices)
        result = analyze_code(question, formatted_choices)

        # save task_id and answer to results to save to submission.csv file
        results.append({
            'task_id': task_id,
            'answer': result.answer,
        })
        # Save the results to a CSV file
        output_df = pd.DataFrame(results)
        output_df.to_csv('submission.csv', index=False, mode='a', header=not os.path.exists('submission.csv'))
