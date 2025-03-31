import csv
import re

def extract_questions(input_csv, output_csv):
    patterns = [
        r"^Question: What will be output\??",
        r"^Question: What will be the output\??",
        r"^Question: What is the output\??",
        r"^Question: What would be the output\??"
    ]
    
    with open(input_csv, newline='', encoding='utf-8') as infile, \
         open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['task_id', 'extracted_text']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            question = row['question'].strip()
            task_id = row['task_id']
            
            for pattern in patterns:
                match = re.match(pattern, question)
                if match:
                    extracted_text = question[question.find('?')+1:].strip()
                    writer.writerow({'task_id': task_id, 'extracted_text': extracted_text})
                    break

extract_questions('b6_test_data.csv', 'b6_code_extracted.csv')
