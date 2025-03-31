import json

with open('code_filtered.json', 'r') as file:
    data = json.load(file)
    print(">>>", len(data), ">>>", data[0])
    for i, d in enumerate(data):
        print("task_id:", d['task_id'])
        python_code = d['code']
        # execute the python code and get the output
        try:
            # ret = exec(python_code, {}, {})
            def execute(python_code):
                local_vars = {}
                exec(python_code, {}, local_vars)
                return local_vars.get("a")  # Returns 42 without affecting outer scope

            print(execute(python_code))  # Output: 42
        except SystemExit as e:
            print(f"SystemExit: {e}")
            d['output'] = str(e)        
        except Exception as e:
            print(f"Error executing code: {e}")
            d['output'] = str(e)
