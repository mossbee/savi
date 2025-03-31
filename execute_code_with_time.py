import json
import multiprocessing

def execute(python_code, return_dict):
    """Executes Python code in an isolated process."""
    try:
        local_vars = {}
        exec(python_code, {}, local_vars)
        return_dict["output"] = local_vars.get("a", "Executed successfully")
    except SystemExit as e:
        return_dict["output"] = f"SystemExit: {e}"
    except Exception as e:
        return_dict["output"] = f"Error: {e}"

if __name__ == "__main__":
    with open("code_filtered.json", "r") as file:
        data = json.load(file)
        print(">>>", len(data), ">>>", data[0])

    for i, d in enumerate(data):
        print("task_id:", d["task_id"])
        python_code = d["code"]

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        process = multiprocessing.Process(target=execute, args=(python_code, return_dict))
        process.start()

        process.join(timeout=3)  # Timeout of 3 seconds

        if process.is_alive():
            print(f"Timeout: Task {d['task_id']} took too long!")
            process.terminate()
            process.join()
            d["output"] = "Timeout: Execution exceeded 3 seconds"
        else:
            d["output"] = return_dict.get("output", "Unknown result")

    with open("code_filtered_output.json", "w") as file:
        json.dump(data, file, indent=4)

    print("Execution complete!")
