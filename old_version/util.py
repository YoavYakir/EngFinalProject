import time

def calculate_statistics(usage_history):
     
    if not usage_history:  # Check if the list is empty
        return 0, 0, 0
    
    peak = max(usage_history)
    average = sum(usage_history) / len(usage_history)
    minimum = min(usage_history)
    
    return peak, average, minimum

def open_main_process_file(started, pid = -1): # writes either "True" or "False" to the file, indicating whether the main process was started
    with open('main_process.txt', 'w') as f:
        f.write(str(started))
        if pid != -1 and started:
            f.write("\n" + str(pid))

def check_main_process_activiation():
    try:
        with open('main_process.txt', 'r') as f:
            started = f.readline().strip() == "True"
            if started:
                pid = f.readline().strip()
                return pid
            return -1
    
    except FileNotFoundError:
        print("main process file does not exist.")
        return -1
    

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.6f} seconds to run.")
        return result # the result of the function being measures is returned

    return wrapper

        