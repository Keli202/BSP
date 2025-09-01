import os

def run_python(script):
    print(f"\n>>> Running: {script}")
    code = os.system(f"python {script}")
    if code != 0:
        raise RuntimeError(f"Script {script} exited with code {code}")

def main():

  
    run_python('train.py')
    run_python('evaluate.py')
    run_python('postprocess_diverse.py')

if __name__ == '__main__':
    main()
