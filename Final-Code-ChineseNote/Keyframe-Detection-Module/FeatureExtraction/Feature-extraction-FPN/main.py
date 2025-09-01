import os

def run_python(script):
    print(f"\n>>> Running: {script}")
    code = os.system(f"python {script}")
    if code != 0:
        raise RuntimeError(f"Script {script} exited with code {code}")

def main():
    
    run_python('process.py')  

    
  
    run_python('extract_features.py')
    
    run_python('analyze_features.py')
    run_python('gradcam_visualize.py')  

if __name__ == '__main__':
    main()
