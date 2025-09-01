import os

def run_python(script):
    print(f"\n>>> Running: {script}")
    code = os.system(f"python {script}")
    if code != 0:
        raise RuntimeError(f"Script {script} exited with code {code}")

def main():
    # 1. 数据处理 数据合适的话要注释掉避免重复运行浪费时间，太慢了
    run_python('process.py')  
    run_python('extract_features.py')
    run_python('analyze_features.py')
    run_python('gradcam_visualize.py')
    # 3. 训练模型
    # run_python('train.py')
    # # 4. 评估
    # run_python('evaluate.py')

if __name__ == '__main__':
    main()
