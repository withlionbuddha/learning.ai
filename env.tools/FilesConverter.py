import os
import json
import nbformat
from nbconvert import PythonExporter
from JupyterNotebook import get_jupyter_current_home

# 홈 디렉토리와 저장 디렉토리 지정
home_dir = get_jupyter_current_home()

# Python Exporter 생성
exporter = PythonExporter()

def get_directories(path):
    return {d: os.path.join(path, d)
            for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
                    and not d.startswith('.')
                    and not d.startswith('env')}

def get_ipynb_files(path):
    return [f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
                    and f.endswith('.ipynb')]

def convert(home_dir):
    for dir in get_directories(home_dir):
        for file_name in get_ipynb_files(dir):
            convert_file(dir, file_name)

def convert_file(target_dir, file_name):
    try:
        # .ipynb 파일 경로 생성 및 체크
        notebook_path = os.path.join(target_dir, file_name).replace('\\', '/')

        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"No such file: {notebook_path}")

        # Jupyter 노트북을 읽어서 .py 파일로 변환
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)  # 파일 객체를 넘겨줘야 함

        print(nbformat.validator.validate(notebook_content))  # 노트북 유효성 검사

        # 변환
        body, _ = exporter.from_notebook_node(notebook_content)

        # 파일 저장 경로
        py_filename = file_name.replace(".ipynb", ".py")
        py_file_path = os.path.join(target_dir, py_filename).replace('\\', '/')
        
        # 변환된 .py 파일 저장
        with open(py_file_path, 'w', encoding='utf-8') as py_file:
            py_file.write(body)

        print(f"Converted: {file_name} to {py_filename}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
    except KeyError as e:
        print(f"Key Error: {e}")
    except Exception as e:
        print(f"Exception: {e}")

def main():
    DEBUG = False
    if DEBUG:
        try:
            convert_file('F:\project\workspace', 'aaa.ipynb')  
        except Exception as e:
            print(e)
    else:
        convert(home_dir)

if __name__ == "__main__":
    main()
