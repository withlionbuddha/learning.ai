import os
import re

def get_saved_home():
    # JUPYTER_CONFIG_DIR 환경변수 확인
    jupyter_config_dir = os.getenv('JUPYTER_CONFIG_DIR')

    # 환경변수가 설정되어 있지 않다면 예외 처리
    if jupyter_config_dir is None:
        raise EnvironmentError("JUPYTER_CONFIG_DIR 환경변수가 설정되지 않았습니다.")
    
    print(f"jupyter_config_dir = {jupyter_config_dir} ")
    # 설정 파일 경로 정의
    config_file = os.path.join(jupyter_config_dir, 'jupyter_notebook_config.py')

    # 설정 파일이 있는지 확인
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Jupyter 설정 파일을 찾을 수 없습니다: {config_file}")

    # 설정 파일에서 notebook_dir 값을 찾기 위한 정규식
    notebook_dir_pattern = re.compile(r"c\.NotebookApp\.notebook_dir\s*=\s*[\"'](.*?)[\"']")

    # 설정 파일을 읽고 notebook_dir 값을 찾음
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = notebook_dir_pattern.search(line)
            if match:
                notebook_dir = match.group(1)
                return notebook_dir
    
    # 설정 파일에서 notebook_dir을 찾지 못했을 때
    raise ValueError("'NotebookApp.notebook_dir' 설정을 찾을 수 없습니다.")

# 환경변수에 등록된 Jupyter Notebook의 Home
try:
    jupyter_home_dir = get_saved_home()
    print(f"Jupyter Notebook의 Saved Home: {jupyter_home_dir}")
except Exception as e:
    print(e)


def get_jupyter_current_home():

    #Jupyter Notebook에서 현재 작업 디렉토리를 가져오는 함수.
    #이는 Jupyter Notebook이 실행된 홈 디렉토리를 반환합니다.
    return os.getcwd()


# Jupyter Notebook이 실행된 홈 디렉토리
try:
    current_directory = get_jupyter_current_home()
    print(f"Jupyter Notebook의 Current Home : {current_directory}")
except Exception as e:
    print(e)
