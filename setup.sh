# 1. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip version
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
#pip install --upgrade pip

# 3. Install required libraries
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
#pip install -r requirements.txt