
# Physical Activity Monitoring Project

## Python Environment Setup (Python 3.13)

Follow these steps to create and activate a virtual environment named `AI_ML_env` using Python 3.13:

### 1. Create the Virtual Environment

Open PowerShell and run:

```powershell
python3.13 -m venv AI_ML_env
python -m venv AI_ML_env
```

### 2. Activate the Environment

for windows
```powershell
.\AI_ML_env\Scripts\Activate    
cd AI_ML/Physical_activity_monitoring
pip install -r requirements_win.txt
```

for mac
```consol
source AI_ML_env/bin/activate
cd AI_ML/Physical_activity_monitoring
pip install -r requirements_mac.txt
```


### 3. Upgrade pip and Install Required Packages

```powershell
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install -U wandb
```

### 4. Save Installed Packages

```powershell
pip freeze > requirements.txt
```

### 5. Deactivate the Environment (when done)

```powershell
deactivate
```

---
Document the dataset source and usage in this file. Data source for the project is located in outside the vertion tracked AI_ML folder.  
