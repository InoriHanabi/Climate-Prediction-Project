# Climate-Prediction-Project
## Setup

It is highly recommended to use a virtual environment.

**1. Setup:**
```bash
git clone https://github.com/InoriHanabi/Climate-Prediction-Project.git
cd your-repo-name
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

**2. Modules:**
Run the following to download necessary modules:
```bash
pip install -r requirements.txt
```

**3. Code Execution Process:**
Execute the scripts in the following order. All outputs will be saved to the plots/ directory and output_log.txt.

Step 1: Prepare Data for EDA
```bash
python prepare_eda_data.py
```

Use code with caution.

Step 2: Run Exploratory Data Analysis

```bash
jupyter notebook data_visualization.ipynb
```

Use code with caution.

Step 3: Run Predictive Modelling
```bash
python model_training.py
```
