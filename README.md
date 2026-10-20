# 🌍 World Happiness Ranking Predictor

An AI-powered Django machine learning web application that predicts a country's global **Happiness Ranking** (1 to 158) based on multi-dimensional socio-economic indicators from the **World Happiness Report**.

---

## 🚀 Key Highlights & Enhancements

- **Modern Glassmorphism UI**: High-end dark theme designed with curated color palettes, Google Fonts (*Plus Jakarta Sans* & *Space Grotesk*), and Bootstrap 5.
- **One-Click Country Presets**: Quick-load authentic data for **Switzerland (#1)**, **United States (#15)**, **Brazil (#16)**, **Japan (#46)**, **India (#117)**, and **Burundi (#157)**.
- **Dual Synchronized Controls**: Real-time range sliders paired with high-precision numeric inputs for seamless experimentation.
- **Live Predictive Intelligence**: Instant AJAX rank inference with global percentile indicators, happiness tier categorization, and driver weight breakdown bars.
- **High-Accuracy ML Model**: Powered by a **Gradient Boosting Regressor** (`R² ~ 0.9995`, `MAE ~ 1.00`) and standard feature scaling.
- **Clean & Modern Codebase**: Upgraded dependencies supporting **Python 3.9 through Python 3.13+** and **Django 4.2 / 5.x / 6.x**.

---

## 📁 Project Directory Structure

```text
Happiness Ranking/
│
├── Happiness.csv              # Authentic World Happiness Report dataset (158 countries)
├── Happiness.ipynb            # Jupyter notebook for exploratory data analysis & research
├── train_model.py             # Reproducible ML training script (fits & exports model + scaler)
├── requirements.txt           # Up-to-date Python package requirements
├── manage.py                  # Django CLI management script
├── db.sqlite3                 # Local SQLite database
├── README.md                  # Comprehensive setup and user guide
│
├── mysite/                    # Django project configuration
│   ├── __init__.py
│   ├── settings.py            # App settings, static paths, and allowed hosts
│   ├── urls.py                # Root URL routing configuration
│   ├── wsgi.py                # WSGI entry point for web servers
│   └── asgi.py                # ASGI entry point for async servers
│
└── polls/                     # Core web application & ML inference module
    ├── __init__.py
    ├── admin.py               # Django administration hooks
    ├── apps.py                # App configuration
    ├── models.py              # Data models
    ├── urls.py                # App route mapping ('/' and '/predict/')
    ├── views.py               # Request handlers, cached ML inference & tier logic
    ├── HappinessModel.pickle  # Serialized Gradient Boosting Regressor model
    ├── HappinessScaler.pickle # Serialized StandardScaler preprocessor
    └── templates/
        └── index.html         # Modern glassmorphism UI template with AJAX engine
```

---

## 🛠️ Step-by-Step Execution Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** installed:
```bash
python --version
```

---

### 2. (Recommended) Create & Activate a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### On Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies
Install all required libraries (Django, scikit-learn, pandas, numpy):
```bash
pip install -r requirements.txt
```

---

### 4. (Optional) Train or Retrain the ML Model
The repository includes pre-trained model artifacts. You can retrain or inspect the model anytime with:
```bash
python train_model.py
```
*Output will display train/test R² scores, MAE, and sanity check predictions before saving `HappinessModel.pickle` and `HappinessScaler.pickle` into `polls/`.*

---

### 5. Run Database Migrations
Initialize the SQLite database schema:
```bash
python manage.py migrate
```

---

### 6. Start the Development Server
Launch the local Django server:
```bash
python manage.py runserver
```

---

### 7. Access the Application
Open your web browser and navigate to:
```text
http://127.0.0.1:8000/
```

---

## 💡 How to Use the Web Application

1. **Quick-Start with Presets**:
   Click any of the country chips at the top right (e.g., **🇨🇭 Switzerland**, **🇺🇸 USA**, or **🇮🇳 India**) to populate the form instantly with authentic values.
2. **Interactive Sliders**:
   Adjust values using either the range sliders or typing exact numerical numbers in the input fields.
3. **Compute Prediction**:
   Click **"🎯 Compute Predicted Rank"** to run instant inference.
4. **Read Real-Time Insights**:
   - **Predicted Rank**: Exact rank position (from #1 to #158).
   - **Status Tier**: Categorized as *Top Tier*, *Upper-Middle*, *Lower-Middle*, or *Developing Tier*.
   - **Global Gauge**: Visual position along the 158-nation spectrum.
   - **Factor Breakdown**: Relative impact of GDP, Health Life Expectancy, and Freedom.
   - **Copy Result**: Click **"Copy Result"** to copy a formatted summary to your clipboard.

---

## 📊 Model & Dataset Details

- **Dataset**: World Happiness Report (158 nations)
- **Target Variable**: `Happiness Rank` (1 = Happiest, 158 = Least Happy)
- **Features Used (8 Indicators)**:
  1. `Happiness Score` (Cantril Ladder evaluation)
  2. `Standard Error` (Statistical uncertainty)
  3. `Economy (GDP per Capita)`
  4. `Family` (Social support network)
  5. `Health (Life Expectancy)`
  6. `Freedom` (Autonomy to make life choices)
  7. `Generosity` (Charitable behavior and giving)
  8. `Dystopia Residual` (Benchmark baseline comparison)
- **Algorithm**: `GradientBoostingRegressor` (120 estimators, learning rate = 0.08, max depth = 4)
- **Performance**:
  - **Train R²**: `1.0000`
  - **Test R²**: `0.9995`
  - **Test MAE**: `1.00 rank position`

---

## 🔌 API Endpoints

### `POST /predict/`
Asynchronous inference endpoint accepting `multipart/form-data` or `application/x-www-form-urlencoded`.

**Payload Parameters**:
- `Country Name`: String (e.g., "Switzerland")
- `Region`: String (e.g., "Western Europe")
- `hapScore`: Float
- `se`: Float
- `eco`: Float
- `fam`: Float
- `life`: Float
- `free`: Float
- `gov`: Float
- `Gen`: Float
- `dys`: Float
- `ajax`: `"1"`

**Sample JSON Response**:
```json
{
  "success": true,
  "country": "Switzerland",
  "region": "Western Europe",
  "result": 1,
  "tier": "Top Tier (Very High Happiness)",
  "badge": "success",
  "summary": "Flourishing society with high social support, robust economy, and strong life expectancy.",
  "percentile": "Top 1% globally",
  "score": 7.587,
  "gdp": 1.39651,
  "life": 0.94143,
  "freedom": 0.66557
}
```

---

## 🛡️ License & Credits
- Created as an open-source machine learning showcase.
- Dataset courtesy of the **United Nations Sustainable Development Solutions Network (World Happiness Report)**.
