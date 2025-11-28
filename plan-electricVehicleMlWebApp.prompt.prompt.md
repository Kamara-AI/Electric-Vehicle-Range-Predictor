Plan: Electric Vehicle ML Web App

TL;DR — Analyze and clean `electric_vehicles_spec_2025.csv`, run EDA, build and compare three regression models for `range_km`, then create a Streamlit app to load the best model and serve predictions. This plan gives clear file targets, commands, and learning-focused checkpoints so you can implement and understand each step in VS Code.

Steps
1. Inspect dataset: run the local analyzer script against `electric_vehicles_spec_2025.csv` (create `analyze_dataset.py`) and review counts, dtypes, and missing-value report.
2. Clean data: implement `utils.py` with cleaning functions (parse numeric fields, normalize `segment`, handle missing `number_of_cells`, create flags).
3. EDA: create `notebooks/eda.ipynb` or `eda.py` to produce at least 6 plots (scatter, histograms, boxplots, heatmap, bar charts).
4. Model building: create `models.py` to train 3 regressors (XGBoost, RandomForest, ElasticNet), use 80/20 split + 5-fold CV, save models with `joblib` in `models/`.
5. Compare models: produce a line graph of metric vs model (RMSE/MAE/R2) in `notebooks/compare_models.ipynb`.
6. Streamlit app: create `app.py` that loads the best model, exposes UI controls (sliders, dropdowns, checkboxes), runs preprocessing via `utils.py`, shows prediction and simple charts.
7. Git + deploy: add `.gitignore`, commit, push to GitHub, and deploy to Streamlit Cloud using `requirements.txt`.

Further Considerations
1. Target type: Prefer regression predicting `range_km` (Option A). If you prefer buckets (short/medium/long), convert to classification (Option B). Which do you want?
2. Missing values: Option A / median imputation per segment; Option B / add binary missing indicators — recommend doing both.
3. Feature encoding: Use one-hot for small categorical features, target-encode `brand` only if performance improves.

Files to create next
- `analyze_dataset.py` — dataset diagnostics and exact numeric summaries
- `utils.py` — cleaning & preprocessing helpers
- `notebooks/eda.ipynb` or `eda.py` — EDA plots
- `models.py` — training and inference wrappers
- `models/` — folder for saved joblib models
- `app.py` — Streamlit app
- `requirements.txt` — pinned packages
- `README.md` — instructions and deploy notes

Quick commands (PowerShell)
```powershell
cd "c:\Users\Administrator\Whole project practice"
# Run dataset analyzer
python .\analyze_dataset.py
# Start Streamlit locally
streamlit run .\app.py
```

Minimal requirements.txt suggestion
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.3.2
xgboost==1.7.6
streamlit==1.25.0
matplotlib==3.8.1
seaborn==0.12.2
joblib==1.3.2
```

Git & deploy summary
- Initialize and push to GitHub:
```powershell
cd "c:\Users\Administrator\Whole project practice"
git init
git add .
git commit -m "Add EV specs dataset and analysis scaffolding"
# create repo on GitHub web UI, then:
# git remote add origin https://github.com/<your-username>/<repo>.git
git push -u origin main
```
- Deploy to Streamlit Cloud: push to GitHub, then in Streamlit Cloud create a new app and point it to `app.py` (ensure `requirements.txt` present).

Notes / Questions
- Do you want regression for `range_km` (recommended), or classification into range buckets?
- Do you prefer I generate all scaffold files now, or produce them step-by-step so you can follow along and learn?

Next actions I can take when you confirm:
- Generate `analyze_dataset.py` and run it locally instructions, plus create `utils.py` scaffold, or
- Generate the full set of scaffolding files (`analyze_dataset.py`, `utils.py`, `models.py`, `app.py`, `requirements.txt`, `README.md`) so you can open them in VS Code and iterate.
