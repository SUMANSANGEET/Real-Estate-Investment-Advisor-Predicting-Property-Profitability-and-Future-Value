# Deploy the Real Estate Investment Advisor (Streamlit) — step by step

**Folder:** `app.py` · `train_models.py` · `housing_utils.py` · `requirements.txt` · `.streamlit/config.toml`
(+ `models/bundle.joblib` and `data/market_sample.csv`, created in step 2)

## 1. Set up (once)
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Put `india_housing_prices.csv` in this folder.

## 2. Train all models (creates the files the app needs)
```bash
python train_models.py --fast     # ~2 min smoke test (60k rows)
python train_models.py            # full run, ~5-10 min
```
Check that `models/bundle.joblib` is **under 100 MB** (GitHub limit): `ls -lh models`.

## 3. Run locally
```bash
streamlit run app.py              # http://localhost:8501
```
Move any sidebar slider: every chart re-scores live.

## 4. Push to GitHub
```bash
git init && git add . && git commit -m "Real Estate Investment Advisor"
git branch -M main
git remote add origin https://github.com/<you>/real-estate-advisor.git
git push -u origin main
```
Do **not** commit the raw CSV (large); commit `models/` and `data/market_sample.csv`.

## 5. Deploy on Streamlit Community Cloud (free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** → pick your repo, branch `main`, main file `app.py`.
3. **Advanced settings** → choose **Python 3.12** (matches the pinned wheels).
4. **Deploy**. First build takes a few minutes; you get a public `*.streamlit.app` URL.
5. Put that URL on your résumé, LinkedIn, GitHub README and portfolio site.

## 6. Troubleshooting
- *ModuleNotFoundError / pickle errors* → models were trained with different library versions; retrain with the pinned `requirements.txt`.
- *File too large* → rerun `train_models.py` with fewer trees (`n_estimators`) or use Git LFS.
- *App sleeps after inactivity* → open the URL once before an interview.

## 7. Optional: Docker
```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["sh","-c","streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
```

## 8. Interview demo script (2 minutes)
1. Change city → verdict and projection move live.  2. Drag the price slider → show the **negotiation insight**.
3. Open **Model Arena**: ROC overlay, 6 models + ensemble.  4. **ROI Simulator**: EMI, equity, net gain.
5. Close with the honesty note: targets are rule-based, not real-market forecasts.
