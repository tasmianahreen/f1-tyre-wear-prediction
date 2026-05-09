# 🏎️ Formula 1 Tyre Wear Prediction

Machine learning + data visualization project for predicting Formula 1 tyre degradation using FastF1 telemetry and race data.

## Features

- Collect F1 race and stint data using FastF1
- Train a tyre degradation prediction model
- Interactive Streamlit dashboard
- Visualize tyre degradation curves
- Compare tyre compounds and driver performance

## Setup

```bash
pip install -r requirements.txt
```

## Run Data Collection

```bash
python src/collect_data.py
```

## Train Model

```bash
python src/train_model.py
```

## Launch Dashboard

```bash
streamlit run src/app.py
```

## Future Improvements

- Add multiple race weekends
- Use XGBoost models
- Add weather features
- Predict optimal pit stop windows
- Build live race simulations
