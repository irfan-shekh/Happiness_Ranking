"""
Views for the World Happiness Ranking Prediction application.
Handles homepage rendering, model inference, and API endpoints.
"""

import os
import pickle
import numpy as np
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Paths to serialized ML artifacts
MODEL_PATH = os.path.join(settings.BASE_DIR, 'polls', 'HappinessModel.pickle')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'polls', 'HappinessScaler.pickle')

# Cached model & scaler in memory
_MODEL = None
_SCALER = None


def get_ml_artifacts():
    """Lazily load and cache the ML model and scaler from disk."""
    global _MODEL, _SCALER
    if _MODEL is None or _SCALER is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                "ML model or scaler not found. Please run 'python train_model.py' to generate them."
            )
        with open(SCALER_PATH, 'rb') as f:
            _SCALER = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            _MODEL = pickle.load(f)
    return _MODEL, _SCALER


def classify_happiness(rank):
    """Categorize rank into a descriptive tier and color theme."""
    if rank <= 30:
        return {
            'tier': 'Top Tier (Very High Happiness)',
            'badge': 'success',
            'summary': 'Flourishing society with high social support, robust economy, and strong life expectancy.',
            'percentile': f"Top {max(1, round((rank / 158.0) * 100))}% globally"
        }
    elif rank <= 75:
        return {
            'tier': 'Upper-Middle Tier (Good Happiness)',
            'badge': 'info',
            'summary': 'Above-average living standards with balanced economic and health metrics.',
            'percentile': f"Top {round((rank / 158.0) * 100)}% globally"
        }
    elif rank <= 120:
        return {
            'tier': 'Lower-Middle Tier (Moderate)',
            'badge': 'warning',
            'summary': 'Moderate happiness with opportunities for socio-economic and public health improvement.',
            'percentile': f"Ranked #{rank} out of 158 nations"
        }
    else:
        return {
            'tier': 'Developing Tier (Needs Improvement)',
            'badge': 'danger',
            'summary': 'Significantly impacted by lower income, social friction, or low life expectancy.',
            'percentile': f"Bottom {max(1, round(((159 - rank) / 158.0) * 100))}% globally"
        }


def extract_features(data):
    """Extract and validate the 8 numerical features from request data."""
    try:
        hap_score = float(data.get('hapScore', 0))
        standard_error = float(data.get('se', 0.04))
        economy = float(data.get('eco', 0))
        family = float(data.get('fam', 0))
        life_exp = float(data.get('life', 0))
        freedom = float(data.get('free', 0))
        generosity = float(data.get('Gen', 0))
        dystopia = float(data.get('dys', 0))

        feature_dict = {
            'Happiness Score': hap_score,
            'Standard Error': standard_error,
            'Economy (GDP per Capita)': economy,
            'Family': family,
            'Health (Life Expectancy)': life_exp,
            'Freedom': freedom,
            'Generosity': generosity,
            'Dystopia Residual': dystopia,
        }
        df = pd.DataFrame([feature_dict])
        return df, None
    except (ValueError, TypeError) as e:
        return None, f"Invalid numerical values in input: {str(e)}"


def predict_core(data):
    """Run model inference and return formatted prediction dictionary."""
    country_name = data.get('Country Name', '').strip()
    region = data.get('Region', '').strip()

    if not country_name:
        return {'error': 'Country name is required.'}

    df, err = extract_features(data)
    if err:
        return {'error': err}

    try:
        model, scaler = get_ml_artifacts()
        scaled_input = scaler.transform(df)
        raw_pred = model.predict(scaled_input)[0]
        # Clamp rank between 1 and 158
        predicted_rank = int(np.clip(round(float(raw_pred)), 1, 158))
        tier_info = classify_happiness(predicted_rank)

        return {
            'success': True,
            'country': country_name,
            'region': region,
            'result': predicted_rank,
            'tier': tier_info['tier'],
            'badge': tier_info['badge'],
            'summary': tier_info['summary'],
            'percentile': tier_info['percentile'],
            'score': float(df['Happiness Score'].iloc[0]),
            'gdp': float(df['Economy (GDP per Capita)'].iloc[0]),
            'life': float(df['Health (Life Expectancy)'].iloc[0]),
            'freedom': float(df['Freedom'].iloc[0]),
        }
    except Exception as e:
        return {'error': f"Prediction failed: {str(e)}"}


def first(request):
    """Homepage view: renders the main UI and handles standard form submissions."""
    result_data = None

    if request.method == 'POST' and request.POST.get('pred_button'):
        is_ajax = (
            request.headers.get('x-requested-with') == 'XMLHttpRequest' or
            request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' or
            request.POST.get('ajax') == '1'
        )
        res = predict_core(request.POST)

        if is_ajax:
            return JsonResponse(res)

        result_data = res

    return render(request, 'index.html', {'result_data': result_data})


@csrf_exempt
def predict_ajax(request):
    """Dedicated API endpoint for JSON and AJAX predictions."""
    if request.method == 'POST':
        res = predict_core(request.POST)
        status_code = 200 if res.get('success') else 400
        return JsonResponse(res, status=status_code)
    return JsonResponse({'error': 'POST method required.'}, status=405)
