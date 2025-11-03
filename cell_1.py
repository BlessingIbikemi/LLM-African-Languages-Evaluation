# @title
import os
import pandas as pd
import numpy as np
import json
import time
import requests
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re

# ----------------------------
# Enhanced Configurations
# ----------------------------

LANG_CODES = ["yor", "ibo", "hau"]  # Sheet names in Excel
MODEL_NAMES = {
    "deepseek": "accounts/fireworks/models/deepseek-v3-0324",
    "qwen": "accounts/fireworks/models/qwen3-235b-a22b",
    "llama": "accounts/fireworks/models/llama-v3p1-405b-instruct"
}

LABEL_MAPPING = {0: "positive", 1: "neutral", 2: "negative"}
REVERSE_LABEL_MAPPING = {"positive": 0, "neutral": 1, "negative": 2}

NUM_EXAMPLES = 1000  # Using full dataset
EXCEL_FILE = "afrisenti_multilang_1000.xlsx"  # Your Excel file
OUTPUT_DIR = "enhanced_results"
SLEEP_BETWEEN_CALLS = 5

# Evaluation settings
EVALUATION_MODES = ["zero_shot", "few_shot"]
FEW_SHOT_EXAMPLES = 3
CONFIDENCE_LEVEL = 0.95

# ----------------------------
# API Configuration
# ----------------------------
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
API_KEY = "fw_3ZnfcXpqdWbJhREJS4CcgQLM"

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# ----------------------------
# Create output directory
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Enhanced Prompt Templates (Documented for Reproducibility)
# ----------------------------

PROMPT_TEMPLATES = {
    "zero_shot": {
        "template": """Classify the sentiment of this tweet as Positive, Negative, or Neutral.

Tweet: "{tweet}"

Sentiment:""",
        "description": "Zero-shot sentiment classification with clear instruction and format"
    },

    "few_shot": {
        "template": """Classify the sentiment of tweets as Positive, Negative, or Neutral.

Examples:
{examples}

Now classify this tweet:
Tweet: "{tweet}"

Sentiment:""",
        "description": "Few-shot learning with 3 examples per class from training data"
    }
}

# ----------------------------
# Data Loading and Preprocessing
# ----------------------------

def load_excel_data(file_path):
    """Load data from Excel sheets for each language"""
    data = {}
    for lang in LANG_CODES:
        try:
            df = pd.read_excel(file_path, sheet_name=lang)
            df = df.head(NUM_EXAMPLES)  # Use first 1000 records
            data[lang] = df
            print(f"✅ Loaded {len(df)} examples for {lang}")
        except Exception as e:
            print(f"❌ Error loading {lang} data: {e}")
    return data

def prepare_few_shot_examples(df, n_examples=1):
    """Prepare few-shot examples - 1 per class for brevity"""
    examples = []
    for label_idx, label_name in LABEL_MAPPING.items():
        class_data = df[df['label'] == label_idx]
        if len(class_data) > 0:
            sample = class_data.iloc[0]
            examples.append(f'Tweet: "{sample["tweet"]}"\nSentiment: {label_name.title()}')
    return "\n\n".join(examples)

# ----------------------------
# Enhanced API Calls with Error Handling
# ----------------------------

def call_fireworks_model(model_name, model_id, prompt, retries=3):
    """Enhanced API call with better error handling and logging"""
    for attempt in range(retries):
        try:
            payload = {
                "model": model_id,
                "max_tokens": 50,  # Reduced for faster responses
                "top_p": 0.9,
                "temperature": 0.1,  # Lower temperature for consistent classification
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(url=API_URL, headers=HEADERS, data=json.dumps(payload))

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                print(f"🔥 Rate limit for {model_name}. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"⚠️ Error {response.status_code}: {response.text[:200]}")

        except Exception as e:
            print(f"❌ Exception for {model_name}: {str(e)[:100]}")
            time.sleep(3)

    return None

# ----------------------------
# Enhanced Response Processing
# ----------------------------

def normalize_label(output):
    """Enhanced label normalization with better pattern matching"""
    if not output:
        return "unknown"

    output_lower = output.lower().strip()

    # Direct matches
    if output_lower in ["positive", "negative", "neutral"]:
        return output_lower

    # Pattern matching
    if re.search(r'\bpositive\b', output_lower):
        return "positive"
    elif re.search(r'\bnegative\b', output_lower):
        return "negative"
    elif re.search(r'\bneutral\b', output_lower):
        return "neutral"

    return "unknown"

def analyze_response_quality(raw_output, predicted_label):
    """Analyze response quality for error analysis"""
    issues = []

    if not raw_output:
        issues.append("no_response")
    elif len(raw_output) > 100:
        issues.append("verbose_response")
    elif predicted_label == "unknown":
        issues.append("unparseable")
    elif "english" in raw_output.lower() or "switch" in raw_output.lower():
        issues.append("language_switching")

    return issues

# ----------------------------
# Cultural and Linguistic Analysis
# ----------------------------

def detect_cultural_elements(tweet):
    """Detect cultural/linguistic elements in tweets"""
    cultural_indicators = {
        "proverbs": ["àgbàlagbà", "omenala", "karin magana"],  # Add more
        "cultural_refs": ["jollof", "fufu", "aso ebi", "owambe"],
        "code_switching": bool(re.search(r'[a-zA-Z]+.*[àáéèíìóòúù]', tweet)),
        "tonal_marks": bool(re.search(r'[àáéèíìóòúù]', tweet))
    }
    return cultural_indicators

# ----------------------------
# Statistical Analysis Functions
# ----------------------------

def calculate_confidence_interval(accuracy, n, confidence=0.95):
    """Calculate confidence interval for accuracy"""
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_error = z_score * np.sqrt((accuracy * (1 - accuracy)) / n)
    return accuracy - margin_error, accuracy + margin_error

def perform_statistical_tests(results_df):
    """Perform statistical significance tests between models"""
    models = results_df['model'].unique()
    languages = results_df['language'].unique()

    significance_results = {}

    for lang in languages:
        lang_data = results_df[results_df['language'] == lang]
        model_accuracies = {}

        for model in models:
            model_data = lang_data[lang_data['model'] == model]
            if len(model_data) > 0:
                model_accuracies[model] = model_data['is_correct'].values

        # Pairwise comparisons
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                if model1 in model_accuracies and model2 in model_accuracies:
                    statistic, p_value = stats.ttest_ind(
                        model_accuracies[model1],
                        model_accuracies[model2]
                    )
                    significance_results[f"{model1}_vs_{model2}_{lang}"] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }

    return significance_results

# ----------------------------
# Enhanced Error Analysis
# ----------------------------

def analyze_errors(results_df):
    """Comprehensive error analysis"""
    error_analysis = defaultdict(lambda: defaultdict(list))

    for _, row in results_df.iterrows():
        if not row['is_correct']:
            model = row['model']
            language = row['language']

            error_info = {
                "tweet": row['tweet'],
                "gold_label": row['gold_label'],
                "predicted_label": row['predicted_label'],
                "raw_output": row['raw_output'],
                "cultural_elements": detect_cultural_elements(row['tweet']),
                "response_issues": analyze_response_quality(row['raw_output'], row['predicted_label'])
            }

            error_analysis[model][language].append(error_info)

    return dict(error_analysis)

# ----------------------------
# Main Evaluation Function
# ----------------------------

def evaluate_model_comprehensive(model_name, model_id, lang, data_df, mode="zero_shot"):
    """Comprehensive model evaluation with enhanced metrics"""

    print(f"\n🔄 Evaluating {model_name} on {lang} ({mode} mode)")

    results = []
    few_shot_examples = prepare_few_shot_examples(data_df) if mode == "few_shot" else ""

    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"{model_name}-{lang}"):
        tweet = row['tweet']
        gold_label = LABEL_MAPPING[row['label']]

        # Create prompt based on mode
        if mode == "zero_shot":
            prompt = PROMPT_TEMPLATES["zero_shot"]["template"].format(tweet=tweet)
        else:
            prompt = PROMPT_TEMPLATES["few_shot"]["template"].format(
                examples=few_shot_examples,
                tweet=tweet
            )

        try:
            raw_output = call_fireworks_model(model_name, model_id, prompt)
            predicted_label = normalize_label(raw_output)
            is_correct = predicted_label == gold_label

            # Cultural analysis
            cultural_elements = detect_cultural_elements(tweet)
            response_issues = analyze_response_quality(raw_output, predicted_label)

            results.append({
                "model": model_name,
                "language": lang,
                "mode": mode,
                "tweet": tweet,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "cultural_elements": str(cultural_elements),
                "response_issues": str(response_issues),
                "prompt_template": mode
            })

            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"❌ Error processing tweet {idx}: {e}")
            continue

    return pd.DataFrame(results)

# ----------------------------
# Comprehensive Analysis and Reporting
# ----------------------------

def generate_comprehensive_report(all_results_df):
    """Generate comprehensive analysis report"""

    report = {
        "summary_stats": {},
        "statistical_tests": {},
        "error_analysis": {},
        "cultural_analysis": {},
        "per_class_performance": {}
    }

    # Summary statistics with confidence intervals
    for model in all_results_df['model'].unique():
        for lang in all_results_df['language'].unique():
            for mode in all_results_df['mode'].unique():
                subset = all_results_df[
                    (all_results_df['model'] == model) &
                    (all_results_df['language'] == lang) &
                    (all_results_df['mode'] == mode)
                ]

                if len(subset) > 0:
                    accuracy = subset['is_correct'].mean()
                    n = len(subset)
                    ci_lower, ci_upper = calculate_confidence_interval(accuracy, n)

                    # Per-class metrics
                    y_true = [REVERSE_LABEL_MAPPING[label] for label in subset['gold_label']]
                    y_pred = [REVERSE_LABEL_MAPPING.get(label, -1) for label in subset['predicted_label']]

                    valid_indices = [i for i, pred in enumerate(y_pred) if pred != -1]
                    if valid_indices:
                        y_true_valid = [y_true[i] for i in valid_indices]
                        y_pred_valid = [y_pred[i] for i in valid_indices]

                        precision, recall, f1, support = precision_recall_fscore_support(
                            y_true_valid, y_pred_valid, average=None, zero_division=0
                        )

                        macro_f1 = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)

                        key = f"{model}_{lang}_{mode}"
                        report["summary_stats"][key] = {
                            "accuracy": accuracy,
                            "confidence_interval": (ci_lower, ci_upper),
                            "macro_f1": macro_f1,
                            "per_class_precision": precision.tolist(),
                            "per_class_recall": recall.tolist(),
                            "per_class_f1": f1.tolist(),
                            "support": support.tolist(),
                            "n_samples": n
                        }

    # Statistical significance tests
    report["statistical_tests"] = perform_statistical_tests(all_results_df)

    # Error analysis
    report["error_analysis"] = analyze_errors(all_results_df)

    return report

# ----------------------------
# Visualization Functions
# ----------------------------

def create_visualizations(all_results_df, report):
    """Create comprehensive visualizations"""

    # Performance comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy by model and language
    pivot_accuracy = all_results_df.groupby(['model', 'language', 'mode'])['is_correct'].mean().reset_index()

    for i, mode in enumerate(["zero_shot", "few_shot"]):
        mode_data = pivot_accuracy[pivot_accuracy['mode'] == mode]
        pivot_plot = mode_data.pivot(index='language', columns='model', values='is_correct')

        sns.heatmap(pivot_plot, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   ax=axes[0, i], cbar_kws={'label': 'Accuracy'})
        axes[0, i].set_title(f'Accuracy Heatmap - {mode.replace("_", " ").title()}')

    # Confidence intervals plot
    models = all_results_df['model'].unique()
    languages = all_results_df['language'].unique()

    x_pos = np.arange(len(languages))
    width = 0.25

    for i, model in enumerate(models):
        accuracies = []
        ci_lowers = []
        ci_uppers = []

        for lang in languages:
            key = f"{model}_{lang}_zero_shot"
            if key in report["summary_stats"]:
                stats = report["summary_stats"][key]
                accuracies.append(stats["accuracy"])
                ci_lower, ci_upper = stats["confidence_interval"]
                ci_lowers.append(stats["accuracy"] - ci_lower)
                ci_uppers.append(ci_upper - stats["accuracy"])
            else:
                accuracies.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)

        axes[1, 0].bar(x_pos + i * width, accuracies, width,
                      yerr=[ci_lowers, ci_uppers], capsize=5, label=model)

    axes[1, 0].set_xlabel('Languages')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Model Performance with Confidence Intervals')
    axes[1, 0].set_xticks(x_pos + width)
    axes[1, 0].set_xticklabels(languages)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Error distribution
    error_counts = defaultdict(int)
    for _, row in all_results_df.iterrows():
        if not row['is_correct']:
            error_key = f"{row['gold_label']}→{row['predicted_label']}"
            error_counts[error_key] += 1

    if error_counts:
        error_types = list(error_counts.keys())
        error_values = list(error_counts.values())

        axes[1, 1].bar(range(len(error_types)), error_values)
        axes[1, 1].set_xlabel('Error Types (Gold→Predicted)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xticks(range(len(error_types)))
        axes[1, 1].set_xticklabels(error_types, rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

# ----------------------------
# Main Execution
# ----------------------------

def main():
    print("🚀 Starting Enhanced AfriSenti Evaluation...")

    # Load data
    data = load_excel_data(EXCEL_FILE)
    if not data:
        print("❌ No data loaded. Please check your Excel file.")
        return

    all_results = []

    # Run evaluations
    for model_name, model_id in MODEL_NAMES.items():
        for lang in LANG_CODES:
            if lang in data:
                for mode in EVALUATION_MODES:
                    results_df = evaluate_model_comprehensive(
                        model_name, model_id, lang, data[lang], mode
                    )
                    all_results.append(results_df)

    # Combine all results
    final_results_df = pd.concat(all_results, ignore_index=True)

    # Generate comprehensive report
    print("\n📊 Generating comprehensive analysis...")
    report = generate_comprehensive_report(final_results_df)

    # Save results
    final_results_df.to_csv(f"{OUTPUT_DIR}/detailed_results.csv", index=False)

    with open(f"{OUTPUT_DIR}/analysis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Create visualizations
    create_visualizations(final_results_df, report)

    # Print summary
    print(f"\n✅ Evaluation complete!")
    print(f"📁 Results saved in: {OUTPUT_DIR}/")
    print(f"📊 Processed {len(final_results_df)} total evaluations")

    # Print key findings
    print("\n🔍 Key Findings Summary:")
    for key, stats in report["summary_stats"].items():
        if "zero_shot" in key:
            print(f"{key}: Accuracy = {stats['accuracy']:.3f} ± {(stats['confidence_interval'][1] - stats['confidence_interval'][0])/2:.3f}")

if __name__ == "__main__":
    main()