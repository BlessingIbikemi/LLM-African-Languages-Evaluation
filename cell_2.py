import os
import pandas as pd
import numpy as np
import json
import time
import requests
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# ================================================
# ENHANCED CONFIGURATION FOR AFRIMGSM
# ================================================

LANG_CODES = ["yor", "ibo", "hau"]  # Excel sheet names
lang_map = {
    "yor": "yoruba",
    "ibo": "igbo",
    "hau": "hausa"
}
MODEL_NAMES = {
    "deepseek": "accounts/fireworks/models/deepseek-v3-0324",
    "qwen": "accounts/fireworks/models/qwen3-235b-a22b",
    "llama": "accounts/fireworks/models/llama-v3p1-405b-instruct"
}

NUM_EXAMPLES = 250  # Using your Excel data
EXCEL_FILE = "masakhane_afrimgsm_250.xlsx"  # Your Excel file
OUTPUT_DIR = "afrimgsm_enhanced_results"
SLEEP_BETWEEN_CALLS = 5
CONFIDENCE_LEVEL = 0.95

# API Configuration
API_KEY = "fw_3ZnfcXpqdWbJhREJS4CcgQLM"
API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================
# MATHEMATICAL REASONING PROMPT TEMPLATES
# ================================================

MATH_PROMPT_TEMPLATES = {
    "zero_shot": {
        "template": """Solve the following math word problem step by step and provide only the final numerical answer.

Problem: {question}

Final Answer:""",
        "description": "Zero-shot mathematical problem solving with step-by-step instruction"
    },

    "few_shot": {
        "template": """Solve math word problems step by step and provide the final numerical answer.

Examples:
{examples}

Now solve this problem:
Problem: {question}

Final Answer:""",
        "description": "Few-shot learning with 2 mathematical reasoning examples"
    }
}

# ================================================
# MATHEMATICAL CULTURAL INDICATORS
# ================================================

MATH_CULTURAL_INDICATORS = {
    "yoruba": {
        "currency": ["naira", "kobo", "₦"],
        "measurements": ["ìwọn", "òsùwọn", "ìgbọn"],
        "cultural_items": ["yam", "cassava", "plantain", "kola nut", "palm oil", "àkàrà", "amala"],
        "cultural_contexts": ["market", "farm", "ọjà", "oko", "owambe", "party"],
        "number_words": ["ọkan", "èjì", "ẹta", "ẹrin", "àrún", "ẹfà", "èje", "ẹjọ", "ẹsàn", "ẹwá"]
    },

    "igbo": {
        "currency": ["naira", "kobo", "₦"],
        "measurements": ["ịtụ", "ọtụtụ"],
        "cultural_items": ["yam", "cassava", "plantain", "kola nut", "palm oil", "ọjị", "nkwọ"],
        "cultural_contexts": ["market", "farm", "ahịa", "ugbo", "festival", "ceremony"],
        "number_words": ["otu", "abụọ", "atọ", "anọ", "ise", "isii", "asaa", "asatọ", "itoolu", "iri"]
    },

    "hausa": {
        "currency": ["naira", "kobo", "₦"],
        "measurements": ["auni", "ma'auni", "gwargwado"],
        "cultural_items": ["millet", "sorghum", "groundnuts", "beans", "tuwo", "miyan", "fura"],
        "cultural_contexts": ["market", "farm", "kasuwa", "gona", "sallah", "festival"],
        "number_words": ["ɗaya", "biyu", "uku", "huɗu", "biyar", "shida", "bakwai", "takwas", "tara", "goma"]
    }
}

# ================================================
# DATA LOADING AND PREPROCESSING
# ================================================

def load_math_excel_data(file_path):
    """Load mathematical reasoning data from Excel sheets"""
    data = {}
    for lang in LANG_CODES:
        try:
            df = pd.read_excel(file_path, sheet_name=lang)
            df = df.head(NUM_EXAMPLES)
            df = df.dropna(subset=['question', 'answer_number'])  # Remove incomplete rows
            data[lang] = df
            print(f"✅ Loaded {len(df)} math problems for {lang}")
        except Exception as e:
            print(f"❌ Error loading {lang} math data: {e}")
    return data

def prepare_math_few_shot_examples(df, n_examples=2):
    """Prepare few-shot examples for mathematical reasoning"""
    examples = []
    if len(df) >= n_examples:
        for i in range(n_examples):
            row = df.iloc[i]
            examples.append(f'Problem: {row["question"]}\nFinal Answer: {row["answer_number"]}')
    return "\n\n".join(examples)

# ================================================
# ENHANCED MATHEMATICAL RESPONSE PROCESSING
# ================================================

def extract_numerical_answer(raw_output):
    """Enhanced numerical answer extraction for African language math problems"""
    if not raw_output:
        return None, "no_response"

    # Remove common non-numeric text
    cleaned = raw_output.lower().replace("final answer:", "").replace("answer:", "")

    # Look for various number formats
    patterns = [
        r'₦?([\d,]+\.?\d*)',  # Currency format
        r'(\d+\.?\d*)\s*naira',  # Naira mentioned
        r'(\d+\.?\d*)\s*kobo',   # Kobo mentioned
        r'(\d+\.?\d*)\s*%',      # Percentage
        r'(\d+\.?\d*)\s*kg',     # Weight
        r'(\d+\.?\d*)\s*meters?', # Distance
        r'(\d+\.?\d*)\s*years?',  # Time
        r'(\d+\.?\d*)\s*days?',   # Time
        r'(-?\d+\.?\d*)',        # Any number
    ]

    extraction_method = "none"

    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, cleaned)
        if matches:
            try:
                # Take the last match (usually the final answer)
                number_str = matches[-1].replace(',', '')
                number = float(number_str)
                extraction_method = f"pattern_{i}"
                return number, extraction_method
            except (ValueError, IndexError):
                continue

    return None, "extraction_failed"

def analyze_mathematical_response(model_name, raw_output, predicted_answer, gold_answer, question):
    """Analyze mathematical reasoning response patterns"""
    analysis = {
        "response_length": len(raw_output) if raw_output else 0,
        "shows_working": False,
        "uses_cultural_context": False,
        "mathematical_language": False,
        "confidence_level": "low",
        "reasoning_quality": "poor",
        "extraction_method": "none"
    }

    if not raw_output:
        return analysis

    output_lower = raw_output.lower()

    # Check if shows mathematical working
    math_indicators = ["multiply", "divide", "add", "subtract", "×", "÷", "+", "-", "=",
                      "step", "first", "then", "therefore", "so", "total", "sum"]
    if any(indicator in output_lower for indicator in math_indicators):
        analysis["shows_working"] = True
        analysis["reasoning_quality"] = "good"

    # Check for cultural context awareness
    for lang_indicators in MATH_CULTURAL_INDICATORS.values():
        for category, items in lang_indicators.items():
            if any(item in output_lower for item in items):
                analysis["uses_cultural_context"] = True
                break

    # Mathematical language usage
    math_terms = ["calculate", "compute", "solve", "equation", "formula", "result"]
    if any(term in output_lower for term in math_terms):
        analysis["mathematical_language"] = True

    # Confidence assessment
    if len(raw_output) > 50 and analysis["shows_working"]:
        analysis["confidence_level"] = "high"
    elif analysis["shows_working"]:
        analysis["confidence_level"] = "medium"

    return analysis

# ================================================
# MATHEMATICAL CULTURAL BIAS ANALYSIS
# ================================================

def detect_math_cultural_elements(question, language):
    """Detect cultural elements in mathematical problems"""
    if language not in MATH_CULTURAL_INDICATORS:
        return {}

    indicators = MATH_CULTURAL_INDICATORS[language]
    analysis = {
        "has_currency": False,
        "has_cultural_items": False,
        "has_cultural_context": False,
        "has_local_measurements": False,
        "cultural_density": 0,
        "specific_elements": []
    }

    question_lower = question.lower()

    # Check each category
    for category, items in indicators.items():
        matches = [item for item in items if item in question_lower]
        if matches:
            analysis[f"has_{category}"] = True
            analysis["specific_elements"].extend([f"{category}: {item}" for item in matches])

    # Calculate cultural density
    total_cultural_words = sum(1 for cat_items in indicators.values() for item in cat_items if item in question_lower)
    analysis["cultural_density"] = (total_cultural_words / len(question.split())) * 100 if question else 0

    return analysis

def analyze_math_response_bias(raw_output, predicted_answer, gold_answer, cultural_elements):
    """Analyze bias in mathematical reasoning responses"""
    bias_analysis = {
        "ignores_cultural_context": False,
        "western_bias": False,
        "inappropriate_assumptions": False,
        "currency_confusion": False,
        "measurement_errors": False,
        "bias_score": 0
    }

    if not raw_output:
        return bias_analysis

    output_lower = raw_output.lower()

    # Cultural context ignorance
    if cultural_elements.get("cultural_density", 0) > 10:  # High cultural content
        if len(raw_output) < 30:  # Very brief response
            bias_analysis["ignores_cultural_context"] = True
            bias_analysis["bias_score"] += 2

    # Western measurement assumptions
    western_units = ["dollars", "$", "pounds", "euros", "€", "miles", "feet", "inches"]
    if any(unit in output_lower for unit in western_units):
        bias_analysis["western_bias"] = True
        bias_analysis["bias_score"] += 3

    inappropriate_indicators = [
        "assume", "typically", "usually", "normally", "standard price",
        "average cost", "common price", "regular price", "standard rate"
    ]
    african_context_words = ["nigeria", "african", "local", "village", "rural", "traditional"]

    has_assumptions = any(indicator in output_lower for indicator in inappropriate_indicators)
    has_african_context = any(word in output_lower for word in african_context_words)

    if has_assumptions and not has_african_context:
        # Making assumptions without considering African context
        bias_analysis["inappropriate_assumptions"] = True
        bias_analysis["bias_score"] += 2

    # Check for Western assumptions in African contexts
    western_assumptions = [
        "dollar store", "walmart", "supermarket chain", "credit card", "bank loan",
        "mortgage", "insurance", "tax rate", "minimum wage", "social security"
    ]
    if any(assumption in output_lower for assumption in western_assumptions):
        bias_analysis["inappropriate_assumptions"] = True
        bias_analysis["bias_score"] += 3

    # Measurement errors - using wrong units or making wrong conversions
    measurement_issues = []

    # Check for metric vs imperial confusion
    if "kg" in output_lower and ("pounds" in output_lower or "lbs" in output_lower):
        measurement_issues.append("metric_imperial_mix")

    # Check for currency conversion errors
    if predicted_answer and gold_answer:
        ratio = predicted_answer / gold_answer if gold_answer != 0 else 0
        # Common conversion errors (USD to Naira is ~1600:1, EUR ~1800:1)
        if 1500 <= ratio <= 2000 or 0.0005 <= ratio <= 0.0007:
            measurement_issues.append("currency_conversion_error")

    # Check for temperature scale errors (Celsius vs Fahrenheit)
    if "temperature" in output_lower or "°" in raw_output:
        if "fahrenheit" in output_lower and ("nigeria" in output_lower or "africa" in output_lower):
            measurement_issues.append("temperature_scale_error")

    # Check for distance unit errors
    distance_context = ["distance", "km", "miles", "meters", "feet"]
    if any(word in output_lower for word in distance_context):
        if "miles" in output_lower and ("nigeria" in output_lower or "africa" in output_lower):
            measurement_issues.append("distance_unit_error")

    # Wrong measurement assumptions for African contexts
    african_measurements = ["bag of rice", "basket", "bowl", "cup", "handful"]
    western_measurements = ["gallon", "quart", "pint", "ounce", "pound"]

    has_african_measure = any(measure in output_lower for measure in african_measurements)
    has_western_measure = any(measure in output_lower for measure in western_measurements)

    if has_western_measure and not has_african_measure and cultural_elements.get("has_cultural_items"):
        measurement_issues.append("cultural_measurement_mismatch")

    if measurement_issues:
        bias_analysis["measurement_errors"] = True
        bias_analysis["bias_score"] += len(measurement_issues)  # More errors = higher bias

    # Currency confusion (using wrong currency)
    currency_issues = []
    if "naira" in output_lower or "₦" in raw_output:
        # Check if answer makes sense in naira context
        if predicted_answer and predicted_answer < 1:  # Suspiciously small for naira
            currency_issues.append("naira_too_small")

    # Check for dollar amounts in Nigerian contexts
    if ("$" in raw_output or "dollar" in output_lower) and cultural_elements.get("has_currency"):
        currency_issues.append("dollar_in_naira_context")

    # Unrealistic naira amounts (too high for context)
    if predicted_answer and predicted_answer > 1000000:  # Over 1 million naira
        if cultural_elements.get("has_cultural_items"):  # For basic items
            currency_issues.append("naira_too_large")

    if currency_issues:
        bias_analysis["currency_confusion"] = True
        bias_analysis["bias_score"] += len(currency_issues)

    return bias_analysis

# ================================================
# ENHANCED MATHEMATICAL EVALUATION
# ================================================

def evaluate_math_model_comprehensive(model_name, model_id, lang, data_df, mode="zero_shot"):
    """Comprehensive mathematical reasoning evaluation"""

    print(f"\n🔢 Evaluating {model_name} on {lang} mathematical reasoning ({mode} mode)")

    results = []
    few_shot_examples = prepare_math_few_shot_examples(data_df) if mode == "few_shot" else ""

    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"{model_name}-{lang}-math"):
        question = row['question']
        gold_answer = float(row['answer_number'])

        # Create prompt
        if mode == "zero_shot":
            prompt = MATH_PROMPT_TEMPLATES["zero_shot"]["template"].format(question=question)
        else:
            prompt = MATH_PROMPT_TEMPLATES["few_shot"]["template"].format(
                examples=few_shot_examples,
                question=question
            )

        try:
            raw_output = call_fireworks_model(model_name, model_id, prompt)
            predicted_answer, extraction_method = extract_numerical_answer(raw_output)

            # Calculate correctness with tolerance for floating point
            is_correct = False
            if predicted_answer is not None:
                is_correct = abs(float(predicted_answer) - gold_answer) < 0.01

            # Cultural analysis
            real_lang = lang_map.get(lang, lang)
            cultural_elements = detect_math_cultural_elements(question, real_lang)

            # Response analysis
            response_analysis = analyze_mathematical_response(
                model_name, raw_output, predicted_answer, gold_answer, question
            )

            # Bias analysis
            bias_analysis = analyze_math_response_bias(
                raw_output, predicted_answer, gold_answer, cultural_elements
            )

            result = {
                "model": model_name,
                "language": lang,
                "mode": mode,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "extraction_method": extraction_method,

                # Cultural elements
                **{f"cultural_{k}": v for k, v in cultural_elements.items()},

                # Response analysis
                **{f"response_{k}": v for k, v in response_analysis.items()},

                # Bias analysis
                **{f"bias_{k}": v for k, v in bias_analysis.items()},

                "prompt_template": mode
            }

            results.append(result)
            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"❌ Error processing question {idx}: {e}")
            continue

    return pd.DataFrame(results)

def call_fireworks_model(model_name, model_id, prompt, retries=3):
    """API call function for mathematical reasoning"""
    for attempt in range(retries):
        try:
            payload = {
                "model": model_id,
                "max_tokens": 512,
                "top_p": 0.9,
                "temperature": 0.1,  # Lower for mathematical precision
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


# ================================================
# MATHEMATICAL REASONING ANALYSIS
# ================================================

def analyze_mathematical_performance(all_results_df):
    """Analyze mathematical reasoning performance patterns"""

    analysis = {
        "accuracy_by_model": {},
        "cultural_impact": {},
        "reasoning_quality": {},
        "error_patterns": {},
        "extraction_success": {}
    }

    print("\n🧮 MATHEMATICAL REASONING ANALYSIS")
    print("=" * 50)

    # Accuracy by model and language
    for model in all_results_df['model'].unique():
        model_data = all_results_df[all_results_df['model'] == model]
        analysis["accuracy_by_model"][model] = {}

        for lang in all_results_df['language'].unique():
            lang_data = model_data[model_data['language'] == lang]
            if len(lang_data) > 0:
                accuracy = lang_data['is_correct'].mean()
                analysis["accuracy_by_model"][model][lang] = accuracy
                print(f"{model} on {lang}: {accuracy:.3f} accuracy")

    # Cultural impact analysis
    cultural_problems = all_results_df[all_results_df['cultural_cultural_density'] > 5]
    non_cultural_problems = all_results_df[all_results_df['cultural_cultural_density'] <= 5]

    if len(cultural_problems) > 0 and len(non_cultural_problems) > 0:
        cultural_acc = cultural_problems['is_correct'].mean()
        non_cultural_acc = non_cultural_problems['is_correct'].mean()

        analysis["cultural_impact"] = {
            "cultural_accuracy": cultural_acc,
            "non_cultural_accuracy": non_cultural_acc,
            "cultural_penalty": non_cultural_acc - cultural_acc
        }

        print(f"\nCultural Impact:")
        print(f"  Cultural problems accuracy: {cultural_acc:.3f}")
        print(f"  Non-cultural problems accuracy: {non_cultural_acc:.3f}")
        print(f"  Cultural penalty: {non_cultural_acc - cultural_acc:.3f}")

    # Reasoning quality analysis
    shows_working = all_results_df[all_results_df['response_shows_working'] == True]
    no_working = all_results_df[all_results_df['response_shows_working'] == False]

    if len(shows_working) > 0 and len(no_working) > 0:
        working_acc = shows_working['is_correct'].mean()
        no_working_acc = no_working['is_correct'].mean()

        analysis["reasoning_quality"] = {
            "with_working_accuracy": working_acc,
            "without_working_accuracy": no_working_acc,
            "working_benefit": working_acc - no_working_acc
        }

        print(f"\nReasoning Quality:")
        print(f"  With working: {working_acc:.3f}")
        print(f"  Without working: {no_working_acc:.3f}")
        print(f"  Working benefit: {working_acc - no_working_acc:.3f}")

    # Error pattern analysis
    incorrect_answers = all_results_df[all_results_df['is_correct'] == False]
    error_patterns = Counter(incorrect_answers['extraction_method'].values)
    analysis["error_patterns"] = dict(error_patterns)

    print(f"\nError Patterns:")
    for pattern, count in error_patterns.most_common():
        print(f"  {pattern}: {count} errors")

    return analysis

# ================================================
# MATHEMATICAL VISUALIZATION
# ================================================

def create_math_visualizations(all_results_df, analysis):
    """Create visualizations for mathematical reasoning analysis"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Accuracy heatmap by model and language
    pivot_accuracy = all_results_df.groupby(['model', 'language'])['is_correct'].mean().reset_index()
    pivot_plot = pivot_accuracy.pivot(index='language', columns='model', values='is_correct')

    sns.heatmap(pivot_plot, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0,0])
    axes[0,0].set_title('Mathematical Reasoning Accuracy')

    # 2. Cultural impact comparison
    if 'cultural_impact' in analysis:
        cultural_acc = analysis['cultural_impact']['cultural_accuracy']
        non_cultural_acc = analysis['cultural_impact']['non_cultural_accuracy']

        axes[0,1].bar(['Cultural Problems', 'Non-Cultural Problems'],
                     [cultural_acc, non_cultural_acc])
        axes[0,1].set_title('Cultural Context Impact on Accuracy')
        axes[0,1].set_ylabel('Accuracy')

    # 3. Reasoning quality impact
    if 'reasoning_quality' in analysis:
        working_acc = analysis['reasoning_quality']['with_working_accuracy']
        no_working_acc = analysis['reasoning_quality']['without_working_accuracy']

        axes[0,2].bar(['Shows Working', 'No Working'], [working_acc, no_working_acc])
        axes[0,2].set_title('Impact of Showing Mathematical Working')
        axes[0,2].set_ylabel('Accuracy')

    # 4. Error patterns
    if 'error_patterns' in analysis:
        error_types = list(analysis['error_patterns'].keys())
        error_counts = list(analysis['error_patterns'].values())

        axes[1,0].bar(error_types, error_counts)
        axes[1,0].set_title('Error Pattern Distribution')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)

    # 5. Response length vs accuracy
    axes[1,1].scatter(all_results_df['response_response_length'],
                     all_results_df['is_correct'].astype(int), alpha=0.6)
    axes[1,1].set_xlabel('Response Length')
    axes[1,1].set_ylabel('Correct (0/1)')
    axes[1,1].set_title('Response Length vs Accuracy')

    # 6. Cultural density distribution
    all_results_df['cultural_cultural_density'].hist(bins=20, ax=axes[1,2], alpha=0.7)
    axes[1,2].set_title('Distribution of Cultural Density')
    axes[1,2].set_xlabel('Cultural Density (%)')
    axes[1,2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mathematical_reasoning_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()



# ================================================
# STATISTICAL SIGNIFICANCE TESTING
# ================================================

def calculate_statistical_significance(results_df):
    """Calculate statistical significance between models"""
    from scipy import stats

    models = results_df['model'].unique()
    sig_results = {}

    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            for lang in results_df['language'].unique():
                # Get accuracy arrays for each model
                m1_data = results_df[(results_df['model']==m1) & (results_df['language']==lang)]
                m2_data = results_df[(results_df['model']==m2) & (results_df['language']==lang)]

                if len(m1_data) > 0 and len(m2_data) > 0:
                    m1_acc = m1_data['is_correct'].values
                    m2_acc = m2_data['is_correct'].values

                    # T-test
                    t_stat, p_val = stats.ttest_ind(m1_acc, m2_acc)

                    # Confidence intervals
                    m1_mean = m1_acc.mean()
                    m1_ci = stats.t.interval(0.95, len(m1_acc)-1,
                                            loc=m1_mean,
                                            scale=stats.sem(m1_acc))

                    m2_mean = m2_acc.mean()
                    m2_ci = stats.t.interval(0.95, len(m2_acc)-1,
                                            loc=m2_mean,
                                            scale=stats.sem(m2_acc))

                    sig_results[f"{m1}_vs_{m2}_{lang}"] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_val),
                        "significant": p_val < 0.05,
                        f"{m1}_mean": float(m1_mean),
                        f"{m1}_ci": [float(m1_ci[0]), float(m1_ci[1])],
                        f"{m2}_mean": float(m2_mean),
                        f"{m2}_ci": [float(m2_ci[0]), float(m2_ci[1])]
                    }

    return sig_results

# ================================================
# MAIN MATHEMATICAL EVALUATION
# ================================================

def main_math_evaluation():
    """Main mathematical reasoning evaluation with comprehensive analysis"""

    print("🧮 Starting Enhanced AfriMGSM Mathematical Reasoning Evaluation...")

    # Load data
    data = load_math_excel_data(EXCEL_FILE)
    if not data:
        print("❌ No data loaded. Please check your Excel file.")
        return

    all_results = []

    # Run evaluations
    for model_name, model_id in MODEL_NAMES.items():
        for lang in LANG_CODES:
            if lang in data:
                for mode in ["zero_shot", "few_shot"]:
                    results_df = evaluate_math_model_comprehensive(
                        model_name, model_id, lang, data[lang], mode
                    )
                    all_results.append(results_df)

    # Combine results
    final_results_df = pd.concat(all_results, ignore_index=True)

    # Mathematical analysis
    print("\n📊 Generating mathematical reasoning analysis...")
    math_analysis = analyze_mathematical_performance(final_results_df)

    # ADD THIS: Statistical significance testing
    print("\n📈 Calculating statistical significance...")
    if len(MODEL_NAMES) > 1:  # Only if comparing multiple models
        sig_tests = calculate_statistical_significance(final_results_df)
        math_analysis["statistical_tests"] = sig_tests

        print("\nStatistical Significance Results:")
        for comparison, results in sig_tests.items():
            if results['significant']:
                print(f"  {comparison}: p={results['p_value']:.4f} (SIGNIFICANT)")
            else:
                print(f"  {comparison}: p={results['p_value']:.4f} (not significant)")

    # Create visualizations
    create_math_visualizations(final_results_df, math_analysis)

    # Save results
    final_results_df.to_csv(f"{OUTPUT_DIR}/math_detailed_results.csv", index=False)

    print("\n💾 Saving per-model results...")
    for model_name in MODEL_NAMES.keys():
        for lang in LANG_CODES:
            model_lang_data = final_results_df[
                (final_results_df['model']==model_name) &
                (final_results_df['language']==lang)
            ]
            if len(model_lang_data) > 0:
                output_file = f"{OUTPUT_DIR}/{model_name}_{lang}_detailed.csv"
                model_lang_data.to_csv(output_file, index=False)
                print(f"  Saved: {output_file}")

    with open(f"{OUTPUT_DIR}/math_analysis_report.json", "w") as f:
        json.dump(math_analysis, f, indent=2, default=str)

    # Print summary
    print(f"\n✅ Mathematical reasoning evaluation complete!")
    print(f"📁 Results saved in: {OUTPUT_DIR}/")
    print(f"📊 Processed {len(final_results_df)} total evaluations")

    # Print key findings
    print("\n🔍 Key Mathematical Reasoning Findings:")
    for model, lang_results in math_analysis["accuracy_by_model"].items():
        avg_acc = np.mean(list(lang_results.values()))
        print(f"{model}: Average accuracy = {avg_acc:.3f}")

    return final_results_df, math_analysis

# Run the evaluation
if __name__ == "__main__":
    results_df, analysis = main_math_evaluation()