import os
import pandas as pd
import numpy as np
import json
import time
import requests
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# ================================================
# ENHANCED CONFIGURATION FOR AFRIMMLU
# ================================================

LANG_CODES = ["yor", "ibo", "hau"]  # Excel sheet names
MODEL_NAMES = {
    "deepseek": "accounts/fireworks/models/deepseek-v3-0324",
    "qwen": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    "llama": "accounts/fireworks/models/llama-v3p1-405b-instruct"
}

NUM_EXAMPLES = 500  # Using your Excel data
EXCEL_FILE = "masakhane_afrimmlu_500.xlsx"  # Your Excel file
OUTPUT_DIR = "afrimmlu_enhanced_results"
SLEEP_BETWEEN_CALLS = 3
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
# MULTIPLE-CHOICE QA PROMPT TEMPLATES
# ================================================

MMLU_PROMPT_TEMPLATES = {
    "zero_shot": {
        "template": """Answer the following multiple-choice question by selecting the best answer.

Question: {question}

Choices:
{choices}

Provide only the letter (A, B, C, or D) of the correct answer.

Answer:""",
        "description": "Zero-shot multiple-choice question answering"
    },

    "few_shot": {
        "template": """Answer multiple-choice questions by selecting the best answer.

Examples:
{examples}

Now answer this question:
Question: {question}

Choices:
{choices}

Provide only the letter (A, B, C, or D) of the correct answer.

Answer:""",
        "description": "Few-shot learning with 2 question-answer examples"
    }
}

# ================================================
# CULTURAL INDICATORS FOR MMLU
# ================================================

QA_CULTURAL_INDICATORS = {
    "yor": {
        "cultural_concepts": ["yoruba", "ọba", "ifá", "ọrìṣà", "lagos", "ibadan", "abeokuta"],
        "historical": ["oduduwa", "sango", "oyo empire", "benin"],
        "geography": ["niger", "benue", "ogun state"],
        "subjects": ["yoruba culture", "nigerian history", "west africa"],
        "tonal_marks": ["à", "á", "è", "é", "ì", "í", "ò", "ó", "ù", "ú", "ẹ", "ọ", "ṣ"]
    },

    "ibo": {
        "cultural_concepts": ["igbo", "eze", "ozo", "omenala", "enugu", "owerri", "aba"],
        "historical": ["nri kingdom", "aro confederacy", "biafra"],
        "geography": ["anambra", "imo", "abia", "ebonyi"],
        "subjects": ["igbo culture", "nigerian history", "eastern nigeria"],
        "tonal_marks": ["ị", "ụ", "ṅ", "ọ"]
    },

    "hau": {
        "cultural_concepts": ["hausa", "sarki", "emir", "kano", "kaduna", "sokoto"],
        "historical": ["sokoto caliphate", "fulani jihad", "dan fodio"],
        "geography": ["kano", "kaduna", "katsina", "zamfara"],
        "subjects": ["hausa culture", "nigerian history", "northern nigeria"],
        "tonal_marks": []
    }
}

# Subject domains for domain-specific analysis
SUBJECT_DOMAINS = {
    "stem": ["mathematics", "physics", "chemistry", "biology", "computer_science"],
    "humanities": ["history", "philosophy", "literature", "language"],
    "social_sciences": ["economics", "sociology", "political_science", "geography"],
    "law": ["constitutional_law", "international_law", "business_law"],
    "other": []
}

# ================================================
# DATA LOADING AND PREPROCESSING
# ================================================

def load_mmlu_excel_data(file_path):
    """Load multiple-choice QA data from Excel sheets"""
    data = {}
    for lang in LANG_CODES:
        try:
            df = pd.read_excel(file_path, sheet_name=lang)
            df = df.head(NUM_EXAMPLES)
            df = df.dropna(subset=['question', 'choices', 'answer'])

            # Parse choices if they're strings
            if isinstance(df['choices'].iloc[0], str):
                df['choices'] = df['choices'].apply(parse_choices)

            data[lang] = df
            print(f"✅ Loaded {len(df)} questions for {lang}")
        except Exception as e:
            print(f"❌ Error loading {lang} data: {e}")
    return data

def parse_choices(choices_str):
    """Parse choices from string format"""
    try:
        # Try JSON parsing first
        if isinstance(choices_str, list):
            return choices_str
        return json.loads(choices_str.replace("'", '"'))
    except:
        try:
            # Fallback to eval (use cautiously)
            return eval(choices_str)
        except:
            print(f"Failed to parse choices: {choices_str}")
            return []

def format_choices(choices):
    """Format choices as A, B, C, D"""
    return "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])

def prepare_mmlu_few_shot_examples(df, n_examples=2):
    """Prepare few-shot examples for QA"""
    examples = []
    if len(df) >= n_examples:
        for i in range(n_examples):
            row = df.iloc[i]
            choices_text = format_choices(row['choices'])
            examples.append(
                f"Question: {row['question']}\n\nChoices:\n{choices_text}\n\nAnswer: {row['answer']}"
            )
    return "\n\n".join(examples)

# ================================================
# ENHANCED RESPONSE PROCESSING
# ================================================

def extract_answer_choice(raw_output):
    """Extract answer choice (A, B, C, D) from model output"""
    if not raw_output:
        return None, "no_response"

    output_upper = raw_output.strip().upper()

    # Direct single letter
    if output_upper in ['A', 'B', 'C', 'D']:
        return output_upper, "direct_letter"

    # Letter at start
    if output_upper[0] in ['A', 'B', 'C', 'D']:
        return output_upper[0], "first_char"

    # Search for pattern like "Answer: A" or "The answer is B"
    patterns = [
        r'\b([ABCD])\b',
        r'answer\s*:?\s*([ABCD])',
        r'option\s*([ABCD])',
        r'choice\s*([ABCD])',
    ]

    for pattern in patterns:
        match = re.search(pattern, output_upper)
        if match:
            return match.group(1), f"pattern_match"

    return None, "extraction_failed"

def analyze_qa_response(model_name, raw_output, predicted_answer, gold_answer, question, choices):
    """Analyze QA response patterns"""
    analysis = {
        "response_length": len(raw_output) if raw_output else 0,
        "provides_explanation": False,
        "shows_reasoning": False,
        "confident": False,
        "verbose": False,
        "extraction_quality": "poor"
    }

    if not raw_output:
        return analysis

    output_lower = raw_output.lower()

    # Check if provides explanation beyond just the letter
    if len(raw_output) > 10:
        analysis["provides_explanation"] = True
        analysis["verbose"] = len(raw_output) > 50

    # Check for reasoning indicators
    reasoning_words = ["because", "since", "therefore", "thus", "as", "due to", "reason"]
    if any(word in output_lower for word in reasoning_words):
        analysis["shows_reasoning"] = True

    # Confidence indicators
    confidence_words = ["clearly", "obviously", "definitely", "certainly", "correct"]
    uncertainty_words = ["maybe", "perhaps", "possibly", "might", "unsure"]

    if any(word in output_lower for word in confidence_words):
        analysis["confident"] = True
    elif any(word in output_lower for word in uncertainty_words):
        analysis["confident"] = False

    # Extraction quality
    if predicted_answer and predicted_answer == gold_answer:
        analysis["extraction_quality"] = "good"
    elif predicted_answer:
        analysis["extraction_quality"] = "fair"

    return analysis

# ================================================
# CULTURAL BIAS ANALYSIS FOR QA
# ================================================

def detect_qa_cultural_elements(question, choices, subject, language):
    """Detect cultural elements in QA content"""
    if language not in QA_CULTURAL_INDICATORS:
        return {}

    indicators = QA_CULTURAL_INDICATORS[language]
    analysis = {
        "has_cultural_concepts": False,
        "has_historical_content": False,
        "has_geographic_content": False,
        "has_tonal_marks": False,
        "is_culturally_specific": False,
        "cultural_density": 0,
        "subject_domain": categorize_subject(subject),
        "specific_elements": []
    }

    # Combine question and choices for analysis
    full_text = question.lower() + " " + " ".join([str(c).lower() for c in choices])

    # Check cultural concepts
    cultural_matches = [c for c in indicators["cultural_concepts"] if c in full_text]
    if cultural_matches:
        analysis["has_cultural_concepts"] = True
        analysis["specific_elements"].extend(["cultural: " + c for c in cultural_matches])

    # Check historical content
    hist_matches = [h for h in indicators["historical"] if h in full_text]
    if hist_matches:
        analysis["has_historical_content"] = True
        analysis["specific_elements"].extend(["historical: " + h for h in hist_matches])

    # Check geographic content
    geo_matches = [g for g in indicators["geography"] if g in full_text]
    if geo_matches:
        analysis["has_geographic_content"] = True
        analysis["specific_elements"].extend(["geographic: " + g for g in geo_matches])

    # Check tonal marks
    if indicators["tonal_marks"]:
        if any(char in question for char in indicators["tonal_marks"]):
            analysis["has_tonal_marks"] = True

    # Culturally specific questions
    if any([analysis["has_cultural_concepts"],
            analysis["has_historical_content"],
            analysis["has_geographic_content"]]):
        analysis["is_culturally_specific"] = True

    # Cultural density
    total_cultural = len(cultural_matches) + len(hist_matches) + len(geo_matches)
    word_count = len(full_text.split())
    analysis["cultural_density"] = (total_cultural / word_count) * 100 if word_count > 0 else 0

    return analysis

def categorize_subject(subject):
    """Categorize subject into domain"""
    subject_lower = subject.lower().strip()

    for domain, subjects in SUBJECT_DOMAINS.items():
        if any(s in subject_lower for s in subjects):
            return domain
    return "other"

def analyze_qa_response_bias(raw_output, predicted_answer, gold_answer, cultural_elements, question, choices):
    """Analyze bias in QA responses"""
    bias_analysis = {
        "western_bias": False,
        "cultural_ignorance": False,
        "avoids_cultural_answer": False,
        "defaults_to_generic": False,
        "shows_cultural_confusion": False,
        "bias_score": 0
    }

    if not raw_output:
        return bias_analysis

    output_lower = raw_output.lower()

    # Western bias indicators
    western_references = ["western", "european", "american", "british", "usa", "uk", "europe"]
    if any(ref in output_lower for ref in western_references):
        if cultural_elements.get("is_culturally_specific"):
            bias_analysis["western_bias"] = True
            bias_analysis["bias_score"] += 3

    # Cultural ignorance - brief response to culturally rich question
    if cultural_elements.get("cultural_density", 0) > 5:
        if len(raw_output) < 5:  # Just a letter
            bias_analysis["cultural_ignorance"] = True
            bias_analysis["bias_score"] += 2

    # Check if model avoids culturally specific answers
    if predicted_answer and gold_answer and predicted_answer != gold_answer:
        # If wrong, check if it's a cultural question
        if cultural_elements.get("is_culturally_specific"):
            bias_analysis["avoids_cultural_answer"] = True
            bias_analysis["bias_score"] += 1

    # Defaults to generic/Western knowledge
    generic_phrases = ["generally", "typically", "usually", "in most cases", "commonly"]
    if any(phrase in output_lower for phrase in generic_phrases):
        if cultural_elements.get("is_culturally_specific"):
            bias_analysis["defaults_to_generic"] = True
            bias_analysis["bias_score"] += 2

    # Cultural confusion indicators
    confusion_phrases = ["unclear", "ambiguous", "difficult to determine", "not sure", "cannot say"]
    if any(phrase in output_lower for phrase in confusion_phrases):
        if not cultural_elements.get("is_culturally_specific"):  # Question is clear
            bias_analysis["shows_cultural_confusion"] = True
            bias_analysis["bias_score"] += 1

    return bias_analysis

# ================================================
# ENHANCED MMLU EVALUATION
# ================================================

def evaluate_mmlu_model_comprehensive(model_name, model_id, lang, data_df, mode="zero_shot"):
    """Comprehensive MMLU evaluation"""

    print(f"\n📚 Evaluating {model_name} on {lang} QA ({mode} mode)")

    results = []
    few_shot_examples = prepare_mmlu_few_shot_examples(data_df) if mode == "few_shot" else ""

    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"{model_name}-{lang}-qa"):
        question = row['question']
        choices = row['choices']
        gold_answer = row['answer'].strip().upper()
        subject = row.get('subject', 'unknown')

        # Format choices
        choices_text = format_choices(choices)

        # Create prompt
        if mode == "zero_shot":
            prompt = MMLU_PROMPT_TEMPLATES["zero_shot"]["template"].format(
                question=question, choices=choices_text
            )
        else:
            prompt = MMLU_PROMPT_TEMPLATES["few_shot"]["template"].format(
                examples=few_shot_examples, question=question, choices=choices_text
            )

        try:
            raw_output = call_fireworks_model(model_name, model_id, prompt)
            predicted_answer, extraction_method = extract_answer_choice(raw_output)

            is_correct = predicted_answer == gold_answer if predicted_answer else False

            # Cultural analysis
            cultural_elements = detect_qa_cultural_elements(question, choices, subject, lang)

            # Response analysis
            response_analysis = analyze_qa_response(
                model_name, raw_output, predicted_answer, gold_answer, question, choices
            )

            # Bias analysis
            bias_analysis = analyze_qa_response_bias(
                raw_output, predicted_answer, gold_answer, cultural_elements, question, choices
            )

            result = {
                "model": model_name,
                "language": lang,
                "mode": mode,
                "question": question,
                "choices": str(choices),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "raw_output": raw_output,
                "extraction_method": extraction_method,
                "subject": subject,
                "subject_domain": cultural_elements["subject_domain"],

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
    """API call function"""
    for attempt in range(retries):
        try:
            payload = {
                "model": model_id,
                "max_tokens": 100,
                "top_p": 1,
                "temperature": 0,  # Deterministic for QA
                "messages": [{"role": "user", "content": prompt}]
            }

            response = requests.post(url=API_URL, headers=HEADERS, data=json.dumps(payload))

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            elif response.status_code == 429:
                print(f"Rate limit for {model_name}. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"Error {response.status_code}: {response.text[:200]}")

        except Exception as e:
            print(f"Exception for {model_name}: {str(e)[:100]}")
            time.sleep(3)

    return None

# ================================================
# MMLU ANALYSIS
# ================================================

def analyze_mmlu_performance(all_results_df):
    """Analyze QA performance patterns"""

    analysis = {
        "accuracy_by_model": {},
        "accuracy_by_domain": {},
        "cultural_impact": {},
        "extraction_success": {}
    }

    print("\n📖 MULTIPLE-CHOICE QA ANALYSIS")
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

    # Accuracy by subject domain
    for domain in all_results_df['subject_domain'].unique():
        domain_data = all_results_df[all_results_df['subject_domain'] == domain]
        if len(domain_data) > 0:
            domain_acc = domain_data['is_correct'].mean()
            analysis["accuracy_by_domain"][domain] = domain_acc
            print(f"\n{domain}: {domain_acc:.3f} accuracy")

    # Cultural impact
    cultural_qs = all_results_df[all_results_df['cultural_is_culturally_specific'] == True]
    non_cultural_qs = all_results_df[all_results_df['cultural_is_culturally_specific'] == False]

    if len(cultural_qs) > 0 and len(non_cultural_qs) > 0:
        cultural_acc = cultural_qs['is_correct'].mean()
        non_cultural_acc = non_cultural_qs['is_correct'].mean()

        analysis["cultural_impact"] = {
            "cultural_accuracy": cultural_acc,
            "non_cultural_accuracy": non_cultural_acc,
            "cultural_penalty": non_cultural_acc - cultural_acc
        }

        print(f"\nCultural Impact:")
        print(f"  Cultural questions: {cultural_acc:.3f}")
        print(f"  Non-cultural questions: {non_cultural_acc:.3f}")
        print(f"  Cultural penalty: {non_cultural_acc - cultural_acc:.3f}")

    return analysis

# ================================================
# STATISTICAL SIGNIFICANCE TESTING
# ================================================

def calculate_statistical_significance(results_df):
    """Calculate statistical significance between models"""
    models = results_df['model'].unique()
    sig_results = {}

    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            for lang in results_df['language'].unique():
                m1_data = results_df[(results_df['model']==m1) & (results_df['language']==lang)]
                m2_data = results_df[(results_df['model']==m2) & (results_df['language']==lang)]

                if len(m1_data) > 0 and len(m2_data) > 0:
                    m1_acc = m1_data['is_correct'].values
                    m2_acc = m2_data['is_correct'].values

                    t_stat, p_val = stats.ttest_ind(m1_acc, m2_acc)

                    m1_mean = m1_acc.mean()
                    m1_ci = stats.t.interval(0.95, len(m1_acc)-1,
                                            loc=m1_mean, scale=stats.sem(m1_acc))

                    m2_mean = m2_acc.mean()
                    m2_ci = stats.t.interval(0.95, len(m2_acc)-1,
                                            loc=m2_mean, scale=stats.sem(m2_acc))

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
# MAIN EVALUATION
# ================================================

def main_mmlu_evaluation():
    """Main MMLU evaluation"""

    print("📚 Starting Enhanced AfriMMLU QA Evaluation...")

    data = load_mmlu_excel_data(EXCEL_FILE)
    if not data:
        print("❌ No data loaded.")
        return

    all_results = []

    for model_name, model_id in MODEL_NAMES.items():
        for lang in LANG_CODES:
            if lang in data:
                for mode in ["zero_shot", "few_shot"]:
                    results_df = evaluate_mmlu_model_comprehensive(
                        model_name, model_id, lang, data[lang], mode
                    )
                    all_results.append(results_df)

    final_results_df = pd.concat(all_results, ignore_index=True)

    print("\n📊 Generating analysis...")
    mmlu_analysis = analyze_mmlu_performance(final_results_df)

    print("\n📈 Calculating statistical significance...")
    if len(MODEL_NAMES) > 1:
        sig_tests = calculate_statistical_significance(final_results_df)
        mmlu_analysis["statistical_tests"] = sig_tests

    # Save results
    final_results_df.to_csv(f"{OUTPUT_DIR}/mmlu_detailed_results.csv", index=False)

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

    with open(f"{OUTPUT_DIR}/mmlu_analysis_report.json", "w") as f:
        json.dump(mmlu_analysis, f, indent=2, default=str)

    print(f"\n✅ MMLU evaluation complete!")
    print(f"📁 Results saved in: {OUTPUT_DIR}/")

    return final_results_df, mmlu_analysis

if __name__ == "__main__":
    results_df, analysis = main_mmlu_evaluation()