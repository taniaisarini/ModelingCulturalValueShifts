import os
import json
import re
import pandas as pd
from dataloader import WVSDataLoader
from thread_gpt_suite.thread_gpt_mp_handler import ThreadGPTMPHandler


class WVSLLMBaselineRunner:
    def __init__(
        self,
        country: str,
        wave_number: int,
        data_path: str = "/home/tchakrab/gb-scratch/wvs/data/WVS_Time_Series_1981-2022_csv_v5_0.csv",
        survey_path: str = "/home/tchakrab/gb-scratch/wvs/data/parsed_survey_data.json",
        system_prompt_path: str = "/home/tchakrab/gb-scratch/wvs/code/prompts/system/llm_baseline_predict_value_changes.txt",
        user_prompt_path: str = "/home/tchakrab/gb-scratch/wvs/code/prompts/user/llm_baseline_predict_value_changes.txt",
        results_dir: str = "/home/tchakrab/gb-scratch/wvs/results",
        df = None,
    ):
        self.country = country
        self.wave_number = wave_number
        self.data_path = data_path
        self.survey_path = survey_path
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        self.results_dir = results_dir

        self.wave_data = None
        self.survey_questions = None
        self.system_prompt = None
        self.user_prompt = None
        self.df = df

        self._load_static_resources()

    # ------------------------------------------------------------------
    # 1. Load data (wave + country)
    # ------------------------------------------------------------------
    def load_data(self):
        
        print("Loading Data...")
        df = self.df
        if df is None:
            loader = WVSDataLoader(self.data_path)
            df = loader.df

        self.wave_data = df[
            (df["S002VS"] == self.wave_number)
            & (df["COUNTRY_ALPHA"] == self.country)
        ]

        return self.wave_data

    # ------------------------------------------------------------------
    # 2. Create prompts
    # ------------------------------------------------------------------
    def create_prompts(self):
        print("Creating prompts...")
        prompts = []
        prompt_to_code = {}

        # Filter questions to only include those where at least 50% of people answered
        filtered_questions = {}
        total_people = len(self.wave_data)
        
        for qcode, qdata in self.survey_questions.items():
            special_code = "A027_A029_A030_A032_A034_A035_A038_A039_A040_A041_A042_A043B"
            sub_codes = ["A027", "A029", "A030", "A032", "A034", "A035", "A038", "A039", "A040", "A041", "A042", "A043B"]
            
            if qcode == special_code:
                # Special handling for combined questions
                if all(code in self.wave_data.columns and pd.api.types.is_numeric_dtype(self.wave_data[code]) for code in sub_codes):
                    valid_count = 0
                    for idx in self.wave_data.index:
                        row = self.wave_data.loc[idx]
                        if any(pd.notna(row[code]) and row[code] > 0 for code in sub_codes):
                            valid_count += 1
                    response_rate = valid_count / total_people
                    if response_rate >= 0.5:
                        filtered_questions[qcode] = qdata
                continue
            
            if qcode in sub_codes:
                continue  # Skip individual sub-codes
            
            if qcode in self.wave_data.columns and pd.api.types.is_numeric_dtype(self.wave_data[qcode]):
                # Count valid responses (> 0 and not NaN)
                valid_responses = self.wave_data[qcode].fillna(-1)
                valid_count = (valid_responses > 0).sum()
                response_rate = valid_count / total_people
                
                if response_rate >= 0.5:  # At least 50% answered
                    filtered_questions[qcode] = qdata
        
        print(f"Filtered questions: {len(filtered_questions)} out of {len(self.survey_questions)} questions have ≥50% response rate")
        question_items = list(filtered_questions.items())

        for row in self.wave_data.sample(50).itertuples():
            for i in range(0, len(question_items), 10):
                batch_questions = question_items[i : i + 10]

                format_args = {
                    "X003": row.X003,
                    "X001": self.X001_map.get(row.X001, "?"),
                    "country": row.COUNTRY_ALPHA,
                    "X002": row.X002,
                    "X049": self.X049_map.get(row.X049, "?"),
                    "X026": self.X026_map.get(row.X026, "?"),
                    "X007": self.X007_map.get(row.X007, "?"),
                    "X011": row.X011,
                    "X025R": self.X025R_map.get(row.X025R, "?"),
                    "X028": self.X028_map.get(row.X028, "?"),
                    "X036": self.X036_map.get(row.X036, "?"),
                    "X040": self.X040_map.get(row.X040, "?"),
                    "X044": self.X044_map.get(row.X044, "?"),
                    "X045": self.X045_map.get(row.X045, "?"),
                    "X047R_WVS": self.X047R_WVS_map.get(row.X047R_WVS, "?"),
                }

                question_codes = []
                for j, (qcode, qdata) in enumerate(batch_questions, 1):
                    format_args[f"question{j}"] = qdata.get("parsed_question", "?")
                    format_args[f"options{j}"] = ", ".join(qdata.get("choices", []))
                    question_codes.append(qcode)

                for j in range(len(batch_questions) + 1, 11):
                    format_args[f"question{j}"] = ""
                    format_args[f"options{j}"] = ""

                filled_prompt = eval(self.user_prompt.format(**format_args))
                if filled_prompt in prompts:
                    continue
                prompts.append(filled_prompt)

                prompt_to_code[filled_prompt] = {
                    "person_id": str(row.Index),
                    "question_codes": question_codes,
                }

        return prompts, prompt_to_code

    # ------------------------------------------------------------------
    # 3. Run LLM and store raw results
    # ------------------------------------------------------------------
    def run_llm(self, prompts, prompt_to_code, num_workers=350):
        print("Running LLM...")
        handler = ThreadGPTMPHandler(
            api_key=os.environ.get("OPENAI_API_KEY"),
            num_worker=num_workers,
        )

        batch = [
            {
                "questions": [prompt],
                "model_name": "gpt-4.1-mini",
                "task_desc": self.system_prompt,
            }
            for prompt in prompts
        ]

        handler.add_batch(batch)
        results = handler.process(rerun_on_error=True)

        results_json = {}
        for result in results:
            for prompt, response in result.items():
                code_info = prompt_to_code[prompt]
                person_id = code_info['person_id']
                question_codes = code_info['question_codes']
                batch_key = f"{person_id}_batch_" + ",".join(map(str, question_codes))


                results_json[batch_key] = {
                    "person_id": code_info["person_id"],
                    "question_codes": code_info["question_codes"],
                    "response": response,
                }

        out_path = self._raw_results_path()
        with open(out_path, "w") as f:
            json.dump(results_json, f, indent=4)

        return out_path

    # ------------------------------------------------------------------
    # 4. Parse LLM responses and store parsed results
    # ------------------------------------------------------------------
    def parse_llm_responses(self, raw_results_path):
        print("Parsing results...")
        with open(raw_results_path, "r") as f:
            results_json = json.load(f)

        parsed = {}
        special_code = "A027_A029_A030_A032_A034_A035_A038_A039_A040_A041_A042_A043B"
        sub_codes = ["A027", "A029", "A030", "A032", "A034", "A035", "A038", "A039", "A040", "A041", "A042", "A043B"]

        def extract_answer(text, qnum):
            sections = re.split(r"(?:Question|Answer)\s*\d+", text)
            if len(sections) > qnum:
                m = re.search(r"\b(\d+)\b", sections[qnum])
                if m:
                    return int(m.group(1))
            return None

        for _, batch in results_json.items():
            pid = batch["person_id"]
            qcodes = batch["question_codes"]
            response = batch["response"]

            if pid not in parsed:
                parsed[pid] = {}

            row = self.wave_data.loc[int(pid)]

            for i, qcode in enumerate(qcodes, 1):
                # Special handling for combined question code
                if qcode == special_code:
                    # Get which sub-codes the person actually selected (actual > 0)
                    actual_selected = [code for code in sub_codes if pd.notna(row[code]) and row[code] > 0]
                    
                    # Extract which sub-codes the LLM thinks were selected from the response
                    llm_selected = []
                    for code in sub_codes:
                        if code in response:
                            llm_selected.append(code)
                    
                    parsed[pid][qcode] = {
                        "question": self.survey_questions[qcode]["parsed_question"],
                        "actual_response": actual_selected,
                        "llm_response": llm_selected,
                        "is_special_code": True
                    }
                else:
                    parsed[pid][qcode] = {
                        "question": self.survey_questions[qcode]["parsed_question"],
                        "actual_response": (
                            int(row[qcode]) if pd.notna(row[qcode]) else None
                        ),
                        "llm_response": extract_answer(response, i),
                        "is_special_code": False
                    }

        out_path = self._parsed_results_path()
        with open(out_path, "w") as f:
            json.dump(parsed, f, indent=4)

        return out_path

    # ------------------------------------------------------------------
    # 5. Compute accuracy metrics
    # ------------------------------------------------------------------
    def compute_accuracy(self, parsed_results_path=None, filter_func=None):
        """
        Compute accuracy metrics between LLM responses and actual responses.
        
        Args:
            parsed_results_path (str, optional): Path to parsed results JSON. 
                                               If None, uses default path.
            filter_func (callable, optional): Function to filter data. Should take 
                                             (person_id, question_code, data_dict) 
                                             and return True to include the data point.
        
        Returns:
            dict: Dictionary containing accuracy metrics:
                - overall_accuracy: Fraction of correct predictions
                - total_predictions: Total number of predictions made
                - correct_predictions: Number of correct predictions
                - question_accuracies: Per-question accuracy breakdown
                - person_accuracies: Per-person accuracy breakdown
        """
        if parsed_results_path is None:
            parsed_results_path = self._parsed_results_path()
        
        with open(parsed_results_path, "r") as f:
            parsed_results = json.load(f)
        
        correct_predictions = 0
        total_predictions = 0
        question_stats = {}  # {question_code: {'correct': int, 'total': int}}
        person_stats = {}    # {person_id: {'correct': int, 'total': int}}
        
        special_code = "A027_A029_A030_A032_A034_A035_A038_A039_A040_A041_A042_A043B"
        
        for person_id, questions in parsed_results.items():
            person_correct = 0
            person_total = 0
            
            for question_code, data in questions.items():
                # Apply filter if provided
                if filter_func and not filter_func(person_id, question_code, data):
                    continue
                
                actual = data.get('actual_response')
                llm_response = data.get('llm_response')
                is_special = data.get('is_special_code', False)
                
                # Skip if either response is missing
                if actual is None or llm_response is None:
                    continue
                
                total_predictions += 1
                person_total += 1
                
                # Initialize question stats if needed
                if question_code not in question_stats:
                    question_stats[question_code] = {'correct': 0, 'total': 0, 'intersection_sum': 0}
                
                question_stats[question_code]['total'] += 1
                
                # Handle special code with set intersection
                if is_special:
                    actual_set = set(actual) if isinstance(actual, list) else set()
                    llm_set = set(llm_response) if isinstance(llm_response, list) else set()
                    intersection = len(actual_set & llm_set)
                    question_stats[question_code]['intersection_sum'] += intersection
                    person_correct += intersection
                else:
                    # Check if prediction is correct
                    if actual == llm_response:
                        correct_predictions += 1
                        person_correct += 1
                        question_stats[question_code]['correct'] += 1
                    else:
                        if 'correct' not in question_stats[question_code]:
                            question_stats[question_code]['correct'] = 0
            
            # Store person stats if they answered any questions
            if person_total > 0:
                person_stats[person_id] = {
                    'correct': person_correct,
                    'total': person_total,
                    'accuracy': person_correct / person_total
                }
        
        # Calculate question accuracies
        question_accuracies = {}
        for qcode, stats in question_stats.items():
            if stats['total'] > 0:
                if qcode == special_code:
                    # For special code, accuracy is based on set intersection overlap
                    accuracy = stats['intersection_sum'] / stats['total'] if stats['total'] > 0 else 0
                    question_accuracies[qcode] = {
                        'accuracy': accuracy,
                        'intersection_sum': stats['intersection_sum'],
                        'total': stats['total'],
                        'question_text': self.survey_questions[qcode]['parsed_question']
                    }
                else:
                    correct_count = stats.get('correct', 0)
                    question_accuracies[qcode] = {
                        'accuracy': correct_count / stats['total'],
                        'correct': correct_count,
                        'total': stats['total'],
                        'question_text': self.survey_questions[qcode]['parsed_question']
                    }
        
        # Calculate overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        results = {
            'overall_accuracy': overall_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'question_accuracies': question_accuracies,
            'person_accuracies': person_stats
        }
        
        return results
    
    def save_accuracy_results(self, accuracy_results, suffix=""):
        """
        Save accuracy results to JSON file.
        
        Args:
            accuracy_results (dict): Results from compute_accuracy()
            suffix (str): Optional suffix for filename
        """
        filename = f"accuracy_results_{self.country}_{self.wave_number}{suffix}.json"
        out_path = f"{self.results_dir}/{filename}"
        
        with open(out_path, "w") as f:
            json.dump(accuracy_results, f, indent=4)
        
        print(f"Accuracy results saved to: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_static_resources(self):
        with open(self.survey_path) as f:
            self.survey_questions = json.load(f)

        with open(self.system_prompt_path) as f:
            self.system_prompt = eval(f.read())

        with open(self.user_prompt_path) as f:
            self.user_prompt = f.read()

    def _raw_results_path(self):
        return f"{self.results_dir}/llm_baseline_predict_value_changes_results_{self.country}_{self.wave_number}.json"

    def _parsed_results_path(self):
        return f"{self.results_dir}/llm_baseline_predict_value_changes_parsed_{self.country}_{self.wave_number}.json"

    # ------------------------------------------------------------------
    # Mappings (unchanged)
    # ------------------------------------------------------------------
    X001_map = {1: "male", 2: "female"}
    X007_map = {
        1: "married",
        2: "living together as married",
        3: "divorced",
        4: "separated",
        5: "widowed",
        6: "either single or never married",
    }
    X026_map = {0: "do not live", 1: "live"}
    X028_map = {
        1: "employed full-time",
        2: "employed part-time",
        3: "self-employed",
        4: "retired",
        5: "housewife",
        6: "student",
        7: "unemployed",
        8: "other",
    }
    X036_map = {
        11: "employer or manager of an establishment with 500 or more employees",
        12: "employer or manager of an establishment with 100 or more employees",
        13: "employer or manager of an establishment with 10 or more employees",
        14: "employer or manager of an establishment with fewer than 500 employees",
        15: "employer or manager of an establishment with fewer than 100 employees",
        16: "employer or manager of an establishment with fewer than 10 employees",
        21: "professional worker",
        22: "middle-level non-manual office worker",
        23: "supervisory non-manual office worker",
        24: "junior-level non-manual worker",
        25: "non-manual office worker",
        31: "foreman or supervisor",
        32: "skilled manual worker",
        33: "semi-skilled manual worker",
        34: "unskilled manual worker",
        41: "farmer with own farm",
        42: "agricultural worker",
        51: "member of the armed forces",
        61: "never had a job",
        81: "other",
    }
    X040_map = {1: "are", 2: "are not"}
    X044_map = {
        1: "was able to save money",
        2: "just got by",
        3: "spent some savings",
        4: "spent savings and borrowed money",
    }
    X045_map = {
        1: "upper class",
        2: "upper middle class",
        3: "lower middle class",
        4: "working class",
        5: "lower class",
    }
    X047R_WVS_map = {1: "low income", 2: "medium income", 3: "high income"}
    X049_map = {
        1: "a town with under 2,000 people",
        2: "a town with 2,000–5,000 people",
        3: "a town with 5,000–10,000 people",
        4: "a town with 10,000–20,000 people",
        5: "a town with 20,000–50,000 people",
        6: "a town with 50,000–100,000 people",
        7: "a city with 100,000–500,000 people",
        8: "a city with over 500,000 people",
    }
    X025R_map = {1: "lower level", 2: "middle level", 3: "upper level"}
