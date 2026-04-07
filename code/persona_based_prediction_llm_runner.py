import os
import json
import re
import pandas as pd
from WVS_dataloader import WVSDataLoader
from thread_gpt_suite.thread_gpt_mp_handler import ThreadGPTMPHandler


class PersonaBasedPredictionLLMRunner:
    def __init__(
        self,
        data_path: str = "/home/tchakrab/gb-scratch/wvs/data/WVS_Time_Series_1981-2022_csv_v5_0.csv",
        survey_path: str = "/home/tchakrab/gb-scratch/wvs/data/parsed_survey_data.json",
        system_prompt_path: str = "/home/tchakrab/gb-scratch/wvs/code/prompts/system/llm_baseline_predict_value_changes.txt",
        user_prompt_path: str = "/home/tchakrab/gb-scratch/wvs/code/prompts/user/llm_baseline_predict_value_changes.txt",
        results_dir: str = "/home/tchakrab/gb-scratch/wvs/results/results_new/",
        df = None,
    ):
        self.data_path = data_path
        self.survey_path = survey_path
        self.system_prompt_path = system_prompt_path
        self.user_prompt_path = user_prompt_path
        self.results_dir = results_dir

        self.wave_data = None
        self.survey_questions = None
        self.system_prompt = None
        self.user_prompt = None
        
        if df is None:
            self.loader = WVSDataLoader(
                    file_path='/home/tchakrab/gb-scratch/wvs/data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                    cache_path='/home/tchakrab/gb-scratch/wvs/data/averages_cached.csv')
            self.df = self.loader.df
        else:
            self.df = df

        self._load_static_resources()

    
    def run_full_pipeline(self, country, wave, target_questions=None):
        prompts, prompt_to_code = self.create_prompts(country, wave, target_questions)
        raw_results_path = self.run_llm(prompts, prompt_to_code, country, wave)
        parsed_results_path = self.parse_llm_responses(raw_results_path, country, wave)
        return raw_results_path, parsed_results_path

    # ------------------------------------------------------------------
    # 1. Load data (wave + country)
    # ------------------------------------------------------------------
    def _load_data(self, country, wave):
        print(f"Loading Data for {country}, Wave {wave}...")
        self.wave_data = self.df[
            (self.df["S002VS"] == wave) & (self.df["COUNTRY_ALPHA"] == country)
        ]
        return self.wave_data

    # ------------------------------------------------------------------
    # 2. Create prompts
    # ------------------------------------------------------------------
    def create_prompts(self, country, wave, target_questions=None):
        """
        Generates prompts where each prompt contains up to 10 questions.
        Deduplicates prompts so identical demographic profiles share one LLM call.
        """
        print("Creating prompts (Batched Question Mode, Deduplicated)...")
        self.wave_data = self._load_data(country, wave)
        prompts = []
        prompt_to_code = {}

        filtered_questions = {}
        
        # 1. Identify which questions to process
        if target_questions is not None:
            special_code = "A027_A029_A030_A032_A034_A035_A038_A039_A040_A041_A042_A043B"
            sub_codes = {
                "A027", "A029", "A030", "A032", "A034", "A035", 
                "A038", "A039", "A040", "A041", "A042", "A043B"
            }
            special_code_added = False

            for qcode in target_questions:
                if qcode in sub_codes:
                    if not special_code_added:
                        if special_code in self.survey_questions:
                            filtered_questions[special_code] = self.survey_questions[special_code]
                            special_code_added = True
                        else:
                            print(f"Warning: Special code {special_code} not found in survey data.")
                            special_code_added = True 
                else:
                    if qcode in self.survey_questions and qcode in self.wave_data.columns:
                        filtered_questions[qcode] = self.survey_questions[qcode]
                    else:
                        print(f"Warning: {qcode} not found in survey data or wave columns.")
        else:
            print("Nothing to run!")
            return [], {}

        question_items = list(filtered_questions.items())

        # 2. Generate prompts (Deduplicated mapping in batches of 10)
        for row in self.wave_data.itertuples():
            # Common demographic context
            base_args = {
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

            # Process questions in batches of 10
            for i in range(0, len(question_items), 10):
                batch_questions = question_items[i : i + 10]
                format_args = base_args.copy()
                batch_question_codes = []

                # Populate the 10 question slots
                for j, (qcode, qdata) in enumerate(batch_questions, 1):
                    format_args[f"question{j}"] = qdata.get("parsed_question", "?")
                    format_args[f"options{j}"] = ", ".join(qdata.get("choices", []))
                    batch_question_codes.append(qcode)

                # Pad remaining empty slots to ensure string formatting doesn't throw KeyErrors
                for j in range(len(batch_questions) + 1, 11):
                    format_args[f"question{j}"] = ""
                    format_args[f"options{j}"] = ""

                filled_prompt = eval(self.user_prompt.format(**format_args))
                
                # Deduplication logic: map one prompt to many person_ids
                if filled_prompt not in prompt_to_code:
                    prompts.append(filled_prompt)
                    prompt_to_code[filled_prompt] = {
                        "person_ids": [str(row.Index)], 
                        "question_codes": batch_question_codes, 
                    }
                else:
                    prompt_to_code[filled_prompt]["person_ids"].append(str(row.Index))

        print(f"Generated {len(prompts)} UNIQUE prompts.")
        return prompts, prompt_to_code

    # ------------------------------------------------------------------
    # 3. Run LLM and store raw results
    # ------------------------------------------------------------------
    def run_llm(self, prompts, prompt_to_code, country, wave, num_workers=350):
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
                person_ids = code_info['person_ids']
                question_codes = code_info['question_codes']
                
                # Map the single response back to all users sharing the prompt
                for person_id in person_ids:
                    batch_key = f"{person_id}_batch_" + ",".join(map(str, question_codes))

                    results_json[batch_key] = {
                        "prompt": prompt,
                        "person_id": person_id,
                        "question_codes": question_codes,
                        "response": response,
                    }

        out_path = self._raw_results_path(country, wave)
        with open(out_path, "w") as f:
            json.dump(results_json, f, indent=4)

        return out_path

    # ------------------------------------------------------------------
    # 4. Parse LLM responses and store parsed results
    # ------------------------------------------------------------------
    def parse_llm_responses(self, raw_results_path, country, wave):
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

        out_path = self._parsed_results_path(country, wave)
        with open(out_path, "w") as f:
            json.dump(parsed, f, indent=4)

        return out_path
    # -----------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_static_resources(self):
        with open(self.survey_path) as f:
            self.survey_questions = json.load(f)

        with open(self.system_prompt_path) as f:
            self.system_prompt = eval(f.read())

        with open(self.user_prompt_path) as f:
            self.user_prompt = f.read()

    def _raw_results_path(self, country, wave):
        return f"{self.results_dir}/persona_based_prediction_raw_{country}_{wave}.json"

    def _parsed_results_path(self, country, wave):
        return f"{self.results_dir}/persona_based_prediction_parsed_{country}_{wave}.json"

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