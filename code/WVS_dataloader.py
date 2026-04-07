import pandas as pd
import numpy as np
from scipy import stats # Added import for the t-test
import os
from tqdm import tqdm


class WVSDataLoader:
    def __init__(self, file_path, cache_path=None):
        self.file_path = file_path
        self.cache_path = cache_path
        self.df = self.load_data()

        if self.cache_path and os.path.exists(self.cache_path):
            print("Loading cached averages...")
            self.averages_df = pd.read_csv(self.cache_path)
        else:
            print("Computing averages...")
            self.averages_df, _ = self.calculate_averages()

            if self.cache_path:
                self.averages_df.to_csv(self.cache_path, index=False)


    def load_data(self):
        return pd.read_csv(self.file_path)
    

    def get_waves(self):
        return self.df['S002VS'].dropna().unique()
    

    def get_countries_by_wave(self, wave_number):
        """
        Returns a list of countries (COUNTRY_ALPHA) that participated in a given wave.
        """
        df_wave = self.df[self.df['S002VS'] == wave_number]
        countries = df_wave['COUNTRY_ALPHA'].dropna().unique()
        return countries.tolist()
    

    def get_common_countries(self, wave1, wave2):
        """
        Returns a list of countries that participated in both specified waves.
        """
        countries_wave1 = set(self.get_countries_by_wave(wave1))
        countries_wave2 = set(self.get_countries_by_wave(wave2))
        common_countries = countries_wave1.intersection(countries_wave2)
        return list(common_countries)
    

    def get_questions_by_wave(self, wave_number, exclude_cols=['S002VS', 'COUNTRY_ALPHA']):
        """
        [HELPER FUNCTION] Returns a list of column names (questions) asked in a given wave.
        A column is considered 'asked' if it has at least one POSITIVE value (> 0) for that wave.
        """
        df_wave = self.df[self.df['S002VS'] == wave_number].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # 1. Drop the exclusion columns
        df_questions = df_wave.drop(columns=exclude_cols, errors='ignore')
        
        # 2. Filter for numeric columns only before comparison (THE FIX)
        numeric_cols = df_questions.select_dtypes(include=['number']).columns
        df_numeric_questions = df_questions[numeric_cols]
        
        # 3. Check if numeric values are strictly greater than 0
        # This operation is now safe because we only have numeric data types
        has_positive_value = (df_numeric_questions > 0).sum()
        
        # 4. Keep only columns where the count of positive values is greater than 0
        asked_questions = has_positive_value[has_positive_value > 0].index.tolist()
        
        return asked_questions
    
    def get_common_questions(self, wave_number_1, wave_number_2, prefix=None, exclude_cols=['S002VS', 'COUNTRY_ALPHA']):
        """
        Returns a list of column names (questions) that were asked in *both* specified waves.
        (A question is counted as asked if it has at least one positive value.)
        """
        questions_1 = self.get_questions_by_wave(wave_number_1)
        questions_2 = self.get_questions_by_wave(wave_number_2)

        # Find the intersection
        common_questions = list(set(questions_1).intersection(questions_2))
        
        # Apply the prefix filter if one is provided
        if prefix:
            filtered_questions = [
                q for q in common_questions if q.startswith(prefix)
            ]
            return filtered_questions
        
        return common_questions
    

    def calculate_averages(self):
        """
        Calculate average values for each wave, country, and question combination.
        Ignores negative values and tracks errors where negative values exist.
        
        Returns:
            tuple: (averages_df, errors_df)
                - averages_df: DataFrame with columns [wave, country, question, average_value]
                - errors_df: DataFrame tracking wave/country/question combos with negative values
        """
        results = []
        errors = []
        
        # Get all waves and countries
        waves = self.get_waves()
        exclude_cols = ['S002VS', 'COUNTRY_ALPHA']
        
        for wave in waves:
            # Get data for this wave
            df_wave = self.df[self.df['S002VS'] == wave].copy()
            countries = df_wave['COUNTRY_ALPHA'].dropna().unique()
            
            # Get numeric question columns for this wave
            df_questions = df_wave.drop(columns=exclude_cols, errors='ignore')
            numeric_cols = df_questions.select_dtypes(include=['number']).columns
            
            for country in countries:
                # Get data for this country and wave
                df_country = df_wave[df_wave['COUNTRY_ALPHA'] == country]
                
                for question in numeric_cols:
                    values = df_country[question].dropna()
                    
                    if len(values) == 0:
                        continue  # Skip if no data
                    
                    # Check for negative values
                    has_negative = (values < 0).any()
                    if has_negative:
                        negative_count = (values < 0).sum()
                        total_count = len(values)
                        errors.append({
                            'wave': wave,
                            'country': country,
                            'question': question,
                            'negative_count': negative_count,
                            'total_count': total_count,
                            'negative_percentage': (negative_count / total_count) * 100
                        })
                    
                    # Calculate average ignoring negative values
                    positive_values = values[values >= 0]
                    if len(positive_values) > 0:
                        avg_value = positive_values.mean()
                        results.append({
                            'wave': wave,
                            'country': country,
                            'question': question,
                            'average_value': avg_value,
                            'sample_size': len(positive_values)
                        })
        
        # Create DataFrames
        averages_df = pd.DataFrame(results)
        errors_df = pd.DataFrame(errors)
        
        return averages_df, errors_df
    
    def get_average_result(self, country, wave, question_code):
        """
        Retrieves the pre-calculated average for a specific country, wave, and question.
        """
        mask = (
            (self.averages_df['country'] == country) & 
            (self.averages_df['wave'] == wave) & 
            (self.averages_df['question'] == question_code)
        )
        result = self.averages_df[mask]
        
        if result.empty:
            return None
        return result.iloc[0]['average_value']
    
    def _get_samples(self, wave, country, question):
        df_subset = self.df[
            (self.df['S002VS'] == wave) &
            (self.df['COUNTRY_ALPHA'] == country)
        ]

        values = df_subset[question].dropna()
        values = values[values >= 0]

        return values.values

    def _run_tests(self, x1, x2, tests):
        if len(x1) == 0 or len(x2) == 0:
            return None

        results = {}

        if 'welch' in tests:
            t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False)
            results['welch_t_stat'] = t_stat
            results['welch_p_value'] = p_val

        if 'ks' in tests:
            ks_stat, ks_p = stats.ks_2samp(x1, x2)
            results['ks_stat'] = ks_stat
            results['ks_p_value'] = ks_p

        if 'wasserstein' in tests:
            results['wasserstein_distance'] = stats.wasserstein_distance(x1, x2)

        return results

    # ---------------- MAIN FUNCTION ---------------- #

    def get_significant_changes(
        self,
        wave1,
        wave2,
        question_prefix=None,
        tests=['welch'],
        save_dir=None
    ):
        wave_label = f"wave{wave1}_to_wave{wave2}"

        # ---------- Check if results already exist ---------- #
        if save_dir:
            all_exist = True
            loaded_dfs = []

            for test in tests:
                test_dir = os.path.join(save_dir, test)
                file_path = os.path.join(test_dir, f"{question_prefix}_{wave_label}.csv")

                if not os.path.exists(file_path):
                    all_exist = False
                    break
                else:
                    loaded_dfs.append(pd.read_csv(file_path))

            if all_exist:
                df_merged = loaded_dfs[0]
                for df in loaded_dfs[1:]:
                    df_merged = df_merged.merge(df, on=['country', 'question'], how='outer')
                print(f"Loaded existing results for {wave_label} from {save_dir}")
                return df_merged

        # ---------- If results do not exist, run the tests ---------- #
        common_countries = self.get_common_countries(wave1, wave2)
        common_questions = self.get_common_questions(wave1, wave2, prefix=question_prefix)

        results = []

        print(f"Running tests {tests} on {len(common_questions)} questions across {len(common_countries)} countries")

        for country in tqdm(common_countries, desc="Countries"):
            for question in tqdm(common_questions, desc=f"Questions for {country}", leave=False):
                x1 = self._get_samples(wave1, country, question)
                x2 = self._get_samples(wave2, country, question)

                if len(x1) == 0 or len(x2) == 0:
                    continue

                row = {
                    'country': country,
                    'question': question,
                    f'mean_wave_{wave1}': np.mean(x1),
                    f'mean_wave_{wave2}': np.mean(x2),
                    'change_magnitude': np.mean(x2) - np.mean(x1),
                }

                # -------- Welch --------
                if 'welch' in tests:
                    t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False)
                    row['welch_t_stat'] = t_stat
                    row['welch_p_value'] = p_val

                # -------- KS --------
                if 'ks' in tests:
                    ks_stat, ks_p = stats.ks_2samp(x1, x2)
                    row['ks_stat'] = ks_stat
                    row['ks_p_value'] = ks_p

                # -------- Wasserstein --------
                if 'wasserstein' in tests:
                    w_dist = stats.wasserstein_distance(x1, x2)
                    row['wasserstein_distance'] = w_dist

                results.append(row)

        if not results:
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # ---------- SAVE TO DISK ---------- #
        if save_dir:
            for test in tqdm(tests, desc="Saving results"):
                test_dir = os.path.join(save_dir, test)
                os.makedirs(test_dir, exist_ok=True)

                # Select only relevant columns for each test
                if test == 'welch':
                    cols = ['country', 'question', f'mean_wave_{wave1}', f'mean_wave_{wave2}',
                            'change_magnitude', 'welch_t_stat', 'welch_p_value']
                elif test == 'ks':
                    cols = ['country', 'question', f'mean_wave_{wave1}', f'mean_wave_{wave2}',
                            'change_magnitude', 'ks_stat', 'ks_p_value']
                elif test == 'wasserstein':
                    cols = ['country', 'question', f'mean_wave_{wave1}', f'mean_wave_{wave2}',
                            'change_magnitude', 'wasserstein_distance']
                else:
                    continue

                df_to_save = results_df[cols]
                file_path = os.path.join(test_dir, f"{question_prefix}_{wave_label}.csv")
                df_to_save.to_csv(file_path, index=False)
                print(f"Saved {test} results to {file_path}")

        return results_df


    def get_answer_distribution(
        self, 
        country, 
        wave, 
        question_code, 
        normalize=False, 
        drop_negative=True, 
        dropna=True
    ):
        # Filter data
        df_subset = self.df[
            (self.df['S002VS'] == wave) &
            (self.df['COUNTRY_ALPHA'] == country)
        ]

        if question_code not in df_subset.columns:
            raise ValueError(f"Question '{question_code}' not found in dataset.")

        values = df_subset[question_code]

        # Handle missing values
        if dropna:
            values = values.dropna()

        # Handle negative values (WVS special codes)
        if drop_negative:
            values = values[values >= 0]

        if len(values) == 0:
            return pd.Series(dtype=float)

        # Compute distribution
        distribution = values.value_counts(normalize=normalize).sort_index()

        return distribution