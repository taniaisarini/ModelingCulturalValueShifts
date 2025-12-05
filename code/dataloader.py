import pandas as pd
import numpy as np
from scipy import stats # Added import for the t-test

class WVSDataLoader:
    # ... (Previous __init__, load_data, get_waves, get_countries_by_wave, 
    #       get_common_countries, get_questions_by_wave, get_common_questions, 
    #       calculate_averages methods remain the same) ...

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()
        self.averages_df, self.errors_df = self.calculate_averages()

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

    # --- NEW FUNCTION FOR SIGNIFICANCE TESTING ---

    def _perform_welch_ttest(self, wave1, wave2, country, question):
        """
        [HELPER FUNCTION] Extracts positive/zero data for a country/question
        in two waves and performs Welch's two-sample t-test.
        Returns: (t_stat, p_value) or (None, None) if data is insufficient.
        """
        # Data for Wave 1
        df_w1 = self.df[
            (self.df['S002VS'] == wave1) & 
            (self.df['COUNTRY_ALPHA'] == country)
        ]
        data1 = df_w1[question].dropna()
        data1 = data1[data1 >= 0]
        
        # Data for Wave 2
        df_w2 = self.df[
            (self.df['S002VS'] == wave2) & 
            (self.df['COUNTRY_ALPHA'] == country)
        ]
        data2 = df_w2[question].dropna()
        data2 = data2[data2 >= 0]

        # Check for sufficient sample size (typically n >= 2 for t-test)
        if len(data1) < 2 or len(data2) < 2:
            return None, None, None, None # Insufficient data
        
        # Welch's t-test (equal_var=False)
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
        
        # Calculate means for change direction
        mean1 = data1.mean()
        mean2 = data2.mean()
        
        return t_stat, p_value, mean1, mean2


    def get_significant_changes(self, wave1, wave2, alpha=0.05, question_prefix=None):
        """
        Identifies countries and questions with a significant change in the average
        value between two specified waves using Welch's two-sample t-test.

        Args:
            wave1 (int): The number of the first wave.
            wave2 (int): The number of the second wave.
            alpha (float, optional): The significance threshold (p-value). Defaults to 0.05.
            question_prefix (str, optional): An optional prefix to filter the questions.

        Returns:
            pd.DataFrame: A DataFrame detailing the significant changes.
        """
        common_countries = self.get_common_countries(wave1, wave2)
        common_questions = self.get_common_questions(wave1, wave2, prefix=question_prefix)
        
        results = []

        print(f"Analyzing {len(common_questions)} common questions in {len(common_countries)} common countries between Wave {wave1} and Wave {wave2} (Prefix: {question_prefix})...")

        for country in common_countries:
            for question in common_questions:
                # Assumes _perform_welch_ttest returns (t_stat, p_value, mean1, mean2)
                t_stat, p_value, mean1, mean2 = self._perform_welch_ttest(wave1, wave2, country, question)
                
                # Only proceed if the t-test returned valid results and the p-value is significant
                if p_value is not None and p_value < alpha:
                    # Determine the direction of change
                    change_direction = 'Increase' if mean2 > mean1 else 'Drop'
                    
                    results.append({
                        'country': country,
                        'question': question,
                        f'mean_wave_{wave1}': mean1,
                        f'mean_wave_{wave2}': mean2,
                        'change_direction': change_direction,
                        'change_magnitude': mean2 - mean1,
                        't_statistic': t_stat,
                        'p_value': p_value
                    })

        # --- START: MODIFIED ERROR HANDLING ---
        if not results:
            # If no significant changes were found, return an empty DataFrame 
            # with the expected columns to prevent KeyError in subsequent operations (like pd.concat)
            
            expected_columns = [
                'country', 'question', f'mean_wave_{wave1}', f'mean_wave_{wave2}', 
                'change_direction', 'change_magnitude', 't_statistic', 'p_value'
            ]
            
            # Adding a clearer warning message when no results are found
            # (This print statement can be removed if you don't want a message for every empty result)
            # print(f"Warning: No significant changes found for prefix '{question_prefix}'.")
            return pd.DataFrame(columns=expected_columns)
        
        # --- END: MODIFIED ERROR HANDLING ---
        
        # If results exist, create the DataFrame and sort it
        return pd.DataFrame(results).sort_values(by='p_value')