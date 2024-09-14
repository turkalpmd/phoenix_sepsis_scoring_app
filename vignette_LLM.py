import os
import json
import re
import warnings
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import phoenix as phx
import numpy as np

warnings.filterwarnings('ignore')
load_dotenv()

class SepsisCaseProcessor:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    def process_sepsis_case(self, case_summary, pid):
        prompt = f"""
        Based on the provided case summary:

        {case_summary}

        Create a JSON object representing the data entry for a sepsis patient. The JSON object should include the following keys with their corresponding values, adhering to the specified guidelines.

        Construct a JSON object with these keys and notes:
        - 'pid': Patient identification number. Use {pid} for this value.
        - 'age': Patient's age in months. Calculate from the case summary, converting years to months always        
        - 'fio2': Fraction of inspired oxygen.
        - 'pao2': Partial pressure of oxygen in arterial blood (mmHg). Notes: pao2 is not po2!
        - 'spo2': Pulse oximetry value (only valid when SpO2 ≤ 97%).
        - 'vent': Indicator for invasive mechanical ventilation (0 for no, 1 for yes).
        - 'gcs_total': Total Glasgow Coma Scale score (integer from 3 to 15).
        - 'pupil': Indicates if pupils are 'reactive' or 'fixed'. If it is not mentioned in the case use null
        - 'platelets': Platelet count measured in 1,000 per microliter.
        - 'inr': International Normalized Ratio.
        - 'd_dimer': D-dimer level in mg/L FEU.
        - 'fibrinogen': Fibrinogen level in mg/dL.
        - 'dbp': Diastolic blood pressure (mmHg).
        - 'sbp': Systolic blood pressure (mmHg).
        - 'lactate': Lactate level in mmol/L (arterial or venous). Note: Lactate dehydrogenase is not lactate.
        - 'dobutamine': Indicator for receiving systemic dobutamine (0 for no, 1 for yes).
        - 'dopamine': Indicator for receiving systemic dopamine (0 for no, 1 for yes).
        - 'epinephrine': Indicator for receiving systemic epinephrine (0 for no, 1 for yes).
        - 'milrinone': Indicator for receiving systemic milrinone (0 for no, 1 for yes).
        - 'norepinephrine': Indicator for receiving systemic norepinephrine (0 for no, 1 for yes).
        - 'vasopressin': Indicator for receiving systemic vasopressin (0 for no, 1 for yes). Note: vasopressin is just vasopressors drug.
        - 'glucose': Glucose level in mg/dL.
        - 'anc': Absolute Neutrophil Count in 1,000 cells per cubic millimeter. If reported as a percentage of the total white blood cell (WBC) count, calculate using ANC = (Percentage × WBC) / 100. Sometimes, the differential includes percentages of bands and segmented neutrophils (e.g., n% bands and m% neutrophils/segmented); in such cases, add the percentages of bands and neutrophils to get the total neutrophil percentage before calculating ANC.
        - 'alc': Absolute Lymphocyte Count in 1,000 cells per cubic millimeter. If not directly provided, calculate using ALC = WBC - ANC.
        - 'creatinine': Creatinine level in mg/dL.
        - 'bilirubin': Bilirubin level in mg/dL.
        - 'alt': Alanine aminotransferase level in IU/L.

        Guidelines:

        1. Patient ID: Use {pid} as the value for 'pid'.
        2. Age Calculation: Calculate the patient's age in months. If age is given in years, multiply by 12.
        3. Missing Values: For any values not mentioned in the case summary, use null.
        4. Indicator Values: For 'vent' and drug administration keys, use 0 for 'no' and 1 for 'yes'.
        5. Data Types: Ensure numerical values use appropriate data types (int or float).
        6. Unit Conversions: Perform necessary unit conversions to match the specified units.
        7. Derived Values: If certain values are missing but can be derived from other data, compute them accordingly.

        Please ensure:

        - No additional text: Do not include any explanations, notes, or code snippets.
        - Valid JSON format: The output must be a valid JSON object, using double quotes for keys and string values, and null for missing values.

        Output:
        Return only the JSON object containing the patient's data.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical data analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                presence_penalty=0,
                max_tokens= 2048,
                frequency_penalty=0,
                n=5,
                top_p=1,
                
                
                stop=["\n\n", "<|endoftext|>"]
            )
            response_text = response.choices[0].message.content.strip()
            # print("Raw response text:", response_text)  # Debug output

            # Extract JSON object from the response
            match = re.search(r'({.*})', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                # Replace single quotes with double quotes for valid JSON
                json_str = json_str.replace("'", '"')
                # Replace Python None with JSON null
                json_str = json_str.replace("None", "null")
                try:
                    patient_data = json.loads(json_str)
                    print(f"Total token usage for this run: {response.usage.total_tokens}")
                    return patient_data
                except json.JSONDecodeError as e:
                    print("Failed to decode JSON after extraction:")
                    print(e)
                    print("Extracted JSON string causing error:", json_str)
                    return None
            else:
                print("No JSON object found in the response.")
                return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def create_dataframe_from_result(self, result):
        df = pd.DataFrame([result])
        df.replace(to_replace=[None, 'null'], value=np.nan, inplace=True)
        return df

    def calculate_phoenix_scores(self, sepsis_df):
        # Handle NaN values in 'pupil' column before processing
        fixed_pupils = (sepsis_df["pupil"].fillna('').astype(str).str.lower() == "fixed").astype(int)
        return phx.phoenix(
            pf_ratio=sepsis_df["pao2"] / sepsis_df["fio2"],
            sf_ratio=sepsis_df["spo2"] / sepsis_df["fio2"],
            imv=sepsis_df["vent"],
            other_respiratory_support=(sepsis_df["fio2"] > 0.21).astype(int).to_numpy(),
            vasoactives=sepsis_df["dobutamine"] + sepsis_df["dopamine"] + sepsis_df["epinephrine"] + sepsis_df["milrinone"] + sepsis_df["norepinephrine"] + sepsis_df["vasopressin"],
            lactate=sepsis_df["lactate"],
            age=sepsis_df["age"],
            map=sepsis_df["dbp"] + (sepsis_df["sbp"] - sepsis_df["dbp"]) / 3,
            platelets=sepsis_df['platelets'],
            inr=sepsis_df['inr'],
            d_dimer=sepsis_df['d_dimer'],
            fibrinogen=sepsis_df['fibrinogen'],
            gcs=sepsis_df["gcs_total"],
            fixed_pupils=fixed_pupils
        )

    def calculate_phoenix_8_scores(self, sepsis_df):
        # Handle NaN values in 'pupil' column before processing
        fixed_pupils = (sepsis_df["pupil"].fillna('').astype(str).str.lower() == "both-fixed").astype(int)
        return phx.phoenix8(
            pf_ratio = sepsis_df["pao2"] / sepsis_df["fio2"],
            sf_ratio = np.where(sepsis_df["spo2"] <= 97, sepsis_df["spo2"] / sepsis_df["fio2"], np.nan),
            imv      = sepsis_df["vent"],
            other_respiratory_support = (sepsis_df["fio2"] > 0.21).astype(int).to_numpy(),
            vasoactives = sepsis_df["dobutamine"] + sepsis_df["dopamine"] + sepsis_df["epinephrine"] + sepsis_df["milrinone"] + sepsis_df["norepinephrine"] + sepsis_df["vasopressin"],
            lactate = sepsis_df["lactate"],
            map = sepsis_df["dbp"] + (sepsis_df["sbp"] - sepsis_df["dbp"]) / 3,
            platelets = sepsis_df['platelets'],
            inr = sepsis_df['inr'],
            d_dimer = sepsis_df['d_dimer'],
            fibrinogen = sepsis_df['fibrinogen'],
            gcs = sepsis_df["gcs_total"],
            fixed_pupils = fixed_pupils,
            glucose = sepsis_df["glucose"],
            anc = sepsis_df["anc"],
            alc = sepsis_df["alc"],
            creatinine = sepsis_df["creatinine"],
            bilirubin = sepsis_df["bilirubin"],
            alt = sepsis_df["alt"],
            age = sepsis_df["age"]
        )

    def print_json_output(self, data):
        json_output = json.dumps(data, indent=2)
        print(json_output)

if __name__ == "__main__":

    # Example usage
    processor = SepsisCaseProcessor()
    case_summary = """
    A 6-year-old boy with a history of prematurity presents with respiratory distress. He has an oxygen saturation of 89% on room air. In the emergency department, he is started on non-invasive positive pressure ventilation. His level of consciousness deteriorates rapidly: Glasgow Coma Scale: 2 for eye response + 2 for verbal response + 4 for motor response = 8. He is intubated and placed on a conventional ventilator with an FiO2 of 0.45 to achieve an oxygen saturation of 92%. Complete blood count reveals a platelet count of 120 K/μL. A coagulation panel reveals an INR of 1.7, a D-Dimer of 4.4 mg/L, and a fibrinogen of 120 mg/dL. Serum lactate is 2.9 mmol/L.
    """
    pid = 1
    result = processor.process_sepsis_case(case_summary, pid)

    if result:
        print(json.dumps(result, indent=2))
        sepsis_df = processor.create_dataframe_from_result(result)
        phoenix_scores = processor.calculate_phoenix_scores(sepsis_df)
        phoenix_dict_list = phoenix_scores.to_dict(orient='records')
        phoenix_dict = phoenix_dict_list[0] if phoenix_dict_list else {}
        processor.print_json_output(phoenix_dict)
        
        # For Phoenix 8
        phoenix_8_scores = processor.calculate_phoenix_8_scores(sepsis_df)
        phoenix_8_dict_list = phoenix_8_scores.to_dict(orient='records')
        phoenix_8_dict = phoenix_8_dict_list[0] if phoenix_8_dict_list else {}
        processor.print_json_output(phoenix_8_dict)




