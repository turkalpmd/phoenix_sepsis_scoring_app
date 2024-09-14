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
        Create a data entry for a sepsis patient using the following case summary:
        {case_summary}
        Construct a Python dictionary with these keys and notes:
        - pid: patient identification number
        - age: age in months
        - fio2: fraction of inspired oxygen
        - pao2: partial pressure of oxygen in arterial blood (mmHg)
        - spo2: pulse oximetry (only valid when SpO2 ≤ 97)
        - vent: indicator for invasive mechanical ventilation (0 for no, 1 for yes)
        - gcs_total: total Glasgow Coma Scale (integer from 3 to 15)
        - pupil: character vector reporting if pupils are reactive or fixed
        - platelets: platelets measured in 1,000/microliter
        - inr: international normalized ratio
        - d_dimer: D-dimer; units of mg/L FEU
        - fibrinogen: units of mg/dL
        - dbp: diagnostic blood pressure (mmHg)
        - sbp: systolic blood pressure (mmHg)
        - lactate: units of mmol/L (arterial or venous)
        - dobutamine: indicator for receiving systemic dobutamine (0 for no, 1 for yes)
        - dopamine: indicator for receiving systemic dopamine (0 for no, 1 for yes)
        - epinephrine: indicator for receiving systemic epinephrine (0 for no, 1 for yes)
        - milrinone: indicator for receiving systemic milrinone (0 for no, 1 for yes)
        - norepinephrine: indicator for receiving systemic norepinephrine (0 for no, 1 for yes)
        - vasopressin: indicator for receiving systemic vasopressin (0 for no, 1 for yes)
        - glucose: units of mg/dL
        - anc: units of 1,000 cells per cubic millimeter
        - alc: units of 1,000 cells per cubic millimeter
        - creatinine: units of mg/dL
        - bilirubin: units of mg/dL
        - alt: units of IU/L
        Rules:
        1. Use {pid} for the pid value.
        2. Calculate age in months.
        3. Use None for values not mentioned in the case summary.
        4. Use 0 (no) or 1 (yes) for ventilator and drug applications.
        5. Use appropriate data types for numerical values (int or float).
        6. Perform correct unit conversions.
        7. Derive Missing Values Where Applicable: If certain values are missing but can be calculated or estimated from other available data in the case summary, compute these values. 
        Return only a valid Python dictionary, without additional explanations.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical data analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            response_text = response.choices[0].message.content.strip()
            # print("Raw response text:", response_text)  # Debug output
            cleaned_response_text = re.sub(r'```python', '', response_text)
            cleaned_response_text = re.sub(r'```', '', cleaned_response_text)
            cleaned_response_text = re.sub(r'#.*', '', cleaned_response_text)
            cleaned_response_text = cleaned_response_text.replace("'", '"').replace("None", "null")
            # print("Cleaned response text:", cleaned_response_text)  # Debug output
            patient_data = json.loads(cleaned_response_text)
            print(f"Total token usage for this run: {response.usage.total_tokens}")
            return patient_data

        except json.JSONDecodeError as e:
            print("Failed to decode JSON:")
            print(e)
            print("Faulty JSON:", cleaned_response_text)  # Show the faulty JSON
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def create_dataframe_from_result(self, result):
        df = pd.DataFrame([result])
        df.replace(to_replace=[None], value=np.nan, inplace=True)
        return df

    def calculate_phoenix_scores(self, sepsis_df):
    # Replace None with default values for numeric calculations
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
            fixed_pupils=(sepsis_df["pupil"] == "Fixed").astype(int)
        )
    
    def calculate_phoenix_8_scores(self,sepsis_df):
        
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
                                fixed_pupils = (sepsis_df["pupil"] == "both-fixed").astype(int),
                                glucose = sepsis_df["glucose"],
                                anc = sepsis_df["anc"],
                                alc = sepsis_df["alc"],
                                creatinine = sepsis_df["creatinine"],
                                bilirubin = sepsis_df["bilirubin"],
                                alt = sepsis_df["alt"],
                                age = sepsis_df["age"])
        

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
#For example, if pao2 is missing but fio2 and spo2 are provided, estimate pao2 using the Severinghaus equation: PaO2 = FiO2 * (713 - 47) - (PaCO2 / 0.8). Assume PaCO2 = 40 mmHg if not provided. Document any assumptions made in such derivations.
#8. In the absence of documented neurological examination results, it is permissible to default the 'gcs_total' to 15, assuming normal neurological function.