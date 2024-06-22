import pandas as pd

def parse_icd10_description(icdfile):
    """ 
    Parse ICD-10-CM file from "https://www.cms.gov/medicare/coding-billing/icd-10-codes/2024-icd-10-cm".
    """
    # Define the column specifications based on the PDF instructions
    col_specs = [
        (0, 5),    # Order number
        (6, 13),   # ICD-10-CM or ICD-10-PCS code
        (14, 15),  # HIPAA-covered transactions indicator
        (16, 76),  # Short description
        (77, None) # Long description
    ]

    # Column names for the parsed data
    col_names = ['OrderNumber', 'ICD-10Code', 'HIPAAIndicator', 'ShortDescription', 'LongDescription']

    # Read the fixed-width formatted text file
    df = pd.read_fwf(icdfile, colspecs=col_specs, names=col_names)

    return df

def filter_symptom_from_icd(symptomfile, icdfile):
    # Load the CSV file
    symptom2disease_df = pd.read_csv(symptomfile)

    # Load the ICD codes file
    icd_df = parse_icd10_description(icdfile)

    # Combine short and long descriptions into a single list
    icd_descriptions = icd_df['ShortDescription'].str.lower().tolist() + icd_df['LongDescription'].str.lower().tolist()

    # Filter rows where label is related to ICD descriptions
    def is_related(label, descriptions):
        for description in descriptions:
            if label.lower() in description :
                return True
        return False

    filtered_df = symptom2disease_df[symptom2disease_df['label'].apply(lambda x: is_related(x, icd_descriptions))]

    # Save the filtered rows to a new CSV file
    # filtered_df.to_csv(symptomfile+'.filtered.csv', index=False)
    return filtered_df

if __name__ == "__main__":
    icdfile_path = './icd10cm_order_2024_top100.txt'  # only first 100 rows for testing

    # icd10_df = parse_icd10_description(icdfile_path)
    # print(icd10_df.head())

    symptom_path = './Symptom2Disease.csv'  # only first 100 rows for testing
    filtered_symptom_df=filter_symptom_from_icd(symptom_path, icdfile_path)
    print(filtered_symptom_df.head())
    filtered_symptom_df.to_csv(symptom_path+'.filtered.csv', index=False)

