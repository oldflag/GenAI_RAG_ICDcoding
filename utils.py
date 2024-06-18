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

if __name__ == "__main__":
    icdfile_path = './icd10cm_order_2024_top100.txt'  # only first 100 rows for testing
    icd10_df = parse_icd10_description(icdfile_path)
    print(icd10_df.head())

