import pandas as pd 

def preprocess_data(df:pd.DataFrame, target_col:str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling

    """
     # tidy headers
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
    #if id is present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # target col conversion (1/0 if it is Yes or No)
    if target_col in df.columns and df[target_col].dtype == "object" :
        df[target_col] = df[target_col].str.strip().map({
            "No": 0,
            "Yes":1
        })

    # casting for "Total Charges"
    if "Total Charges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

    # Fill NA for Senior Citizen
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)


    ## simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df




