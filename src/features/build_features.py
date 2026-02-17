import pandas as pd

def _map_binary_series(s:pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.

    """
    values = list(pd.Series(s.dropna().unique()).astype(str))
    values = set(values)

    # Yes/No mapping (most common pattern in telecom data)
    if values == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
    
    # Gender mapping (For Demographic Features)
    if values == {"Male","Female"}:
        return s.map({"Female":0, "Male":1}).astype("Int64")
    
    # === GENERIC BINARY MAPPING ===
    # For any other 2-category feature, use stable alphabetical ordering

    if len(values) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(values)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")
    

    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s

def build_features(df:pd.DataFrame, target_col:str = "Churn")  -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.

    """

    df = df.copy()

    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify feature types ===
    # Find categorical columns excluding target col
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    num_cols = [c for c in df.select_dtypes(include=["int64","float64"]).columns]

    print(f"📊 Found {len(cat_cols)} categorical and {len(num_cols)} numeric columns")

    # === STEP 2: Split Categorical by Cardinality ===
    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 unique values) get one-hot encoding
    binary_cols = [c for c in cat_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in cat_cols if df[c].dropna().nunique() > 2]

    print(f"🔢 Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")

    if binary_cols:
        print(f"Binary: {binary_cols}")
    if multi_cols:
        print(f"Multi-category: {multi_cols}")

    # === STEP 3: Apply Binary Encoding ===
    # Convert 2-category features to 0/1 using deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"✅ {c}: {original_dtype} → binary (0/1)")

    # === STEP 4: Convert Boolean Columns ===
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # === STEP 5: One-Hot Encoding for Multi-Category Features ===
    # CRITICAL: drop_first=True prevents multicollinearity
    if multi_cols:
        print(f"🌟 Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        # Apply one-hot encoding with drop_first=True (same as serving)
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"✅ Created {new_features} new features from {len(multi_cols)} categorical columns")

    # === STEP 6: Data Type Cleanup ===
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    
    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df


    
    