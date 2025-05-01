import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

def load_and_combine(train_path: str, test_path: str):
    """Load train and test datasets, set index to Id, and combine into one DataFrame."""
    train_df = pd.read_csv(train_path, index_col="Id")
    test_df = pd.read_csv(test_path, index_col="Id")

    y_train = train_df["SalePrice"].copy()
    train_df = train_df.drop("SalePrice", axis=1)

    train_df = remove_outliers(train_df, y_train)

    train_df["Dataset"] = "train"
    test_df["Dataset"] = "test"

    data = pd.concat([train_df, test_df], axis=0)
    return data, y_train

def remove_outliers(train_df: pd.DataFrame, y_train: pd.Series):
    """Remove known outliers from training data (e.g., huge GrLivArea with low SalePrice)."""
    train_df_temp = train_df.copy()
    train_df_temp["SalePrice"] = y_train 

    outlier_condition = (train_df_temp["GrLivArea"] > 4500)
    outliers = train_df_temp.index[outlier_condition]
    if len(outliers) > 0:
        train_df_temp = train_df_temp.drop(outliers)

    y_train.drop(outliers, inplace=True)
    train_df_temp = train_df_temp.drop("SalePrice", axis=1)
    return train_df_temp

def fill_missing_values(data: pd.DataFrame):
    """Impute or fill missing values in the combined dataset."""
    ## Based on feature type and domain knowledge
    none_cols = ["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", 
                 "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                 "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", 
                 "MasVnrType"]
    for col in none_cols:
        if col in data.columns:
            data[col] = data[col].fillna("None")

    zero_cols = ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", 
                 "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", 
                 "MasVnrArea", "PoolArea"]
    for col in zero_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    if "LotFrontage" in data.columns:
        data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median()))

    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode()[0])
    return data

def transform_skewed_features(data: pd.DataFrame, skew_thresh=0.75):
    """Apply log1p (or Yeo-Johnson) transformation to numeric features with skewness above threshold."""
    numeric_cols = data.select_dtypes(include=["int64","float64"]).columns

    skew_vals = data[numeric_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    high_skew = skew_vals[abs(skew_vals) > skew_thresh]
    for col in high_skew.index:
        if (data[col] <= 0).any():
            pt = PowerTransformer(method='yeo-johnson')
            data[col] = pt.fit_transform(data[[col]])
        else:
            data[col] = np.log1p(data[col])
    return data

def engineer_features(data: pd.DataFrame):
    """Create new features (total bathrooms, total area, porch area, luxury flag, cyclical month, etc.) and drop redundant ones."""

    if {"FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"}.issubset(data.columns):
        data["TotalBath"] = data[["FullBath", "BsmtFullBath", "HalfBath", "BsmtHalfBath"]].fillna(0)\
                                .dot([1, 1, 0.5, 0.5])

    if {"1stFlrSF","2ndFlrSF","TotalBsmtSF"}.issubset(data.columns):
        data["TotalSF"] = data[["1stFlrSF", "2ndFlrSF", "TotalBsmtSF"]].fillna(0).sum(axis=1)

    porch_feats = ["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]
    common_porch = [col for col in porch_feats if col in data.columns]
    if common_porch:
        data["TotalPorch"] = data[common_porch].fillna(0).sum(axis=1)

    lux_feats = []
    for col in ["PoolQC", "MiscFeature", "Fence", "Alley"]:
        if col in data.columns:
            data[f"Has{col}"] = np.where(data[col] != "None", 1, 0)
            lux_feats.append(f"Has{col}")
    if "Fireplaces" in data.columns:
        lux_feats.append("Fireplaces")
    if lux_feats:
        data["LuxuryFeature"] = data[lux_feats].sum(axis=1)

    if "MoSold" in data.columns:
        data["MoSold_sin"] = np.sin(2 * np.pi * data["MoSold"] / 12.0)
        data["MoSold_cos"] = np.cos(2 * np.pi * data["MoSold"] / 12.0)

    if "MSSubClass" in data.columns:
        data["MSSubClass"] = data["MSSubClass"].astype(str)

    drop_cols = ["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath",
                 "1stFlrSF","2ndFlrSF","TotalBsmtSF",
                 "OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch",
                 "PoolQC","MiscFeature","Fence","Alley","MiscVal","Fireplaces","FireplaceQu"]
    for col in drop_cols:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    return data

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Scale continuous numerical features (StandardScaler) for models that benefit from normalization."""
    scaler = StandardScaler()
    binary_cols = [col for col in X_train.columns 
                   if X_train[col].nunique() == 2 and sorted(X_train[col].unique()) == [0,1]]
    numeric_cols = X_train.select_dtypes(include=["int64","float64"]).columns.tolist()

    numeric_to_scale = [col for col in numeric_cols if col not in binary_cols]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_to_scale] = scaler.fit_transform(X_train[numeric_to_scale])
    X_test_scaled[numeric_to_scale] = scaler.transform(X_test[numeric_to_scale])
    return X_train_scaled, X_test_scaled

def preprocess_data(train_path: str, test_path: str, log_transform_target=True):
    """
    End-to-end data preprocessing: returns X_train, X_test, y_train (and scaled versions).
    """
    data, y_train = load_and_combine(train_path, test_path)
    data = fill_missing_values(data)
    data = engineer_features(data)
    data = transform_skewed_features(data)
    data = pd.get_dummies(data, drop_first=True) 

    X_train = data[data["Dataset_train"] == 1].drop(["Dataset_train"], axis=1)
    X_test = data[data["Dataset_train"] != 1].drop(["Dataset_train"], axis=1)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train_index = X_train.index
    y_train = y_train.loc[X_train_index]

    if log_transform_target:
        y_train = np.log1p(y_train)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    return X_train, X_test, y_train, X_train_scaled, X_test_scaled

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Preprocess House Prices data")
    parser.add_argument("--train", default="train.csv", help="Path to training CSV file")
    parser.add_argument("--test", default="test.csv", help="Path to test CSV file")
    parser.add_argument("--out_dir", default=".", help="Directory to save processed output files")
    args = parser.parse_args()
    X_train, X_test, y_train, X_train_scaled, X_test_scaled = preprocess_data(args.train, args.test)

    X_train.to_csv(os.path.join(args.out_dir, "X_train_processed.csv"), index=True)
    X_test.to_csv(os.path.join(args.out_dir, "X_test_processed.csv"), index=True)
    y_train.to_csv(os.path.join(args.out_dir, "y_train_processed.csv"), index=True)
