from conn_test import create_session
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


session = create_session()

# Read raw survey data
tb_name = ("WID_HACKATHON_PRIVATE_DATASETS." +
           "SURVEY_FEATURES." +
           "RAW_SAFETY_SURVEY_DATA")

df = pd.DataFrame(session.sql(
    f"SELECT * FROM {tb_name}").collect())

df.columns = [c.lower() for c in df.columns]

# Inspect
df.info()

col_cats = set([c.lower().split("_")[0] + "_*"
                for c in df.columns])

for cc in col_cats:
    print(cc)


# Features
col_feat_match = [c for c in df.columns
                  if c.startswith(("demog_", "optional"))
                  and "optionalreporting" not in c
                  and "_other" not in c
                  and c != "optional"
                  and "demog_age" not in c]

df_feat = df[col_feat_match]

# Feature encoding
df_feat_float = df_feat.select_dtypes("float")
df_feat_ohe = pd.get_dummies(df_feat.select_dtypes("object"))
df_feat_proc = pd.concat([df[["id"]],
                          df_feat_float,
                          df_feat_ohe], axis=1)
df_feat_proc.columns = ["".join(c for c in col if c.isalnum())
                        for col in df_feat_proc.columns]


# Ordinal variables
enc = OrdinalEncoder(encoded_missing_value=np.nan)
df["demog_age_oe"] = enc.fit_transform(df[["demog_age"]].fillna("-1"))
# enc.categories_
df["demog_age_oe"].replace(0, np.nan, inplace=True)
df_feat_ord = df[["demog_age_oe"]]


# Target
col_tgt_match = [c for c in df.columns
                 if c.startswith(("experienced_"))
                 and "public" in c and "none" not in c]
df["target"] = df[col_tgt_match].notna().max(axis=1)
df["target"] = df["target"].astype(int)
df_target = df[["target"]]

# Sample weights
df["weight"] = df["respondent_weight"]
df_weight = df[["weight"]]

# Combine
df_model = pd.concat([df_feat_ord,
                      df_feat_proc,
                      df_target,
                      df_weight],
                     axis=1)

# Save feature set
df_model.to_csv("data/demog_optional_experienced_public_wo.csv",
                index=False)
