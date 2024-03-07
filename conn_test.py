import json
from tabulate import tabulate
from pprint import pprint
from snowflake.snowpark.session import Session


# COMMENT: Import login parameters
conn_config = json.load(open('cfg/connection_params.json', 'r'))

def create_session():  # COMMENT: Create snowpark session
    params = {
        "account": conn_config['WiD']['account'],
        "user": conn_config['WiD']['user'],
        "password": conn_config['WiD']['password'],
        "role": "WID_HACKER",
        "warehouse": conn_config['WiD']['warehouse']
        }

    # COMMENT: Return session object
    return Session.builder.configs(params).create()

# COMMENT: Create session
session = create_session()

table = "WID_HACKATHON_PRIVATE_DATASETS.SURVEY_FEATURES.SAFETY_SURVEY_DATA"

# COMMENT: Test connection
res = session.sql(f"SELECT NONREPORTREASON_PUBLIC_OTHER FROM {table} LIMIT").collect()

print(tabulate(res))
