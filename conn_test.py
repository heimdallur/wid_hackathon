import json
from snowflake.snowpark.session import Session


# COMMENT: Import login parameters
conn_config = json.load(open('cfg/connection_params.json', 'r'))

def create_session():  # COMMENT: Create snowpark session
    params = {
        "account": conn_config['WiD']['account'],
        "user": conn_config['WiD']['user'],
        "password": conn_config['WiD']['password'],
        "role": "WID_HACKER",
        "warehouse": conn_config['WiD']['warehouse'],
        # "database": conn_config['WiD']['database'],
        # "schema": conn_config['WiD']['schema']
        }

    # COMMENT: Return session object
    return Session.builder.configs(params).create()

# COMMENT: Create session
session = create_session()

# COMMENT: Test connection
res = session.sql("SHOW TABLES").collect()

for table in res:
    print(table.name)


