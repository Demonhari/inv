import great_expectations as ge
from great_expectations import get_context
import pandas as pd

# Step 1: Load sample transaction data
df = pd.read_json('tmp_transactions/data.json', lines=True)

# Step 2: Save dataframe to a temp CSV file
df.to_csv('tmp_transactions/sample_transactions.csv', index=False)

# Step 3: Create Great Expectations Context
context = get_context()

# Step 4: Add a datasource dynamically
datasource_name = "transactions_datasource"

context.add_datasource(
    name=datasource_name,
    class_name="Datasource",
    execution_engine={"class_name": "PandasExecutionEngine"},
    data_connectors={
        "default_runtime_data_connector_name": {
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": ["default_identifier_name"],
        }
    },
)

# Step 5: Create a batch request
batch_request = {
    "datasource_name": datasource_name,
    "data_connector_name": "default_runtime_data_connector_name",
    "data_asset_name": "transactions",
    "runtime_parameters": {"path": "tmp_transactions/sample_transactions.csv"},
    "batch_identifiers": {"default_identifier_name": "default_identifier"},
}

# Step 6: Create Validator
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="transaction_data_suite",
    create_expectation_suite_with_name_if_missing=True,
)

# Step 7: Add Expectations
validator.expect_column_to_exist("transaction_id")
validator.expect_column_values_to_not_be_null("transaction_id")
validator.expect_column_values_to_be_between("amount", min_value=0, max_value=10000)
validator.expect_column_values_to_be_in_set("fraud", [0, 1])

# Step 8: Save the Expectation Suite
validator.save_expectation_suite(discard_failed_expectations=False)
