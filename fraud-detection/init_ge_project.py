import great_expectations as ge

context = ge.get_context()

context.create_expectation_suite(
    expectation_suite_name="dummy_suite",
    overwrite_existing=True
)

print("✅ Great Expectations project initialized successfully!")