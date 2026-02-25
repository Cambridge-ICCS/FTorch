# Generate TorchScript models for testing
set(TEST_MODEL_SCRIPTS
    "create_simplenet_test_model.py"
    "create_batchingnet_test_model.py"
    "create_multiio_test_model.py"
)

foreach(model ${TEST_MODEL_SCRIPTS})
  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${model}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
endforeach()
