# Generate TorchScript models for testing
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/create_simplenet_test_model.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
