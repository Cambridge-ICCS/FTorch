# Test utils operators on CPU
add_pfunit_ctest(unittest_test_utils_operators
                 TEST_SOURCES unittest_test_utils_operators.pf
                 LINK_LIBRARIES FTorch::ftorch)
