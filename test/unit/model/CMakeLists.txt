add_pfunit_ctest(unittest_model_constructors_destructors
                 TEST_SOURCES unittest_model_constructors_destructors.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_model_interrogation
                 TEST_SOURCES unittest_model_interrogation.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_model_forward
                 TEST_SOURCES unittest_model_forward.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_model_batching
                 TEST_SOURCES unittest_model_batching.pf
                 LINK_LIBRARIES FTorch::ftorch)
