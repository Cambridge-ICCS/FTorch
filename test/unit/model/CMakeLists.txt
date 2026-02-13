add_pfunit_ctest(unittest_constructors_destructors
                 TEST_SOURCES unittest_constructors_destructors.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_interrogation
                 TEST_SOURCES unittest_interrogation.pf
                 LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_forward
                 TEST_SOURCES unittest_forward.pf
                 LINK_LIBRARIES FTorch::ftorch)
