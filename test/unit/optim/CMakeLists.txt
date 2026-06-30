add_pfunit_ctest(unittest_optim
                  TEST_SOURCES unittest_optim.pf
                  LINK_LIBRARIES FTorch::ftorch)

add_pfunit_ctest(unittest_optim_sgd
                  TEST_SOURCES unittest_optim_sgd.pf
                  LINK_LIBRARIES FTorch::ftorch)

add_pfunit_ctest(unittest_optim_adam
                  TEST_SOURCES unittest_optim_adam.pf
                  LINK_LIBRARIES FTorch::ftorch)

add_pfunit_ctest(unittest_optim_adamw
                  TEST_SOURCES unittest_optim_adamw.pf
                  LINK_LIBRARIES FTorch::ftorch)
