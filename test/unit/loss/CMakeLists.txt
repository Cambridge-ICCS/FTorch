add_pfunit_ctest(unittest_loss_mse
                  TEST_SOURCES unittest_loss_mse.pf
                  LINK_LIBRARIES FTorch::ftorch)
add_pfunit_ctest(unittest_loss_cross_entropy
                  TEST_SOURCES unittest_loss_cross_entropy.pf
                  LINK_LIBRARIES FTorch::ftorch)
