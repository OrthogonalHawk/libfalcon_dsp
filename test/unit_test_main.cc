#include <gtest/gtest.h>

extern void initialize_cuda_runtime(void);

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    initialize_cuda_runtime();
    
    /* run the tests */
    int res = RUN_ALL_TESTS();
    
    return res;   
}