find_package(GTest REQUIRED)

add_executable(TypeCheckerTest TypeCheckerTest.cpp)
target_link_libraries(TypeCheckerTest
    PRIVATE
    TantuSema
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME TypeCheckerTest COMMAND TypeCheckerTest)