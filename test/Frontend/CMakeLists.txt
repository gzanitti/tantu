find_package(GTest REQUIRED)

add_executable(LexerTest LexerTest.cpp)
target_link_libraries(LexerTest
    PRIVATE
    TantuFrontend
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME LexerTest COMMAND LexerTest)

add_executable(ParserTest ParserTest.cpp)
target_link_libraries(ParserTest
    PRIVATE
    TantuFrontend
    GTest::gtest
    GTest::gtest_main
)
add_test(NAME ParserTest COMMAND ParserTest)