add_executable(MST_test)
target_sources(MST_test PRIVATE MST.cpp)
target_link_libraries(MST_test PRIVATE graphy)

add_executable(MST_DS_test)
target_sources(MST_DS_test PRIVATE MST_DS.cpp)
target_link_libraries(MST_DS_test PRIVATE graphy)
