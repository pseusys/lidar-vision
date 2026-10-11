add_library(Takheliped STATIC IMPORTED)
find_library(LIBRARY_PATH Takheliped HINTS ${CMAKE_CURRENT_LIST_DIR}/../lib)
set_target_properties(Takheliped PROPERTIES IMPORTED_LOCATION ${LIBRARY_PATH})
