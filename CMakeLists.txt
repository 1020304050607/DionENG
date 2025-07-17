cmake_minimum_required(VERSION 3.15)
project(DionENG LANGUAGES NONE)

# Find Python 3.8 or higher
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python 3.8 or higher is required")
endif()

# Set build output directory
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/builds)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Define source directories
set(DIONENG_DIR ${CMAKE_SOURCE_DIR}/dioneng)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)
set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/assets)
set(MAPS_DIR ${CMAKE_SOURCE_DIR}/maps)
set(MODS_DIR ${CMAKE_SOURCE_DIR}/mods)

# Define main entry points
set(MAIN_PY ${CMAKE_SOURCE_DIR}/main.py)
set(DEMO_SCENE_PY ${EXAMPLES_DIR}/demo_scene_1/main.py)
set(DEMO_SCENE_JSON ${EXAMPLES_DIR}/demo_scene_1/assets/scene.json)

# Install dependencies from requirements.txt
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip install -r ${CMAKE_SOURCE_DIR}/requirements.txt
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE PIP_RESULT
)
if(NOT PIP_RESULT EQUAL 0)
    message(WARNING "Failed to install dependencies from requirements.txt")
endif()

# Copy Python source files to build directory
file(GLOB_RECURSIVE DIONENG_FILES "${DIONENG_DIR}/*.py")
file(COPY ${DIONENG_FILES} DESTINATION ${CMAKE_BINARY_DIR}/dioneng)
file(COPY ${MAIN_PY} DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${DEMO_SCENE_PY} DESTINATION ${CMAKE_BINARY_DIR}/examples/demo_scene_1)
file(COPY ${DEMO_SCENE_JSON} DESTINATION ${CMAKE_BINARY_DIR}/examples/demo_scene_1/assets)

# Copy assets to build directory
file(GLOB ASSET_FILES
    "${ASSETS_DIR}/*.png"
    "${ASSETS_DIR}/*.obj"
    "${ASSETS_DIR}/*.wav"
    "${ASSETS_DIR}/*.vert"
)
file(COPY ${ASSET_FILES} DESTINATION ${CMAKE_BINARY_DIR}/assets)

# Copy maps and mods directories
file(COPY ${MAPS_DIR} DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${MODS_DIR} DESTINATION ${CMAKE_BINARY_DIR})

# Add custom target to run main.py
add_custom_target(
    RunMain ALL
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_BINARY_DIR}/main.py --scene ${CMAKE_BINARY_DIR}/examples/demo_scene_1/assets/scene.json --debug
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running DionENG main.py"
    DEPENDS ${MAIN_PY} ${DIONENG_FILES} ${DEMO_SCENE_JSON} ${ASSET_FILES}
)

# Add custom target to run demo_scene_1/main.py
add_custom_target(
    RunDemoScene ALL
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_BINARY_DIR}/examples/demo_scene_1/main.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running DionENG demo_scene_1"
    DEPENDS ${DEMO_SCENE_PY} ${DIONENG_FILES} ${DEMO_SCENE_JSON} ${ASSET_FILES}
)

# Install project files to distribution directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/dioneng DESTINATION ${CMAKE_INSTALL_PREFIX}/DionENG)
install(FILES ${CMAKE_BINARY_DIR}/main.py DESTINATION ${CMAKE_INSTALL_PREFIX}/DionENG)
install(DIRECTORY ${CMAKE_BINARY_DIR}/examples DESTINATION ${CMAKE_INSTALL_PREFIX}/DionENG)
install(DIRECTORY ${CMAKE_BINARY_DIR}/assets DESTINATION ${CMAKE_INSTALL_PREFIX}/DionENG)
install(DIRECTORY ${CMAKE_BINARY_DIR}/maps DESTINATION ${CMAKE_INSTALL_PREFIX}/DionENG)
install(DIRECTORY ${CMAKE_BINARY_DIR}/mods DESTINATION ${CMAKE_INSTALL_PREFIX}/DionENG)

# Enable testing
enable_testing()
add_test(
    NAME TestMain
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_BINARY_DIR}/main.py --scene ${CMAKE_BINARY_DIR}/examples/demo_scene_1/assets/scene.json
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_test(
    NAME TestDemoScene
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_BINARY_DIR}/examples/demo_scene_1/main.py
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)