# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /project/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /project/build

# Include any dependencies generated for this target.
include testpkg/CMakeFiles/talkernode.dir/depend.make

# Include the progress variables for this target.
include testpkg/CMakeFiles/talkernode.dir/progress.make

# Include the compile flags for this target's objects.
include testpkg/CMakeFiles/talkernode.dir/flags.make

testpkg/CMakeFiles/talkernode.dir/src/talker.cpp.o: testpkg/CMakeFiles/talkernode.dir/flags.make
testpkg/CMakeFiles/talkernode.dir/src/talker.cpp.o: /project/src/testpkg/src/talker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object testpkg/CMakeFiles/talkernode.dir/src/talker.cpp.o"
	cd /project/build/testpkg && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/talkernode.dir/src/talker.cpp.o -c /project/src/testpkg/src/talker.cpp

testpkg/CMakeFiles/talkernode.dir/src/talker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/talkernode.dir/src/talker.cpp.i"
	cd /project/build/testpkg && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /project/src/testpkg/src/talker.cpp > CMakeFiles/talkernode.dir/src/talker.cpp.i

testpkg/CMakeFiles/talkernode.dir/src/talker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/talkernode.dir/src/talker.cpp.s"
	cd /project/build/testpkg && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /project/src/testpkg/src/talker.cpp -o CMakeFiles/talkernode.dir/src/talker.cpp.s

# Object files for target talkernode
talkernode_OBJECTS = \
"CMakeFiles/talkernode.dir/src/talker.cpp.o"

# External object files for target talkernode
talkernode_EXTERNAL_OBJECTS =

/project/devel/lib/testpkg/talkernode: testpkg/CMakeFiles/talkernode.dir/src/talker.cpp.o
/project/devel/lib/testpkg/talkernode: testpkg/CMakeFiles/talkernode.dir/build.make
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/libroscpp.so
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libpthread.so
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/librosconsole.so
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/librosconsole_log4cxx.so
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/librosconsole_backend_interface.so
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/libxmlrpcpp.so
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/libroscpp_serialization.so
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/librostime.so
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
/project/devel/lib/testpkg/talkernode: /opt/ros/noetic/lib/libcpp_common.so
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
/project/devel/lib/testpkg/talkernode: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/project/devel/lib/testpkg/talkernode: testpkg/CMakeFiles/talkernode.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /project/devel/lib/testpkg/talkernode"
	cd /project/build/testpkg && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/talkernode.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
testpkg/CMakeFiles/talkernode.dir/build: /project/devel/lib/testpkg/talkernode

.PHONY : testpkg/CMakeFiles/talkernode.dir/build

testpkg/CMakeFiles/talkernode.dir/clean:
	cd /project/build/testpkg && $(CMAKE_COMMAND) -P CMakeFiles/talkernode.dir/cmake_clean.cmake
.PHONY : testpkg/CMakeFiles/talkernode.dir/clean

testpkg/CMakeFiles/talkernode.dir/depend:
	cd /project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /project/src /project/src/testpkg /project/build /project/build/testpkg /project/build/testpkg/CMakeFiles/talkernode.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : testpkg/CMakeFiles/talkernode.dir/depend
