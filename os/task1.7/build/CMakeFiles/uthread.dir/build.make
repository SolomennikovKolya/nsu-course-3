# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/uikvel/github/nsu-course-3/os/task1.7

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/uikvel/github/nsu-course-3/os/task1.7/build

# Include any dependencies generated for this target.
include CMakeFiles/uthread.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/uthread.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/uthread.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/uthread.dir/flags.make

CMakeFiles/uthread.dir/uthread/uthread.c.o: CMakeFiles/uthread.dir/flags.make
CMakeFiles/uthread.dir/uthread/uthread.c.o: ../uthread/uthread.c
CMakeFiles/uthread.dir/uthread/uthread.c.o: CMakeFiles/uthread.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/uikvel/github/nsu-course-3/os/task1.7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/uthread.dir/uthread/uthread.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/uthread.dir/uthread/uthread.c.o -MF CMakeFiles/uthread.dir/uthread/uthread.c.o.d -o CMakeFiles/uthread.dir/uthread/uthread.c.o -c /home/uikvel/github/nsu-course-3/os/task1.7/uthread/uthread.c

CMakeFiles/uthread.dir/uthread/uthread.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/uthread.dir/uthread/uthread.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/uikvel/github/nsu-course-3/os/task1.7/uthread/uthread.c > CMakeFiles/uthread.dir/uthread/uthread.c.i

CMakeFiles/uthread.dir/uthread/uthread.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/uthread.dir/uthread/uthread.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/uikvel/github/nsu-course-3/os/task1.7/uthread/uthread.c -o CMakeFiles/uthread.dir/uthread/uthread.c.s

# Object files for target uthread
uthread_OBJECTS = \
"CMakeFiles/uthread.dir/uthread/uthread.c.o"

# External object files for target uthread
uthread_EXTERNAL_OBJECTS =

libuthread.so: CMakeFiles/uthread.dir/uthread/uthread.c.o
libuthread.so: CMakeFiles/uthread.dir/build.make
libuthread.so: CMakeFiles/uthread.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/uikvel/github/nsu-course-3/os/task1.7/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library libuthread.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/uthread.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/uthread.dir/build: libuthread.so
.PHONY : CMakeFiles/uthread.dir/build

CMakeFiles/uthread.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/uthread.dir/cmake_clean.cmake
.PHONY : CMakeFiles/uthread.dir/clean

CMakeFiles/uthread.dir/depend:
	cd /home/uikvel/github/nsu-course-3/os/task1.7/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/uikvel/github/nsu-course-3/os/task1.7 /home/uikvel/github/nsu-course-3/os/task1.7 /home/uikvel/github/nsu-course-3/os/task1.7/build /home/uikvel/github/nsu-course-3/os/task1.7/build /home/uikvel/github/nsu-course-3/os/task1.7/build/CMakeFiles/uthread.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/uthread.dir/depend
