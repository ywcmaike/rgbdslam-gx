# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/ywc/work/cplusplus/slam

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ywc/work/cplusplus/slam/build

# Include any dependencies generated for this target.
include src/CMakeFiles/generatePointCloud.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/generatePointCloud.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/generatePointCloud.dir/flags.make

src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o: src/CMakeFiles/generatePointCloud.dir/flags.make
src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o: ../src/generatePointCloud.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ywc/work/cplusplus/slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o"
	cd /home/ywc/work/cplusplus/slam/build/src && g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o -c /home/ywc/work/cplusplus/slam/src/generatePointCloud.cpp

src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.i"
	cd /home/ywc/work/cplusplus/slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ywc/work/cplusplus/slam/src/generatePointCloud.cpp > CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.i

src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.s"
	cd /home/ywc/work/cplusplus/slam/build/src && g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ywc/work/cplusplus/slam/src/generatePointCloud.cpp -o CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.s

src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.requires:

.PHONY : src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.requires

src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.provides: src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/generatePointCloud.dir/build.make src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.provides.build
.PHONY : src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.provides

src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.provides.build: src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o


# Object files for target generatePointCloud
generatePointCloud_OBJECTS = \
"CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o"

# External object files for target generatePointCloud
generatePointCloud_EXTERNAL_OBJECTS =

../bin/generatePointCloud: src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o
../bin/generatePointCloud: src/CMakeFiles/generatePointCloud.dir/build.make
../bin/generatePointCloud: /usr/local/lib/libopencv_videostab.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_ts.a
../bin/generatePointCloud: /usr/local/lib/libopencv_superres.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_stitching.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_photo.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_legacy.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_contrib.so.2.4.13
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/generatePointCloud: /usr/local/lib/libpcl_common.so
../bin/generatePointCloud: /usr/local/lib/libpcl_octree.so
../bin/generatePointCloud: /usr/lib/libOpenNI.so
../bin/generatePointCloud: /usr/local/lib/libpcl_io.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/generatePointCloud: /usr/local/lib/libpcl_kdtree.so
../bin/generatePointCloud: /usr/local/lib/libpcl_search.so
../bin/generatePointCloud: /usr/local/lib/libpcl_visualization.so
../bin/generatePointCloud: /usr/local/lib/libpcl_sample_consensus.so
../bin/generatePointCloud: /usr/local/lib/libpcl_filters.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/generatePointCloud: /usr/lib/libOpenNI.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/generatePointCloud: /usr/local/lib/libvtkIOAMR-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersAMR-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkInteractionImage-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersTexture-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOSQL-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtksqlite-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOTecplotTable-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersFlowPaths-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersHyperTree-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingStencil-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersSMP-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkDomainsChemistryOpenGL2-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkDomainsChemistry-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOInfovis-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtklibxml2-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOMINC-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOImport-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOExport-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingGL2PSOpenGL2-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkgl2ps-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersProgrammable-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingVolumeOpenGL2-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingContextOpenGL2-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOExodus-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOParallel-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOLSDyna-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOMovie-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkoggtheora-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkViewsInfovis-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOEnSight-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkGeovisCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingStatistics-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersGeneric-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOPLY-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingLOD-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersVerdict-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOVideo-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOParallelXML-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingImage-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkViewsContext2D-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersParallelImaging-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersPoints-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersSelection-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingMorphological-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libpcl_common.so
../bin/generatePointCloud: /usr/local/lib/libpcl_octree.so
../bin/generatePointCloud: /usr/local/lib/libpcl_io.so
../bin/generatePointCloud: /usr/local/lib/libpcl_kdtree.so
../bin/generatePointCloud: /usr/local/lib/libpcl_search.so
../bin/generatePointCloud: /usr/local/lib/libpcl_visualization.so
../bin/generatePointCloud: /usr/local/lib/libpcl_sample_consensus.so
../bin/generatePointCloud: /usr/local/lib/libpcl_filters.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/generatePointCloud: /usr/local/lib/libopencv_nonfree.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_ocl.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_video.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_objdetect.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_ml.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_calib3d.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_features2d.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_highgui.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_imgproc.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_flann.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libopencv_core.so.2.4.13
../bin/generatePointCloud: /usr/local/lib/libtbb.so
../bin/generatePointCloud: /usr/local/lib/libvtkImagingMath-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingOpenGL2-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkglew-7.1.so.1
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libXt.so
../bin/generatePointCloud: /usr/local/lib/libvtkIOGeometry-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkexoIIc-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkjsoncpp-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIONetCDF-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkNetCDF_cxx-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkNetCDF-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkhdf5_hl-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkhdf5-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkChartsCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingLabel-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkInfovisLayout-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkInfovisCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkproj4-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkverdict-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingContext2D-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkViewsCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkInteractionWidgets-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersHybrid-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkInteractionStyle-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingHybrid-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOImage-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkDICOMParser-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkmetaio-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkpng-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtktiff-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkjpeg-7.1.so.1
../bin/generatePointCloud: /usr/lib/x86_64-linux-gnu/libm.so
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingAnnotation-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingColor-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingFreeType-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkfreetype-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingVolume-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOXML-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOXMLParser-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkexpat-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersImaging-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersParallel-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkParallelCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOLegacy-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkIOCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkzlib-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkRenderingCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonColor-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersGeometry-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersExtraction-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersStatistics-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingFourier-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkalglib-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersModeling-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersSources-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersGeneral-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonComputationalGeometry-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkFiltersCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingGeneral-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingSources-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkImagingCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonExecutionModel-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonDataModel-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonTransforms-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonMisc-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonMath-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonSystem-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtkCommonCore-7.1.so.1
../bin/generatePointCloud: /usr/local/lib/libvtksys-7.1.so.1
../bin/generatePointCloud: src/CMakeFiles/generatePointCloud.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ywc/work/cplusplus/slam/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/generatePointCloud"
	cd /home/ywc/work/cplusplus/slam/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/generatePointCloud.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/generatePointCloud.dir/build: ../bin/generatePointCloud

.PHONY : src/CMakeFiles/generatePointCloud.dir/build

src/CMakeFiles/generatePointCloud.dir/requires: src/CMakeFiles/generatePointCloud.dir/generatePointCloud.cpp.o.requires

.PHONY : src/CMakeFiles/generatePointCloud.dir/requires

src/CMakeFiles/generatePointCloud.dir/clean:
	cd /home/ywc/work/cplusplus/slam/build/src && $(CMAKE_COMMAND) -P CMakeFiles/generatePointCloud.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/generatePointCloud.dir/clean

src/CMakeFiles/generatePointCloud.dir/depend:
	cd /home/ywc/work/cplusplus/slam/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ywc/work/cplusplus/slam /home/ywc/work/cplusplus/slam/src /home/ywc/work/cplusplus/slam/build /home/ywc/work/cplusplus/slam/build/src /home/ywc/work/cplusplus/slam/build/src/CMakeFiles/generatePointCloud.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/generatePointCloud.dir/depend

