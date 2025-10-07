################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2014-2017, Advanced Micro Devices, Inc. All rights reserved. 
##
## Developed by:
##
##                 AMD Research and AMD HSA Software Development
##
##                 Advanced Micro Devices, Inc. All rights reserved.
##
##                 www.amd.com
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and#or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and#or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc. All rights reserved.
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

## Parses the VERSION_STRING variable and places
## the first, second and third number values in
## the major, minor and patch variables.
function( parse_version VERSION_STRING )

    string ( FIND ${VERSION_STRING} "-" STRING_INDEX )

    if ( ${STRING_INDEX} GREATER -1 )
        math ( EXPR STRING_INDEX "${STRING_INDEX} + 1" )
        string ( SUBSTRING ${VERSION_STRING} ${STRING_INDEX} -1 VERSION_BUILD )
    endif ()

    string ( REGEX MATCHALL "[0123456789]+" VERSIONS ${VERSION_STRING} )
    list ( LENGTH VERSIONS VERSION_COUNT )

    if ( ${VERSION_COUNT} GREATER 0)
        list ( GET VERSIONS 0 MAJOR )
        set ( VERSION_MAJOR ${MAJOR} PARENT_SCOPE )
        set ( TEMP_VERSION_STRING "${MAJOR}" )
    endif ()

    if ( ${VERSION_COUNT} GREATER 1 )
        list ( GET VERSIONS 1 MINOR )
        set ( VERSION_MINOR ${MINOR} PARENT_SCOPE )
        set ( TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${MINOR}" )
    endif ()

    if ( ${VERSION_COUNT} GREATER 2 )
        list ( GET VERSIONS 2 PATCH )
        set ( VERSION_PATCH ${PATCH} PARENT_SCOPE )
        set ( TEMP_VERSION_STRING "${TEMP_VERSION_STRING}.${PATCH}" )
    endif ()

    if ( DEFINED VERSION_BUILD )
        set ( VERSION_BUILD "${VERSION_BUILD}" PARENT_SCOPE )
    endif ()

    set ( VERSION_STRING "${TEMP_VERSION_STRING}" PARENT_SCOPE )

endfunction ()

## Gets the current version of the repository
## using versioning tags and git describe.
## Passes back a packaging version string
## and a library version string.
function ( get_version DEFAULT_VERSION_STRING )

    parse_version ( ${DEFAULT_VERSION_STRING} )

    find_program ( GIT NAMES git )

    if ( GIT )

        execute_process ( COMMAND git describe --abbrev=0 --tags
                          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                          OUTPUT_VARIABLE GIT_TAG_STRING
                          OUTPUT_STRIP_TRAILING_WHITESPACE
                          RESULT_VARIABLE RESULT )
		  # we should not mimic version of ROCM to RVS. RVS is independent entity
		  # tagging is usually done with respect to ROCM release, so removing it.
		  #if ( ${RESULT} EQUAL 0 )
		  #parse_version ( ${GIT_TAG_STRING} )

		  #endif ()

    endif ()

    set( VERSION_STRING "${VERSION_STRING}" PARENT_SCOPE )
    set( VERSION_MAJOR  "${VERSION_MAJOR}" PARENT_SCOPE )
    set( VERSION_MINOR  "${VERSION_MINOR}" PARENT_SCOPE )
    if (DEFINED ENV{ROCM_LIBPATCH_VERSION})
        set (VERSION_PATCH $ENV{ROCM_LIBPATCH_VERSION} PARENT_SCOPE )
    else ()
        set (VERSION_PATCH ${VERSION_PATCH} PARENT_SCOPE )
    endif()
    set( VERSION_BUILD  "${VERSION_BUILD}" PARENT_SCOPE )

endfunction()

## Configure Copyright File for Debian Package
function( configure_pkg PACKAGE_NAME_T COMPONENT_NAME_T PACKAGE_VERSION_T MAINTAINER_NM_T MAINTAINER_EMAIL_T)
    # Check If Debian Platform
    find_file (DEBIAN debian_version debconf.conf PATHS /etc)
    if(DEBIAN)
      set( BUILD_DEBIAN_PKGING_FLAG ON CACHE BOOL "Internal Status Flag to indicate Debian Packaging Build" FORCE )
      set_debian_pkg_cmake_flags( ${PACKAGE_NAME_T} ${PACKAGE_VERSION_T}
                                  ${MAINTAINER_NM_T} ${MAINTAINER_EMAIL_T} )

      # Create debian directory in build tree
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN")

      # Configure the copyright file
      configure_file(
        "${CMAKE_SOURCE_DIR}/DEBIAN/copyright.in"
        "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
        @ONLY
      )

      # Install copyright file
      install ( FILES "${CMAKE_BINARY_DIR}/DEBIAN/copyright"
	        DESTINATION "${CMAKE_INSTALL_DOCDIR}"
	        COMPONENT ${COMPONENT_NAME_T} )

      # Configure the changelog file
      configure_file(
        "${CMAKE_SOURCE_DIR}/DEBIAN/changelog.in"
        "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian"
        @ONLY
      )

      if( BUILD_ENABLE_LINTIAN_OVERRIDES )
	if(DEFINED BUILD_SHARED_LIBS AND NOT ${BUILD_SHARED_LIBS} STREQUAL "")
	  string(FIND ${DEB_OVERRIDES_INSTALL_FILENM} "static" OUT_VAR1)
	  if(OUT_VAR1 EQUAL -1)
	    set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_OVERRIDES_INSTALL_FILENM}-static" )
          endif()
	else()
          if(ENABLE_ASAN_PACKAGING)
	    string( FIND ${DEB_OVERRIDES_INSTALL_FILENM} "asan" OUT_VAR2)
	    if(OUT_VAR2 EQUAL -1)
	      set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_OVERRIDES_INSTALL_FILENM}-asan" )
	    endif()
          endif()
	endif()
	set( DEB_OVERRIDES_INSTALL_FILENM
		"${DEB_OVERRIDES_INSTALL_FILENM}" CACHE STRING "Debian Package Lintian Override File Name" FORCE)
        # Configure the changelog file
        configure_file(
          "${CMAKE_SOURCE_DIR}/DEBIAN/overrides.in"
          "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_OVERRIDES_INSTALL_FILENM}"
	   FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
          @ONLY
        )
      endif()

      # Install Change Log
      find_program ( DEB_GZIP_EXEC gzip )
      if(EXISTS "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian" )
        execute_process(
          COMMAND ${DEB_GZIP_EXEC} -f -n -9 "${CMAKE_BINARY_DIR}/DEBIAN/changelog.Debian"
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/DEBIAN"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE output
          ERROR_VARIABLE error
        )
        if(NOT ${result} EQUAL 0)
          message(FATAL_ERROR "Failed to compress: ${error}")
        endif()
        install ( FILES "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_CHANGELOG_INSTALL_FILENM}"
                  DESTINATION ${CMAKE_INSTALL_DOCDIR}
                  COMPONENT ${COMPONENT_NAME_T})
      endif()
    else()
        # License file
        install ( FILES ${LICENSE_FILE}
            DESTINATION ${CMAKE_INSTALL_DOCDIR} RENAME LICENSE.txt
            COMPONENT ${COMPONENT_NAME_T})
    endif()

    # Install lintian overrides
    if( BUILD_ENABLE_LINTIAN_OVERRIDES STREQUAL "ON" AND BUILD_DEBIAN_PKGING_FLAG STREQUAL "ON")
      set( OVERRIDE_FILE "${CMAKE_BINARY_DIR}/DEBIAN/${DEB_OVERRIDES_INSTALL_FILENM}" )
      install ( FILES ${OVERRIDE_FILE}
	  DESTINATION ${DEB_OVERRIDES_INSTALL_PATH}
          COMPONENT ${COMPONENT_NAME_T})
    endif()
endfunction()

# Set variables for changelog and copyright
# For Debian specific Packages
function( set_debian_pkg_cmake_flags DEB_PACKAGE_NAME_T DEB_PACKAGE_VERSION_T DEB_MAINTAINER_NM_T DEB_MAINTAINER_EMAIL_T )
    # Setting configure flags
    set( DEB_PACKAGE_NAME             "${DEB_PACKAGE_NAME_T}" CACHE STRING "Debian Package Name" )
    set( DEB_PACKAGE_VERSION          "${DEB_PACKAGE_VERSION_T}" CACHE STRING "Debian Package Version String" )
    set( DEB_MAINTAINER_NAME          "${DEB_MAINTAINER_NM_T}" CACHE STRING "Debian Package Maintainer Name" )
    set( DEB_MAINTAINER_EMAIL         "${DEB_MAINTAINER_EMAIL_T}" CACHE STRING "Debian Package Maintainer Email" )
    set( DEB_COPYRIGHT_YEAR           "2025" CACHE STRING "Debian Package Copyright Year" )
    set( DEB_LICENSE                  "MIT" CACHE STRING "Debian Package License Type" )
    set( DEB_CHANGELOG_INSTALL_FILENM "changelog.Debian.gz" CACHE STRING "Debian Package ChangeLog File Name" )

    if( BUILD_ENABLE_LINTIAN_OVERRIDES )
      set( DEB_OVERRIDES_INSTALL_FILENM "${DEB_PACKAGE_NAME}" CACHE STRING "Debian Package Lintian Override File Name" )
      set( DEB_OVERRIDES_INSTALL_PATH   "/usr/share/lintian/overrides/" CACHE STRING "Deb Pkg Lintian Override Install Loc" )
    endif()

    # Get TimeStamp
    find_program( DEB_DATE_TIMESTAMP_EXEC date )
    set ( DEB_TIMESTAMP_FORMAT_OPTION "-R" )
    execute_process (
        COMMAND ${DEB_DATE_TIMESTAMP_EXEC} ${DEB_TIMESTAMP_FORMAT_OPTION}
        OUTPUT_VARIABLE TIMESTAMP_T
    )
    set( DEB_TIMESTAMP                "${TIMESTAMP_T}" CACHE STRING "Current Time Stamp for Copyright/Changelog" )

    message(STATUS "DEB_PACKAGE_NAME             : ${DEB_PACKAGE_NAME}" )
    message(STATUS "DEB_PACKAGE_VERSION          : ${DEB_PACKAGE_VERSION}" )
    message(STATUS "DEB_MAINTAINER_NAME          : ${DEB_MAINTAINER_NAME}" )
    message(STATUS "DEB_MAINTAINER_EMAIL         : ${DEB_MAINTAINER_EMAIL}" )
    message(STATUS "DEB_COPYRIGHT_YEAR           : ${DEB_COPYRIGHT_YEAR}" )
    message(STATUS "DEB_LICENSE                  : ${DEB_LICENSE}" )
    message(STATUS "DEB_TIMESTAMP                : ${DEB_TIMESTAMP}" )
    message(STATUS "DEB_CHANGELOG_INSTALL_FILENM : ${DEB_CHANGELOG_INSTALL_FILENM}" )
endfunction()
