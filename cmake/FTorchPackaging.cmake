# Set package information
set(
  CPACK_PACKAGE_VENDOR
  "University of Cambridge Institute of Computing for Climate Science")
set(CPACK_PACKAGE_CONTACT "Jared Frazier (cscidev001@gmail.com)")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Fortran interface to PyTorch models")
set(CPACK_PACKAGE_DESCRIPTION
  "FTorch is a library for calling PyTorch machine learning models directly
from Fortran code. It enables coupling TorchScript models trained in PyTorch
to existing Fortran applications without requiring Python at runtime.")
set(CPACK_PACKAGE_VERSION_MAJOR "${CMAKE_PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${CMAKE_PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${CMAKE_PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION "${CMAKE_PROJECT_VERSION}")

# Explicitly set architecture
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")

# Set the names of the package that get generated
# e.g., `make package` produces ${CPACK_PACKAGE_FILE_NAME}.tar.gz
set(
  CPACK_PACKAGE_FILE_NAME
  "${CMAKE_PROJECT_NAME}-"
  "${CPACK_PACKAGE_VERSION}-"
  "${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")
set(
  CPACK_SOURCE_PACKAGE_FILE_NAME
  "${CMAKE_PROJECT_NAME}-"
  "${CPACK_PACKAGE_VERSION}-"
  "${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")

# Generate better dependency list
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS TRUE)

include(CPack)
