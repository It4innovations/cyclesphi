# SPDX-FileCopyrightText: 2011-2022 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

set(INC
  ..
)

set(INC_SYS
)

set(SRC
  node.cpp
  node_type.cpp
  #node_xml.cpp
  node_xml_bin.cpp
)

set(SRC_HEADERS
  node.h
  node_enum.h
  node_type.h
  #node_xml.h
  node_xml_bin.h
  node_xml_util.h
)

set(LIB
  cycles_util
)

include_directories(${INC})
include_directories(SYSTEM ${INC_SYS})

cycles_add_library(cycles_graph "${LIB}" ${SRC} ${SRC_HEADERS})
