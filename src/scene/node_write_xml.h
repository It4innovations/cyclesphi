/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "graph/node.h"

#include "util/map.h"
#include "util/string.h"
#include "util/xml.h"

CCL_NAMESPACE_BEGIN

//struct XMLReader {
//  map<ustring, Node *> node_map;
//};

struct XMLWriter {
  //std::string base;       /* Base path to current file. */
  size_t offset = 0;
  std::ofstream file;
};

//void xml_read_node(XMLReader &reader, Node *node, xml_node xml_node);
xml_node xml_write_node(XMLWriter &writer, Node *node, xml_node xml_root);
string xml_pointer_to_name(void* ptr);
bool xml_pointer_to_name_check(Node* node);

CCL_NAMESPACE_END
