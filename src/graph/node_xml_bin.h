/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_PUGIXML

#  include "util/map.h"
#  include "util/param.h"
#  include "util/xml.h"

CCL_NAMESPACE_BEGIN

struct Node;

struct XMLReader {
  map<ustring, Node *> node_map;
  //string base;       /* Base path to current file. */
  //size_t offset = 0;
  std::shared_ptr<std::ifstream> file;
};

void xml_read_node(XMLReader &reader, Node *node, const xml_node xml_node);
xml_node xml_write_node(Node *node, xml_node xml_root);
bool xml_is_digit(const std::string& str);

CCL_NAMESPACE_END

#endif /* WITH_PUGIXML */
