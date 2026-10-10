/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_PUGIXML

#  include <functional>

#  include "util/map.h"
#  include "util/param.h"
#  include "util/string.h"
#  include "util/xml.h"

CCL_NAMESPACE_BEGIN

struct Node;

/* Receives the warnings and errors of the XML reader. `error` is set for problems which make
 * the result unusable (an unknown node type, a connection to an unknown node), and not set for
 * values and links which were ignored (an unknown socket, a wrong number of values). */
using XMLReportFunc = std::function<void(bool error, const string &message)>;

struct XMLReader {
  map<ustring, Node *> node_map;
  //string base;       /* Base path to current file. */
  //size_t offset = 0;
  /* Binary sidecar file with the arrays. Arrays are read from the XML text when it is not open. */
  std::shared_ptr<std::ifstream> file;
  /* Optional, the messages are printed to stderr when it is not set. */
  XMLReportFunc report;
};

void xml_report(const XMLReader &reader, bool error, const char *format, ...);
bool is_file_open(XMLReader &reader);
void xml_read_node(XMLReader &reader, Node *node, const xml_node xml_node);
xml_node xml_write_node(Node *node, xml_node xml_root);
bool xml_is_digit(const std::string& str);

CCL_NAMESPACE_END

#endif /* WITH_PUGIXML */
