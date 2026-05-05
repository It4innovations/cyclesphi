/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_PUGIXML

#  include "util/map.h"
#  include "util/param.h"
#  include "util/xml.h"
#  include "util/types_image.h"
#  include "kernel/types.h"

CCL_NAMESPACE_BEGIN

////////////////ENUM TO STR//////////////////////////////
// InterpolationType

const char* enum_to_str(InterpolationType t);
bool str_to_enum(const char* str, InterpolationType& out);

// ExtensionType

const char* enum_to_str(ExtensionType t);
bool str_to_enum(const char* str, ExtensionType& out);

// ImageDataType

const char* enum_to_str(ImageDataType t);
bool str_to_enum(const char* str, ImageDataType& out);

// AttributeStandard

const char* enum_to_str(AttributeStandard a);
bool str_to_enum(const char* str, AttributeStandard& out);

// AttributeElement

const char* enum_to_str(AttributeElement e);
bool str_to_enum(const char* str, AttributeElement& out);

// ParamValue::Interp

const char* enum_to_str(ParamValue::Interp i);
bool str_to_enum(const char* str, ParamValue::Interp& out);

// TypeDesc

const char* typedesc_to_cstr(const OIIO::TypeDesc& t);
OIIO::TypeDesc str_to_typedesc(OIIO::string_view s);

CCL_NAMESPACE_END

#endif /* WITH_PUGIXML */
