/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "graph/node_xml_util.h"

#ifdef WITH_PUGIXML

CCL_NAMESPACE_BEGIN

////////////////ENUM TO STR//////////////////////////////
// InterpolationType

const char* enum_to_str(InterpolationType t)
{
  switch (t) {
  case INTERPOLATION_NONE:      return "INTERPOLATION_NONE";
  case INTERPOLATION_LINEAR:    return "INTERPOLATION_LINEAR";
  case INTERPOLATION_CLOSEST:   return "INTERPOLATION_CLOSEST";
  case INTERPOLATION_CUBIC:     return "INTERPOLATION_CUBIC";
  case INTERPOLATION_SMART:     return "INTERPOLATION_SMART";
  case INTERPOLATION_NUM_TYPES: return "INTERPOLATION_NUM_TYPES";
  default: {
    printf("enum_to_str: Unknown InterpolationType enum: %d\n", (int)t);
    return "UNKNOWN_INTERPOLATION_TYPE";
  }
  }
}

bool str_to_enum(const char* str, InterpolationType& out)
{
  if (!str) {
    return false;
  }

  if (std::strcmp(str, "INTERPOLATION_NONE") == 0) {
    out = INTERPOLATION_NONE;
  }
  else if (std::strcmp(str, "INTERPOLATION_LINEAR") == 0) {
    out = INTERPOLATION_LINEAR;
  }
  else if (std::strcmp(str, "INTERPOLATION_CLOSEST") == 0) {
    out = INTERPOLATION_CLOSEST;
  }
  else if (std::strcmp(str, "INTERPOLATION_CUBIC") == 0) {
    out = INTERPOLATION_CUBIC;
  }
  else if (std::strcmp(str, "INTERPOLATION_SMART") == 0) {
    out = INTERPOLATION_SMART;
  }
  else if (std::strcmp(str, "INTERPOLATION_NUM_TYPES") == 0) {
    out = INTERPOLATION_NUM_TYPES;
  }
  else {
    printf("str_to_enum: Unknown InterpolationType string: %s\n", str);
    return false;
  }

  return true;
}

// ExtensionType

const char* enum_to_str(ExtensionType t)
{
  switch (t) {
  case EXTENSION_REPEAT:    return "EXTENSION_REPEAT";
  case EXTENSION_EXTEND:    return "EXTENSION_EXTEND";
  case EXTENSION_CLIP:      return "EXTENSION_CLIP";
  case EXTENSION_MIRROR:    return "EXTENSION_MIRROR";
  case EXTENSION_NUM_TYPES: return "EXTENSION_NUM_TYPES";
  default: {
    printf("enum_to_str: Unknown ExtensionType enum: %d\n", (int)t);
    return "UNKNOWN_EXTENSION_TYPE";
  }
  }
}

bool str_to_enum(const char* str, ExtensionType& out)
{
  if (!str) {
    return false;
  }

  if (std::strcmp(str, "EXTENSION_REPEAT") == 0) {
    out = EXTENSION_REPEAT;
  }
  else if (std::strcmp(str, "EXTENSION_EXTEND") == 0) {
    out = EXTENSION_EXTEND;
  }
  else if (std::strcmp(str, "EXTENSION_CLIP") == 0) {
    out = EXTENSION_CLIP;
  }
  else if (std::strcmp(str, "EXTENSION_MIRROR") == 0) {
    out = EXTENSION_MIRROR;
  }
  else if (std::strcmp(str, "EXTENSION_NUM_TYPES") == 0) {
    out = EXTENSION_NUM_TYPES;
  }
  else {
    printf("str_to_enum: Unknown ExtensionType string: %s\n", str);
    return false;
  }

  return true;
}

// ImageDataType

const char* enum_to_str(ImageDataType t)
{
  switch (t) {
  case IMAGE_DATA_TYPE_FLOAT4:         return "IMAGE_DATA_TYPE_FLOAT4";
  case IMAGE_DATA_TYPE_BYTE4:          return "IMAGE_DATA_TYPE_BYTE4";
  case IMAGE_DATA_TYPE_HALF4:          return "IMAGE_DATA_TYPE_HALF4";
  case IMAGE_DATA_TYPE_FLOAT:          return "IMAGE_DATA_TYPE_FLOAT";
  case IMAGE_DATA_TYPE_BYTE:           return "IMAGE_DATA_TYPE_BYTE";
  case IMAGE_DATA_TYPE_HALF:           return "IMAGE_DATA_TYPE_HALF";
  case IMAGE_DATA_TYPE_USHORT4:        return "IMAGE_DATA_TYPE_USHORT4";
  case IMAGE_DATA_TYPE_USHORT:         return "IMAGE_DATA_TYPE_USHORT";
  case IMAGE_DATA_TYPE_NANOVDB_FLOAT:  return "IMAGE_DATA_TYPE_NANOVDB_FLOAT";
  case IMAGE_DATA_TYPE_NANOVDB_FLOAT3: return "IMAGE_DATA_TYPE_NANOVDB_FLOAT3";
  case IMAGE_DATA_TYPE_NANOVDB_FLOAT4: return "IMAGE_DATA_TYPE_NANOVDB_FLOAT4";
  case IMAGE_DATA_TYPE_NANOVDB_FPN:    return "IMAGE_DATA_TYPE_NANOVDB_FPN";
  case IMAGE_DATA_TYPE_NANOVDB_FP16:   return "IMAGE_DATA_TYPE_NANOVDB_FP16";
  case IMAGE_DATA_TYPE_NANOVDB_EMPTY:  return "IMAGE_DATA_TYPE_NANOVDB_EMPTY";
  case IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT: return "IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT";
  case IMAGE_DATA_TYPE_NANOVDB_DERIVATES: return "IMAGE_DATA_TYPE_NANOVDB_DERIVATES";
  case IMAGE_DATA_TYPE_NANOVDB_DERIVATES_VEC4: return "IMAGE_DATA_TYPE_NANOVDB_DERIVATES_VEC4";
  case IMAGE_DATA_TYPE_RAW3D_FLOAT:    return "IMAGE_DATA_TYPE_RAW3D_FLOAT";
  case IMAGE_DATA_TYPE_RAW3D_FLOAT3:   return "IMAGE_DATA_TYPE_RAW3D_FLOAT3";
  case IMAGE_DATA_TYPE_ZFP_FLOAT:      return "IMAGE_DATA_TYPE_ZFP_FLOAT";
  case IMAGE_DATA_NUM_TYPES:           return "IMAGE_DATA_NUM_TYPES";
  default: {
    printf("enum_to_str: Unknown ImageDataType enum: %d\n", (int)t);
    return "UNKNOWN_IMAGE_DATA_TYPE";
  }
  }
}

bool str_to_enum(const char* str, ImageDataType& out)
{
  if (!str) {
    return false;
  }

  if (std::strcmp(str, "IMAGE_DATA_TYPE_FLOAT4") == 0) {
    out = IMAGE_DATA_TYPE_FLOAT4;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_BYTE4") == 0) {
    out = IMAGE_DATA_TYPE_BYTE4;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_HALF4") == 0) {
    out = IMAGE_DATA_TYPE_HALF4;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_FLOAT") == 0) {
    out = IMAGE_DATA_TYPE_FLOAT;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_BYTE") == 0) {
    out = IMAGE_DATA_TYPE_BYTE;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_HALF") == 0) {
    out = IMAGE_DATA_TYPE_HALF;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_USHORT4") == 0) {
    out = IMAGE_DATA_TYPE_USHORT4;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_USHORT") == 0) {
    out = IMAGE_DATA_TYPE_USHORT;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_FLOAT") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_FLOAT;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_FLOAT3") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_FLOAT3;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_FLOAT4") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_FLOAT4;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_FPN") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_FPN;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_FP16") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_FP16;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_EMPTY") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_EMPTY;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_DERIVATES") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_DERIVATES;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_NANOVDB_DERIVATES_VEC4") == 0) {
    out = IMAGE_DATA_TYPE_NANOVDB_DERIVATES_VEC4;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_RAW3D_FLOAT") == 0) {
    out = IMAGE_DATA_TYPE_RAW3D_FLOAT;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_RAW3D_FLOAT3") == 0) {
    out = IMAGE_DATA_TYPE_RAW3D_FLOAT3;
  }
  else if (std::strcmp(str, "IMAGE_DATA_TYPE_ZFP_FLOAT") == 0) {
    out = IMAGE_DATA_TYPE_ZFP_FLOAT;
  }
  else if (std::strcmp(str, "IMAGE_DATA_NUM_TYPES") == 0) {
    out = IMAGE_DATA_NUM_TYPES;
  }
  else {
    printf("str_to_enum: Unknown ImageDataType string: %s\n", str);
    return false;
  }

  return true;
}

// AttributeStandard
const char* enum_to_str(AttributeStandard a)
{
  switch (a) {
  case ATTR_STD_NONE:                     return "ATTR_STD_NONE";
  case ATTR_STD_VERTEX_NORMAL:            return "ATTR_STD_VERTEX_NORMAL";
  case ATTR_STD_CORNER_NORMAL:            return "ATTR_STD_CORNER_NORMAL";
  case ATTR_STD_UV:                       return "ATTR_STD_UV";
  case ATTR_STD_UV_TANGENT:               return "ATTR_STD_UV_TANGENT";
  case ATTR_STD_UV_TANGENT_SIGN:          return "ATTR_STD_UV_TANGENT_SIGN";
  case ATTR_STD_UV_TANGENT_UNDISPLACED:   return "ATTR_STD_UV_TANGENT_UNDISPLACED";
  case ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED:
    return "ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED";
  case ATTR_STD_VERTEX_COLOR:             return "ATTR_STD_VERTEX_COLOR";
  case ATTR_STD_GENERATED:                return "ATTR_STD_GENERATED";
  case ATTR_STD_GENERATED_TRANSFORM:      return "ATTR_STD_GENERATED_TRANSFORM";
  case ATTR_STD_POSITION_UNDEFORMED:      return "ATTR_STD_POSITION_UNDEFORMED";
  case ATTR_STD_POSITION_UNDISPLACED:     return "ATTR_STD_POSITION_UNDISPLACED";
  case ATTR_STD_NORMAL_UNDISPLACED:       return "ATTR_STD_NORMAL_UNDISPLACED";
  case ATTR_STD_MOTION_VERTEX_POSITION:   return "ATTR_STD_MOTION_VERTEX_POSITION";
  case ATTR_STD_MOTION_VERTEX_NORMAL:     return "ATTR_STD_MOTION_VERTEX_NORMAL";
  case ATTR_STD_MOTION_CORNER_NORMAL:     return "ATTR_STD_MOTION_CORNER_NORMAL";
  case ATTR_STD_PARTICLE:                 return "ATTR_STD_PARTICLE";
  case ATTR_STD_CURVE_INTERCEPT:          return "ATTR_STD_CURVE_INTERCEPT";
  case ATTR_STD_CURVE_LENGTH:             return "ATTR_STD_CURVE_LENGTH";
  case ATTR_STD_CURVE_RANDOM:             return "ATTR_STD_CURVE_RANDOM";
  case ATTR_STD_POINT_RANDOM:             return "ATTR_STD_POINT_RANDOM";
  case ATTR_STD_PTEX_FACE_ID:              return "ATTR_STD_PTEX_FACE_ID";
  case ATTR_STD_PTEX_UV:                   return "ATTR_STD_PTEX_UV";
  case ATTR_STD_VOLUME_DENSITY:            return "ATTR_STD_VOLUME_DENSITY";
  case ATTR_STD_VOLUME_COLOR:              return "ATTR_STD_VOLUME_COLOR";
  case ATTR_STD_VOLUME_FLAME:              return "ATTR_STD_VOLUME_FLAME";
  case ATTR_STD_VOLUME_HEAT:               return "ATTR_STD_VOLUME_HEAT";
  case ATTR_STD_VOLUME_TEMPERATURE:        return "ATTR_STD_VOLUME_TEMPERATURE";
  case ATTR_STD_VOLUME_VELOCITY:           return "ATTR_STD_VOLUME_VELOCITY";
  case ATTR_STD_VOLUME_VELOCITY_X:         return "ATTR_STD_VOLUME_VELOCITY_X";
  case ATTR_STD_VOLUME_VELOCITY_Y:         return "ATTR_STD_VOLUME_VELOCITY_Y";
  case ATTR_STD_VOLUME_VELOCITY_Z:         return "ATTR_STD_VOLUME_VELOCITY_Z";
  case ATTR_STD_POINTINESS:                return "ATTR_STD_POINTINESS";
  case ATTR_STD_RANDOM_PER_ISLAND:         return "ATTR_STD_RANDOM_PER_ISLAND";
  case ATTR_STD_SHADOW_TRANSPARENCY:       return "ATTR_STD_SHADOW_TRANSPARENCY";
  case ATTR_STD_NUM:                       return "ATTR_STD_NUM";
  case ATTR_STD_NOT_FOUND:                 return "ATTR_STD_NOT_FOUND";
  default: {
    printf("enum_to_str: Unknown AttributeStandard enum: %d\n", (int)a);
    return "UNKNOWN_ATTRIBUTE_STANDARD";
  }
  }
}

bool str_to_enum(const char* str, AttributeStandard& out)
{
  if (!str) {
    return false;
  }

  if (std::strcmp(str, "ATTR_STD_NONE") == 0)                     out = ATTR_STD_NONE;
  else if (std::strcmp(str, "ATTR_STD_VERTEX_NORMAL") == 0)            out = ATTR_STD_VERTEX_NORMAL;
  else if (std::strcmp(str, "ATTR_STD_CORNER_NORMAL") == 0)            out = ATTR_STD_CORNER_NORMAL;
  else if (std::strcmp(str, "ATTR_STD_UV") == 0)                       out = ATTR_STD_UV;
  else if (std::strcmp(str, "ATTR_STD_UV_TANGENT") == 0)               out = ATTR_STD_UV_TANGENT;
  else if (std::strcmp(str, "ATTR_STD_UV_TANGENT_SIGN") == 0)          out = ATTR_STD_UV_TANGENT_SIGN;
  else if (std::strcmp(str, "ATTR_STD_UV_TANGENT_UNDISPLACED") == 0)
    out = ATTR_STD_UV_TANGENT_UNDISPLACED;
  else if (std::strcmp(str, "ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED") == 0)
    out = ATTR_STD_UV_TANGENT_SIGN_UNDISPLACED;
  else if (std::strcmp(str, "ATTR_STD_VERTEX_COLOR") == 0)             out = ATTR_STD_VERTEX_COLOR;
  else if (std::strcmp(str, "ATTR_STD_GENERATED") == 0)                out = ATTR_STD_GENERATED;
  else if (std::strcmp(str, "ATTR_STD_GENERATED_TRANSFORM") == 0)      out = ATTR_STD_GENERATED_TRANSFORM;
  else if (std::strcmp(str, "ATTR_STD_POSITION_UNDEFORMED") == 0)      out = ATTR_STD_POSITION_UNDEFORMED;
  else if (std::strcmp(str, "ATTR_STD_POSITION_UNDISPLACED") == 0)     out = ATTR_STD_POSITION_UNDISPLACED;
  else if (std::strcmp(str, "ATTR_STD_NORMAL_UNDISPLACED") == 0)       out = ATTR_STD_NORMAL_UNDISPLACED;
  else if (std::strcmp(str, "ATTR_STD_MOTION_VERTEX_POSITION") == 0)   out = ATTR_STD_MOTION_VERTEX_POSITION;
  else if (std::strcmp(str, "ATTR_STD_MOTION_VERTEX_NORMAL") == 0)     out = ATTR_STD_MOTION_VERTEX_NORMAL;
  else if (std::strcmp(str, "ATTR_STD_MOTION_CORNER_NORMAL") == 0)     out = ATTR_STD_MOTION_CORNER_NORMAL;
  else if (std::strcmp(str, "ATTR_STD_PARTICLE") == 0)                 out = ATTR_STD_PARTICLE;
  else if (std::strcmp(str, "ATTR_STD_CURVE_INTERCEPT") == 0)          out = ATTR_STD_CURVE_INTERCEPT;
  else if (std::strcmp(str, "ATTR_STD_CURVE_LENGTH") == 0)             out = ATTR_STD_CURVE_LENGTH;
  else if (std::strcmp(str, "ATTR_STD_CURVE_RANDOM") == 0)             out = ATTR_STD_CURVE_RANDOM;
  else if (std::strcmp(str, "ATTR_STD_POINT_RANDOM") == 0)             out = ATTR_STD_POINT_RANDOM;
  else if (std::strcmp(str, "ATTR_STD_PTEX_FACE_ID") == 0)              out = ATTR_STD_PTEX_FACE_ID;
  else if (std::strcmp(str, "ATTR_STD_PTEX_UV") == 0)                   out = ATTR_STD_PTEX_UV;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_DENSITY") == 0)            out = ATTR_STD_VOLUME_DENSITY;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_COLOR") == 0)              out = ATTR_STD_VOLUME_COLOR;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_FLAME") == 0)              out = ATTR_STD_VOLUME_FLAME;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_HEAT") == 0)               out = ATTR_STD_VOLUME_HEAT;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_TEMPERATURE") == 0)        out = ATTR_STD_VOLUME_TEMPERATURE;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_VELOCITY") == 0)           out = ATTR_STD_VOLUME_VELOCITY;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_VELOCITY_X") == 0)         out = ATTR_STD_VOLUME_VELOCITY_X;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_VELOCITY_Y") == 0)         out = ATTR_STD_VOLUME_VELOCITY_Y;
  else if (std::strcmp(str, "ATTR_STD_VOLUME_VELOCITY_Z") == 0)         out = ATTR_STD_VOLUME_VELOCITY_Z;
  else if (std::strcmp(str, "ATTR_STD_POINTINESS") == 0)                out = ATTR_STD_POINTINESS;
  else if (std::strcmp(str, "ATTR_STD_RANDOM_PER_ISLAND") == 0)         out = ATTR_STD_RANDOM_PER_ISLAND;
  else if (std::strcmp(str, "ATTR_STD_SHADOW_TRANSPARENCY") == 0)       out = ATTR_STD_SHADOW_TRANSPARENCY;
  else if (std::strcmp(str, "ATTR_STD_NUM") == 0)                       out = ATTR_STD_NUM;
  else if (std::strcmp(str, "ATTR_STD_NOT_FOUND") == 0)                 out = ATTR_STD_NOT_FOUND;
  else {
    printf("str_to_enum: Unknown AttributeStandard string: %s\n", str);
    return false;
  }

  return true;
}

const char* enum_to_str(AttributeElement e)
{
  switch (e) {
  case ATTR_ELEMENT_NONE:               return "ATTR_ELEMENT_NONE";
  case ATTR_ELEMENT_OBJECT:             return "ATTR_ELEMENT_OBJECT";
  case ATTR_ELEMENT_MESH:               return "ATTR_ELEMENT_MESH";
  case ATTR_ELEMENT_FACE:               return "ATTR_ELEMENT_FACE";
  case ATTR_ELEMENT_VERTEX:             return "ATTR_ELEMENT_VERTEX";
  case ATTR_ELEMENT_VERTEX_MOTION:      return "ATTR_ELEMENT_VERTEX_MOTION";
  case ATTR_ELEMENT_VERTEX_NORMAL:      return "ATTR_ELEMENT_VERTEX_NORMAL";
  case ATTR_ELEMENT_VERTEX_NORMAL_MOTION:
    return "ATTR_ELEMENT_VERTEX_NORMAL_MOTION";
  case ATTR_ELEMENT_CORNER:              return "ATTR_ELEMENT_CORNER";
  case ATTR_ELEMENT_CORNER_BYTE:         return "ATTR_ELEMENT_CORNER_BYTE";
  case ATTR_ELEMENT_CORNER_NORMAL:       return "ATTR_ELEMENT_CORNER_NORMAL";
  case ATTR_ELEMENT_CORNER_NORMAL_MOTION:
    return "ATTR_ELEMENT_CORNER_NORMAL_MOTION";
  case ATTR_ELEMENT_CURVE:               return "ATTR_ELEMENT_CURVE";
  case ATTR_ELEMENT_CURVE_KEY:           return "ATTR_ELEMENT_CURVE_KEY";
  case ATTR_ELEMENT_CURVE_KEY_MOTION:    return "ATTR_ELEMENT_CURVE_KEY_MOTION";
  case ATTR_ELEMENT_CURVE_KEY_NORMAL:    return "ATTR_ELEMENT_CURVE_KEY_NORMAL";
  case ATTR_ELEMENT_CURVE_KEY_NORMAL_MOTION:
    return "ATTR_ELEMENT_CURVE_KEY_NORMAL_MOTION";
  case ATTR_ELEMENT_VOXEL:               return "ATTR_ELEMENT_VOXEL";
  case ATTR_ELEMENT_IS_MOTION:           return "ATTR_ELEMENT_IS_MOTION";
  case ATTR_ELEMENT_IS_NORMAL:           return "ATTR_ELEMENT_IS_NORMAL";
  case ATTR_ELEMENT_IS_BYTE:             return "ATTR_ELEMENT_IS_BYTE";
  default: {
    printf("enum_to_str: Unknown AttributeElement enum: %d\n", (int)e);
    return "UNKNOWN_ATTRIBUTE_ELEMENT";
  }
  }
}

bool str_to_enum(const char* str, AttributeElement& out)
{
  if (!str) {
    return false;
  }

  if (std::strcmp(str, "ATTR_ELEMENT_NONE") == 0)                out = ATTR_ELEMENT_NONE;
  else if (std::strcmp(str, "ATTR_ELEMENT_OBJECT") == 0)              out = ATTR_ELEMENT_OBJECT;
  else if (std::strcmp(str, "ATTR_ELEMENT_MESH") == 0)                out = ATTR_ELEMENT_MESH;
  else if (std::strcmp(str, "ATTR_ELEMENT_FACE") == 0)                out = ATTR_ELEMENT_FACE;
  else if (std::strcmp(str, "ATTR_ELEMENT_VERTEX") == 0)              out = ATTR_ELEMENT_VERTEX;
  else if (std::strcmp(str, "ATTR_ELEMENT_VERTEX_MOTION") == 0)       out = ATTR_ELEMENT_VERTEX_MOTION;
  else if (std::strcmp(str, "ATTR_ELEMENT_VERTEX_NORMAL") == 0)       out = ATTR_ELEMENT_VERTEX_NORMAL;
  else if (std::strcmp(str, "ATTR_ELEMENT_VERTEX_NORMAL_MOTION") == 0)
    out = ATTR_ELEMENT_VERTEX_NORMAL_MOTION;
  else if (std::strcmp(str, "ATTR_ELEMENT_CORNER") == 0)               out = ATTR_ELEMENT_CORNER;
  else if (std::strcmp(str, "ATTR_ELEMENT_CORNER_BYTE") == 0)          out = ATTR_ELEMENT_CORNER_BYTE;
  else if (std::strcmp(str, "ATTR_ELEMENT_CORNER_NORMAL") == 0)        out = ATTR_ELEMENT_CORNER_NORMAL;
  else if (std::strcmp(str, "ATTR_ELEMENT_CORNER_NORMAL_MOTION") == 0)
    out = ATTR_ELEMENT_CORNER_NORMAL_MOTION;
  else if (std::strcmp(str, "ATTR_ELEMENT_CURVE") == 0)                out = ATTR_ELEMENT_CURVE;
  else if (std::strcmp(str, "ATTR_ELEMENT_CURVE_KEY") == 0)            out = ATTR_ELEMENT_CURVE_KEY;
  else if (std::strcmp(str, "ATTR_ELEMENT_CURVE_KEY_MOTION") == 0)     out = ATTR_ELEMENT_CURVE_KEY_MOTION;
  else if (std::strcmp(str, "ATTR_ELEMENT_CURVE_KEY_NORMAL") == 0)     out = ATTR_ELEMENT_CURVE_KEY_NORMAL;
  else if (std::strcmp(str, "ATTR_ELEMENT_CURVE_KEY_NORMAL_MOTION") == 0)
    out = ATTR_ELEMENT_CURVE_KEY_NORMAL_MOTION;
  else if (std::strcmp(str, "ATTR_ELEMENT_VOXEL") == 0)                out = ATTR_ELEMENT_VOXEL;
  else if (std::strcmp(str, "ATTR_ELEMENT_IS_MOTION") == 0)            out = ATTR_ELEMENT_IS_MOTION;
  else if (std::strcmp(str, "ATTR_ELEMENT_IS_NORMAL") == 0)            out = ATTR_ELEMENT_IS_NORMAL;
  else if (std::strcmp(str, "ATTR_ELEMENT_IS_BYTE") == 0)              out = ATTR_ELEMENT_IS_BYTE;
  else {
    printf("str_to_enum: Unknown AttributeElement string: %s\n", str);
    return false;
  }

  return true;
}

const char* enum_to_str(ParamValue::Interp i)
{
  switch (i) {
  case ParamValue::Interp::INTERP_CONSTANT: return "INTERP_CONSTANT";
  case ParamValue::Interp::INTERP_PERPIECE: return "INTERP_PERPIECE";
  case ParamValue::Interp::INTERP_LINEAR:   return "INTERP_LINEAR";
  case ParamValue::Interp::INTERP_VERTEX:   return "INTERP_VERTEX";
  default:              return "UNKNOWN_INTERP";
  }
}

bool str_to_enum(const char* str, ParamValue::Interp& out)
{
  if (!str) return false;

  if (std::strcmp(str, "INTERP_CONSTANT") == 0) out = ParamValue::Interp::INTERP_CONSTANT;
  else if (std::strcmp(str, "INTERP_PERPIECE") == 0) out = ParamValue::Interp::INTERP_PERPIECE;
  else if (std::strcmp(str, "INTERP_LINEAR") == 0)   out = ParamValue::Interp::INTERP_LINEAR;
  else if (std::strcmp(str, "INTERP_VERTEX") == 0)   out = ParamValue::Interp::INTERP_VERTEX;
  else return false;

  return true;
}

//////////////////////////////////////////TYPEDESC///////
const char* typedesc_to_cstr(const OIIO::TypeDesc& t)
{
  return t.c_str();
}

OIIO::TypeDesc str_to_typedesc(OIIO::string_view s)
{
  // If parsing fails, TypeDesc will become UNKNOWN (per comment in the header).
  return OIIO::TypeDesc(s);
}

CCL_NAMESPACE_END

#endif /* WITH_PUGIXML */
