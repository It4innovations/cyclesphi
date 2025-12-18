/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/image_vdb.h"

#include "util/log.h"
#include "util/openvdb.h"

#ifdef WITH_OPENVDB
#  include <openvdb/tools/Dense.h>
#endif
#ifdef WITH_NANOVDB
#  define NANOVDB_USE_OPENVDB
#  include <nanovdb/NanoVDB.h>
#  if NANOVDB_MAJOR_VERSION_NUMBER > 32 || \
      (NANOVDB_MAJOR_VERSION_NUMBER == 32 && NANOVDB_MINOR_VERSION_NUMBER >= 7)
#    include <nanovdb/tools/CreateNanoGrid.h>
#  else
#    include <nanovdb/util/OpenToNanoVDB.h>
#  endif
#endif

CCL_NAMESPACE_BEGIN

#ifdef WITH_OPENVDB
struct NumChannelsOp {
  int num_channels = 0;

  template<typename GridType, typename FloatGridType, typename FloatDataType, const int channels>
  bool operator()(const openvdb::GridBase::ConstPtr & /*unused*/)
  {
    num_channels = channels;
    return true;
  }
};

struct ToDenseOp {
  openvdb::CoordBBox bbox;
  void *pixels;

  template<typename GridType, typename FloatGridType, typename FloatDataType, const int channels>
  bool operator()(const openvdb::GridBase::ConstPtr &grid)
  {
    openvdb::tools::Dense<FloatDataType, openvdb::tools::LayoutXYZ> dense(bbox,
                                                                          (FloatDataType *)pixels);
    openvdb::tools::copyToDense(*openvdb::gridConstPtrCast<GridType>(grid), dense);
    return true;
  }
};

#  ifdef WITH_NANOVDB
struct ToNanoOp {
  nanovdb::GridHandle<> nanogrid;
  int precision;

  template<typename GridType, typename FloatGridType, typename FloatDataType, const int channels>
  bool operator()(const openvdb::GridBase::ConstPtr &grid)
  {
    if constexpr (!std::is_same_v<GridType, openvdb::MaskGrid>) {
      try {
#    if NANOVDB_MAJOR_VERSION_NUMBER > 32 || \
        (NANOVDB_MAJOR_VERSION_NUMBER == 32 && NANOVDB_MINOR_VERSION_NUMBER >= 6)
#      if NANOVDB_MAJOR_VERSION_NUMBER > 32 || \
          (NANOVDB_MAJOR_VERSION_NUMBER == 32 && NANOVDB_MINOR_VERSION_NUMBER >= 7)
        /* OpenVDB 12. */
        using nanovdb::tools::createNanoGrid;
        using nanovdb::tools::StatsMode;
#      else
        /* OpenVDB 11. */
        using nanovdb::createNanoGrid;
        using nanovdb::StatsMode;
#      endif

        if constexpr (std::is_same_v<FloatGridType, openvdb::FloatGrid>) {
          const openvdb::FloatGrid floatgrid(*openvdb::gridConstPtrCast<GridType>(grid));
          if (precision == 0) {
            nanogrid = createNanoGrid<openvdb::FloatGrid, nanovdb::FpN>(floatgrid);
          }
          else if (precision == 16) {
            nanogrid = createNanoGrid<openvdb::FloatGrid, nanovdb::Fp16>(floatgrid);
          }
          else {
            nanogrid = createNanoGrid<openvdb::FloatGrid, float>(floatgrid);
          }
        }
        else if constexpr (std::is_same_v<FloatGridType, openvdb::Vec3fGrid>) {
          const openvdb::Vec3fGrid floatgrid(*openvdb::gridConstPtrCast<GridType>(grid));
          nanogrid = createNanoGrid<openvdb::Vec3fGrid, nanovdb::Vec3f>(floatgrid,
                                                                        StatsMode::Disable);
        }
#    else
        /* OpenVDB 10. */
        if constexpr (std::is_same_v<FloatGridType, openvdb::FloatGrid>) {
          openvdb::FloatGrid floatgrid(*openvdb::gridConstPtrCast<GridType>(grid));
          if (precision == 0) {
            nanogrid =
                nanovdb::openToNanoVDB<nanovdb::HostBuffer, openvdb::FloatTree, nanovdb::FpN>(
                    floatgrid);
          }
          else if (precision == 16) {
            nanogrid =
                nanovdb::openToNanoVDB<nanovdb::HostBuffer, openvdb::FloatTree, nanovdb::Fp16>(
                    floatgrid);
          }
          else {
            nanogrid = nanovdb::openToNanoVDB(floatgrid);
          }
        }
        else if constexpr (std::is_same_v<FloatGridType, openvdb::Vec3fGrid>) {
          openvdb::Vec3fGrid floatgrid(*openvdb::gridConstPtrCast<GridType>(grid));
          nanogrid = nanovdb::openToNanoVDB(floatgrid);
        }
#    endif
      }
      catch (const std::exception &e) {
        VLOG_WARNING << "Error converting OpenVDB to NanoVDB grid: " << e.what();
      }
      catch (...) {
        VLOG_WARNING << "Error converting OpenVDB to NanoVDB grid: Unknown error";
      }
      return true;
    }
    else {
      return false;
    }
  }
};
#  endif

VDBImageLoader::VDBImageLoader(openvdb::GridBase::ConstPtr grid_, const string &grid_name)
    : grid_name(grid_name), grid(grid_)
{
}
#endif

VDBImageLoader::VDBImageLoader(const string &grid_name) : grid_name(grid_name) {}

VDBImageLoader::~VDBImageLoader() = default;

#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)

nanovdb::GridHandle<> VDBImageLoader::convert(openvdb::GridBase::ConstPtr g, int p)
{
    ToNanoOp op;
    op.precision = p;
    if (!openvdb::grid_type_operation(g, op)) {
        return nanovdb::GridHandle<>();
    }

    return std::move(op.nanogrid);
}

void VDBImageLoader::get_texture_info(nanovdb::NanoGrid<float>* nanogrid, size_t ng_size, TextureInfo& info)
{
    //nanovdb::NanoGrid<float>* nanogrid = (nanovdb::NanoGrid<float>*) ng.data();

    int data_type = (nanogrid->gridType() == nanovdb::GridType::Float) ? 1 : 3; // TODO

    ///* Set dimensions. */
    //auto bbox = nanogrid->worldBBox();
    //if (bbox.empty()) {
    //    return;
    //}

    //auto dim = bbox.dim();
    //info.width = dim[0];
    //info.height = dim[1];
    //info.depth = dim[2];

    info.width = ng_size;
    info.height = 0;
    info.depth = 0;    

    if (nanogrid) {
        //metadata.byte_size = nanogrid_data.size();
        if (data_type == 1) {
            info.data_type = IMAGE_DATA_TYPE_NANOVDB_FLOAT;
        }
        else {
            info.data_type = IMAGE_DATA_TYPE_NANOVDB_FLOAT3;
        }
    }

    /* Set transform from object space to voxel index. */
    //matMult(mInvMatD, Vec3T(xyz[0] - mVecD[0], xyz[1] - mVecD[1], xyz[2] - mVecD[2]));
    const double* matD = nanogrid->map().mMatD;
    const double* vecD = nanogrid->map().mVecD;

    Transform index_to_object;
    //for (int col = 0; col < 4; col++) {
    //    for (int row = 0; row < 3; row++) {
    //        index_to_object[row][col] = (float)grid_matrix[col][row];
    //    }
    //}

    for (int i = 0; i < 3; ++i) {
        index_to_object[i].x = static_cast<float>(matD[i * 3 + 0]);
        index_to_object[i].y = static_cast<float>(matD[i * 3 + 1]);
        index_to_object[i].z = static_cast<float>(matD[i * 3 + 2]);
        index_to_object[i].w = static_cast<float>(vecD[i]);
    }

    //Transform texture_to_index;
    //if (get_nanogrid()) {
    //    texture_to_index = transform_identity();
    //}

    info.transform_3d = transform_inverse(index_to_object);
    info.use_transform_3d = true;

    ///* Set dimensions. */
    //openvdb::CoordBBox bbox = g->evalActiveVoxelBoundingBox();
    //if (bbox.empty()) {
    //    return;
    //}

    //openvdb::Coord dim = bbox.dim();
    //info.width = dim.x();
    //info.height = dim.y();
    //info.depth = dim.z();

    //info.transform_3d
}
#endif

bool VDBImageLoader::load_metadata(const ImageDeviceFeatures &features, ImageMetaData &metadata)
{
#ifdef WITH_OPENVDB
  if (!grid) {
    return false;
  }

  /* Get number of channels from type. */
  NumChannelsOp op;
  if (!openvdb::grid_type_operation(grid, op)) {
    return false;
  }

  metadata.channels = op.num_channels;

  /* Set data type. */
#  ifdef WITH_NANOVDB
  if (features.has_nanovdb) {
    /* NanoVDB expects no inactive leaf nodes. */
#    if 0
    openvdb::FloatGrid &pruned_grid = *openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
    openvdb::tools::pruneInactive(pruned_grid.tree());
    nanogrid = nanovdb::openToNanoVDB(pruned_grid);
#    endif
    ToNanoOp op;
    op.precision = precision;
    if (!openvdb::grid_type_operation(grid, op)) {
      return false;
    }
    nanogrid = std::move(op.nanogrid);
  }
#  endif

  /* Set dimensions. */
  bbox = grid->evalActiveVoxelBoundingBox();
  if (bbox.empty()) {
    return false;
  }

  openvdb::Coord dim = bbox.dim();
  metadata.width = dim.x();
  metadata.height = dim.y();
  metadata.depth = dim.z();

#  ifdef WITH_NANOVDB
  if (nanogrid) {
    metadata.byte_size = nanogrid.size();
    if (metadata.channels == 1) {
      if (precision == 0) {
        metadata.type = IMAGE_DATA_TYPE_NANOVDB_FPN;
      }
      else if (precision == 16) {
        metadata.type = IMAGE_DATA_TYPE_NANOVDB_FP16;
      }
      else {
        metadata.type = IMAGE_DATA_TYPE_NANOVDB_FLOAT;
      }
    }
    else {
      metadata.type = IMAGE_DATA_TYPE_NANOVDB_FLOAT3;
    }
  }
  else
#  endif
  {
    if (metadata.channels == 1) {
      metadata.type = IMAGE_DATA_TYPE_FLOAT;
    }
    else {
      metadata.type = IMAGE_DATA_TYPE_FLOAT4;
    }
  }

  /* Set transform from object space to voxel index. */
  openvdb::math::Mat4f grid_matrix = grid->transform().baseMap()->getAffineMap()->getMat4();
  Transform index_to_object;
  for (int col = 0; col < 4; col++) {
    for (int row = 0; row < 3; row++) {
      index_to_object[row][col] = (float)grid_matrix[col][row];
    }
  }

  Transform texture_to_index;
#  ifdef WITH_NANOVDB
  if (nanogrid) {
    texture_to_index = transform_identity();
  }
  else
#  endif
  {
    openvdb::Coord min = bbox.min();
    texture_to_index = transform_translate(min.x(), min.y(), min.z()) *
                       transform_scale(dim.x(), dim.y(), dim.z());
  }

  metadata.transform_3d = transform_inverse(index_to_object * texture_to_index);
  metadata.use_transform_3d = true;

#  ifndef WITH_NANOVDB
  (void)features;
#  endif
  return true;
#else
  (void)metadata;
  (void)features;
  return false;
#endif
}

bool VDBImageLoader::load_pixels(const ImageMetaData & /*metadata*/,
                                 void *pixels,
                                 const size_t /*pixels_size*/,
                                 const bool /*associate_alpha*/)
{
#ifdef WITH_OPENVDB
#  ifdef WITH_NANOVDB
  if (nanogrid) {
    memcpy(pixels, nanogrid.data(), nanogrid.size());
  }
  else
#  endif
  {
    ToDenseOp op;
    op.pixels = pixels;
    op.bbox = bbox;
    openvdb::grid_type_operation(grid, op);
  }
  return true;
#else
  (void)pixels;
  return false;
#endif
}

string VDBImageLoader::name() const
{
  return grid_name;
}

bool VDBImageLoader::equals(const ImageLoader &other) const
{
#ifdef WITH_OPENVDB
  const VDBImageLoader &other_loader = (const VDBImageLoader &)other;
  return grid == other_loader.grid;
#else
  (void)other;
  return true;
#endif
}

void VDBImageLoader::cleanup()
{
//#ifdef WITH_OPENVDB
//  /* Free OpenVDB grid memory as soon as we can. */
//  grid.reset();
//#endif
//#ifdef WITH_NANOVDB
//  nanogrid.reset();
//#endif
}

bool VDBImageLoader::is_vdb_loader() const
{
  return true;
}

bool VDBImageLoader::is_simple_mesh() const
{
    return false;
}

void VDBImageLoader::get_bbox(int3& min_bbox, int3& max_bbox)
{
#ifdef WITH_OPENVDB  
    auto bbox = grid->evalActiveVoxelBoundingBox();
    auto grid_bbox_min = bbox.min();
    auto grid_bbox_max = bbox.max();

    min_bbox = make_int3(grid_bbox_min.x(), grid_bbox_min.y(), grid_bbox_min.z());
    max_bbox = make_int3(grid_bbox_max.x(), grid_bbox_max.y(), grid_bbox_max.z());
#endif    
}

float3 VDBImageLoader::index_to_world(float3 in)
{
#ifdef WITH_OPENVDB  
    openvdb::Vec3d p = grid->indexToWorld(openvdb::Vec3d(in[0], in[1], in[2]));
    return make_float3((float)p[0], (float)p[1], (float)p[2]);
#else
    return make_float3(0, 0, 0);
#endif    
}

#ifdef WITH_OPENVDB
openvdb::GridBase::ConstPtr VDBImageLoader::get_grid()
{
  return grid;
}
#endif

#ifdef WITH_NANOVDB
NanoVDBImageLoader::NanoVDBImageLoader(vector<char> &g)
    : nanogrid_data(std::move(g)), VDBImageLoader("")
{
}

NanoVDBImageLoader::~NanoVDBImageLoader()
{
}

bool NanoVDBImageLoader::load_metadata(const ImageDeviceFeatures& features, ImageMetaData& metadata)
{
    metadata.channels = (get_nanogrid()->gridType() == nanovdb::GridType::Float) ? 1 : 3; // TODO

    /* Set dimensions. */
    auto bbox = get_nanogrid()->worldBBox();
    if (bbox.empty()) {
        return false;
    }

    auto dim = bbox.dim();
    metadata.width = dim[0];
    metadata.height = dim[1];
    metadata.depth = dim[2];

    if (get_nanogrid()) {
        metadata.byte_size = nanogrid_data.size();
        if (metadata.channels == 1) {
            metadata.type = IMAGE_DATA_TYPE_NANOVDB_FLOAT;
        }
        else {
            metadata.type = IMAGE_DATA_TYPE_NANOVDB_FLOAT3;
        }
    }

    /* Set transform from object space to voxel index. */
    //matMult(mInvMatD, Vec3T(xyz[0] - mVecD[0], xyz[1] - mVecD[1], xyz[2] - mVecD[2]));
    const double* matD = get_nanogrid()->map().mMatD;
    const double* vecD = get_nanogrid()->map().mVecD;

    Transform index_to_object;
    //for (int col = 0; col < 4; col++) {
    //    for (int row = 0; row < 3; row++) {
    //        index_to_object[row][col] = (float)grid_matrix[col][row];
    //    }
    //}

    for (int i = 0; i < 3; ++i) {
        index_to_object[i].x = static_cast<float>(matD[i * 3 + 0]);
        index_to_object[i].y = static_cast<float>(matD[i * 3 + 1]);
        index_to_object[i].z = static_cast<float>(matD[i * 3 + 2]);
        index_to_object[i].w = static_cast<float>(vecD[i]);
    }

    //Transform texture_to_index;
    //if (get_nanogrid()) {
    //    texture_to_index = transform_identity();
    //}

    metadata.transform_3d = transform_inverse(index_to_object);
    metadata.use_transform_3d = true;

    return true;
}

bool NanoVDBImageLoader::load_pixels(const ImageMetaData&, void* pixels, const size_t, const bool)
{
    if (nanogrid_data.size() > 0) {
        memcpy(pixels, get_nanogrid(), nanogrid_data.size());
    }

    return true;
}

string NanoVDBImageLoader::name() const
{
    return get_nanogrid()->gridName();
}

bool NanoVDBImageLoader::equals(const ImageLoader& other) const
{
    const NanoVDBImageLoader& other_loader = (const NanoVDBImageLoader&)other;
    return get_nanogrid() == other_loader.get_nanogrid();
}

void NanoVDBImageLoader::cleanup()
{
}

bool NanoVDBImageLoader::is_vdb_loader() const
{
    return true;
}
bool NanoVDBImageLoader::is_simple_mesh() const
{
    return true;
}

void NanoVDBImageLoader::get_bbox(int3 &min_bbox, int3 &max_bbox)
{
    auto bbox = get_nanogrid()->indexBBox();

    auto grid_bbox_min = bbox.min();
    auto grid_bbox_max = bbox.max();

    min_bbox = make_int3(grid_bbox_min.x(), grid_bbox_min.y(), grid_bbox_min.z());
    max_bbox = make_int3(grid_bbox_max.x(), grid_bbox_max.y(), grid_bbox_max.z());
}

float3 NanoVDBImageLoader::index_to_world(float3 in)
{
    nanovdb::Vec3d p = get_nanogrid()->indexToWorld(nanovdb::Vec3d(in[0], in[1], in[2]));
    return make_float3((float)p[0], (float)p[1], (float)p[2]);
}

///////////////////// NanoVDBMultiResImageLoader
// format description of bin file:
// size_t : number of levels (aligned to 32 bytes)
// size_t : offset to grid1
// grid0 data (aligned to 32 bytes)
// size_t : offset to grid2
// grid1 data (aligned to 32 bytes)
// ...
NanoVDBMultiResImageLoader::NanoVDBMultiResImageLoader(vector<char>& g)
    : VDBImageLoader("")
{
    grids = std::move(g);

	size_t offset = 0;
	// read number of levels
	levels = *((size_t*)(grids.data() + offset));	
	// align to 32 bytes after num_levels
	offset = 32;

	// read grid offsets
	grid_offsets.resize(levels);
	for (int i = 0; i < levels; ++i) {
		// grid data starts at current offset (after optional offset field)
		if (i < levels - 1) {
			// read next grid offset (but we track positions sequentially)
			size_t next_offset = *((size_t*)(grids.data() + offset));
			grid_offsets[i] = offset + sizeof(size_t);
			// jump to next aligned position
			offset = next_offset - sizeof(size_t);
		}
		else {
			// last grid has no offset field
			grid_offsets[i] = offset;
		}
	}
}

NanoVDBMultiResImageLoader::~NanoVDBMultiResImageLoader()
{
}

bool NanoVDBMultiResImageLoader::load_metadata(const ImageDeviceFeatures& features, ImageMetaData& metadata)
{
    metadata.channels = (get_nanogrid(0)->gridType() == nanovdb::GridType::Float) ? 1 : 3; // TODO

    /* Set dimensions. */
	nanovdb::Vec3dBBox bbox = get_nanogrid(0)->worldBBox();
    //auto bbox = get_nanogrid(0)->worldBBox();
    //if (bbox.empty()) {
    //    return false;
    //}
	for (int i = 1; i < levels; ++i) {
		nanovdb::Vec3dBBox lbbox = get_nanogrid(i)->worldBBox();
		bbox.expand(lbbox);
	}

    auto dim = bbox.dim();
    metadata.width = dim[0];
    metadata.height = dim[1];
    metadata.depth = dim[2];

    metadata.byte_size = grids.size();
    if (metadata.channels == 1) {
        metadata.type = IMAGE_DATA_TYPE_NANOVDB_MULTIRES_FLOAT;
    }
    else {
        //TODO
		printf("NanoVDBMultiResImageLoader: only float type supported now.\n");
        //metadata.type = IMAGE_DATA_TYPE_NANOVDB_FLOAT3;
    }

    /* Set transform from object space to voxel index. */
    //matMult(mInvMatD, Vec3T(xyz[0] - mVecD[0], xyz[1] - mVecD[1], xyz[2] - mVecD[2]));
    const double* matD = get_nanogrid(0)->map().mMatD;
    const double* vecD = get_nanogrid(0)->map().mVecD;

    Transform index_to_object;
    //for (int col = 0; col < 4; col++) {
    //    for (int row = 0; row < 3; row++) {
    //        index_to_object[row][col] = (float)grid_matrix[col][row];
    //    }
    //}

    for (int i = 0; i < 3; ++i) {
        index_to_object[i].x = static_cast<float>(matD[i * 3 + 0]);
        index_to_object[i].y = static_cast<float>(matD[i * 3 + 1]);
        index_to_object[i].z = static_cast<float>(matD[i * 3 + 2]);
        index_to_object[i].w = static_cast<float>(vecD[i]);
    }

    //Transform texture_to_index;
    //if (get_nanogrid()) {
    //    texture_to_index = transform_identity();
    //}

    metadata.transform_3d = transform_inverse(index_to_object);
    metadata.use_transform_3d = false;

    return true;
}

bool NanoVDBMultiResImageLoader::load_pixels(const ImageMetaData&, void* pixels, const size_t, const bool)
{
    if (grids.size() > 0) {
        memcpy(pixels, grids.data(), grids.size());
    }

    return true;
}

string NanoVDBMultiResImageLoader::name() const
{
    return get_nanogrid(0)->gridName();
}

bool NanoVDBMultiResImageLoader::equals(const ImageLoader& other) const
{
    const NanoVDBMultiResImageLoader& other_loader = (const NanoVDBMultiResImageLoader&)other;
    return get_nanogrid(0) == other_loader.get_nanogrid(0);
}

void NanoVDBMultiResImageLoader::cleanup()
{
}

bool NanoVDBMultiResImageLoader::is_vdb_loader() const
{
    return true;
}
bool NanoVDBMultiResImageLoader::is_simple_mesh() const
{
    return true;
}

void NanoVDBMultiResImageLoader::get_bbox(int3 &min_bbox, int3 &max_bbox)
{
    nanovdb::CoordBBox bbox = get_nanogrid(0)->indexBBox();
    //auto bbox = get_nanogrid(0)->indexBBox();
	for (int i = 1; i < levels; ++i) {
		nanovdb::CoordBBox lbbox = get_nanogrid(i)->indexBBox();
		bbox.expand(lbbox);
	}

    auto grid_bbox_min = bbox.min();
    auto grid_bbox_max = bbox.max();

    min_bbox = make_int3(grid_bbox_min.x(), grid_bbox_min.y(), grid_bbox_min.z());
    max_bbox = make_int3(grid_bbox_max.x(), grid_bbox_max.y(), grid_bbox_max.z());
}

float3 NanoVDBMultiResImageLoader::index_to_world(float3 in)
{
    nanovdb::Vec3d p = get_nanogrid(0)->indexToWorld(nanovdb::Vec3d(in[0], in[1], in[2]));
    return make_float3((float)p[0], (float)p[1], (float)p[2]);
}

#endif

RAWImageLoader::RAWImageLoader(vector<char> &g, int dx, int dy, int dz, float sx, float sy, float sz, RAWImageLoaderType t, int c)
    : grid(std::move(g)), dimx(dx), dimy(dy), dimz(dz), scal_x(sx), scal_y(sy), scal_z(sz), raw_type(t), channels(c), VDBImageLoader("")
{
}

RAWImageLoader::~RAWImageLoader()
{
}

bool RAWImageLoader::load_metadata(const ImageDeviceFeatures& features, ImageMetaData& metadata)
{
    metadata.channels = channels;
    metadata.width = dimx;
    metadata.height = dimy;
    metadata.depth = dimz;

    metadata.byte_size = grid.size();

    if (metadata.channels == 4) {
        switch (raw_type) {
        case RAWImageLoader::eRawFloat:
            metadata.type = IMAGE_DATA_TYPE_FLOAT4;
            break;

        case RAWImageLoader::eRawByte:
            metadata.type = IMAGE_DATA_TYPE_BYTE4;
            break;

        case RAWImageLoader::eRawHalf:
            metadata.type = IMAGE_DATA_TYPE_HALF4;
            break;

        case RAWImageLoader::eRawUShort:
            metadata.type = IMAGE_DATA_TYPE_USHORT4;
            break;
        }
    }
    else if (metadata.channels == 1) {
        switch (raw_type) {
            case RAWImageLoader::eRawFloat:
                metadata.type = IMAGE_DATA_TYPE_FLOAT;
                break;

            case RAWImageLoader::eRawByte:
                metadata.type = IMAGE_DATA_TYPE_BYTE;
                break;

            case RAWImageLoader::eRawHalf:
                metadata.type = IMAGE_DATA_TYPE_HALF;
                break;

            case RAWImageLoader::eRawUShort:
                metadata.type = IMAGE_DATA_TYPE_USHORT;
                break;
        }
    }    

    metadata.transform_3d = ccl::transform_scale(ccl::make_float3(1.0f / (float)dimx, 1.0f / (float)dimy, 1.0f / (float)dimz));
    //metadata.transform_3d = ccl::transform_scale(ccl::make_float3(scal_x / (float)dimx, scal_y / (float)dimy, scal_z / (float)dimz));
    metadata.use_transform_3d = true;

    return true;
}

bool RAWImageLoader::load_pixels(const ImageMetaData&, void* pixels, const size_t, const bool)
{
    if (grid.size() > 0) {
        memcpy(pixels, grid.data(), grid.size());
    }

    return true;
}

string RAWImageLoader::name() const
{
    return "RAW Volume";
}

bool RAWImageLoader::equals(const ImageLoader& other) const
{
    const RAWImageLoader& other_loader = (const RAWImageLoader&)other;

    if (dimx != other_loader.dimx || dimy != other_loader.dimy || dimz != other_loader.dimz)
        return false;

    if (channels != other_loader.channels || raw_type != other_loader.raw_type)
        return false;

    if (grid.size() == 0 || grid.size() != other_loader.grid.size())
        return false;

    return !memcmp(grid.data(), other_loader.grid.data(), grid.size());
}

void RAWImageLoader::cleanup()
{
    //grid.clear();
}

bool RAWImageLoader::is_vdb_loader() const
{
    return true;
}
bool RAWImageLoader::is_simple_mesh() const
{
    return true;
}

void RAWImageLoader::get_bbox(int3& min_bbox, int3& max_bbox)
{
    min_bbox = make_int3(0, 0, 0);
    max_bbox = make_int3(dimx - 1, dimy - 1, dimz - 1);

    //min_bbox = make_int3(-dimx / 2, -dimy / 2, -dimz / 2);
    //max_bbox = make_int3(dimx / 2 - 1, dimy / 2 - 1, dimz / 2 - 1);
}

float3 RAWImageLoader::index_to_world(float3 in)
{    
    return make_float3((float)in[0], (float)in[1], (float)in[2]);
}

CCL_NAMESPACE_END
