/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/image_vdb.h"

#include "util/log.h"
#include "util/nanovdb.h"
#include "util/openvdb.h"
#include "util/texture.h"

#ifdef WITH_OPENVDB
#  include <openvdb/tools/Dense.h>
#endif

CCL_NAMESPACE_BEGIN

#ifdef WITH_OPENVDB
VDBImageLoader::VDBImageLoader(openvdb::GridBase::ConstPtr grid_,
                               const string &grid_name,
                               const float clipping)
    : grid_name(grid_name), clipping(clipping), grid(grid_)
{
}
#endif

VDBImageLoader::VDBImageLoader(const string &grid_name, const float clipping)
    : grid_name(grid_name), clipping(clipping)
{
    if(!grid_name.empty())
        printf("VDBImageLoader: grid_name: %s\n", grid_name.c_str());
}

VDBImageLoader::~VDBImageLoader() = default;

//#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)
//
//nanovdb::GridHandle<> VDBImageLoader::convert(openvdb::GridBase::ConstPtr g, int p)
//{
//    ToNanoOp op;
//    op.precision = p;
//    if (!openvdb::grid_type_operation(g, op)) {
//        return nanovdb::GridHandle<>();
//    }
//
//    return std::move(op.nanogrid);
//}
//
//void VDBImageLoader::get_texture_info(nanovdb::NanoGrid<float>* nanogrid, size_t ng_size, TextureInfo& info)
//{
//    //nanovdb::NanoGrid<float>* nanogrid = (nanovdb::NanoGrid<float>*) ng.data();
//
//    int data_type = (nanogrid->gridType() == nanovdb::GridType::Float) ? 1 : 3; // TODO
//
//    ///* Set dimensions. */
//    //auto bbox = nanogrid->worldBBox();
//    //if (bbox.empty()) {
//    //    return;
//    //}
//
//    //auto dim = bbox.dim();
//    //info.width = dim[0];
//    //info.height = dim[1];
//    //info.depth = dim[2];
//
//    info.width = ng_size;
//    info.height = 0;
//    info.depth = 0;    
//
//    if (nanogrid) {
//        //metadata.byte_size = nanogrid_data.size();
//        if (data_type == 1) {
//            info.data_type = IMAGE_DATA_TYPE_NANOVDB_FLOAT;
//        }
//        else {
//            info.data_type = IMAGE_DATA_TYPE_NANOVDB_FLOAT3;
//        }
//    }
//
//    /* Set transform from object space to voxel index. */
//    //matMult(mInvMatD, Vec3T(xyz[0] - mVecD[0], xyz[1] - mVecD[1], xyz[2] - mVecD[2]));
//    const double* matD = nanogrid->map().mMatD;
//    const double* vecD = nanogrid->map().mVecD;
//
//    Transform index_to_object;
//    //for (int col = 0; col < 4; col++) {
//    //    for (int row = 0; row < 3; row++) {
//    //        index_to_object[row][col] = (float)grid_matrix[col][row];
//    //    }
//    //}
//
//    for (int i = 0; i < 3; ++i) {
//        index_to_object[i].x = static_cast<float>(matD[i * 3 + 0]);
//        index_to_object[i].y = static_cast<float>(matD[i * 3 + 1]);
//        index_to_object[i].z = static_cast<float>(matD[i * 3 + 2]);
//        index_to_object[i].w = static_cast<float>(vecD[i]);
//    }
//
//    //Transform texture_to_index;
//    //if (get_nanogrid()) {
//    //    texture_to_index = transform_identity();
//    //}
//
//    info.transform_3d = transform_inverse(index_to_object);
//    info.use_transform_3d = true;
//
//    ///* Set dimensions. */
//    //openvdb::CoordBBox bbox = g->evalActiveVoxelBoundingBox();
//    //if (bbox.empty()) {
//    //    return;
//    //}
//
//    //openvdb::Coord dim = bbox.dim();
//    //info.width = dim.x();
//    //info.height = dim.y();
//    //info.depth = dim.z();
//
//    //info.transform_3d
//}
//#endif

bool VDBImageLoader::load_metadata(const ImageDeviceFeatures &features, ImageMetaData &metadata)
{
  if (!features.has_nanovdb) {
    return false;
  }

#ifdef WITH_NANOVDB
  load_grid();

  if (!grid) {
    return false;
  }

  /* Convert to the few float types that we know. */
  grid = openvdb_convert_to_known_type(grid);
  if (!grid) {
    return false;
  }

  /* Get number of channels from type. */
  metadata.channels = openvdb_num_channels(grid);

  /* Convert OpenVDB to NanoVDB grid. */
  nanogrid = openvdb_to_nanovdb(grid, precision, clipping);
  if (!nanogrid) {
    grid.reset();
    return false;
  }

  /* Set dimensions. */
  bbox = grid->evalActiveVoxelBoundingBox();
  if (bbox.empty()) {
    metadata.type = IMAGE_DATA_TYPE_NANOVDB_EMPTY;
    metadata.byte_size = 1;
    grid.reset();
    return true;
  }

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
  else if (metadata.channels == 3) {
    metadata.type = IMAGE_DATA_TYPE_NANOVDB_FLOAT3;
  }
  else if (metadata.channels == 4) {
    metadata.type = IMAGE_DATA_TYPE_NANOVDB_FLOAT4;
  }
  else {
    grid.reset();
    return false;
  }

  metadata.byte_size = nanogrid.size();

  /* Set transform from object space to voxel index. */
  openvdb::math::Mat4f grid_matrix = grid->transform().baseMap()->getAffineMap()->getMat4();
  Transform index_to_object;
  for (int col = 0; col < 4; col++) {
    for (int row = 0; row < 3; row++) {
      index_to_object[row][col] = (float)grid_matrix[col][row];
    }
  }

  metadata.transform_3d = transform_inverse(index_to_object);
  metadata.use_transform_3d = true;

  /* Only NanoGrid needed now, free OpenVDB grid. */
  grid.reset();

  return true;
#else
  (void)metadata;
  return false;
#endif
}

bool VDBImageLoader::load_pixels(const ImageMetaData &metadata,
                                 void *pixels,
                                 const size_t /*pixels_size*/,
                                 const bool /*associate_alpha*/)
{
#ifdef WITH_NANOVDB
  if (metadata.type == IMAGE_DATA_TYPE_NANOVDB_EMPTY) {
    memset(pixels, 0, metadata.byte_size);
    return true;
  }
  if (nanogrid) {
    memcpy(pixels, nanogrid.data(), nanogrid.size());
    return true;
  }
#else
  (void)metadata;
  (void)pixels;
#endif

  return false;
}

string VDBImageLoader::name() const
{
  return grid_name;
}

bool VDBImageLoader::equals(const ImageLoader &other) const
{
#ifdef WITH_OPENVDB
  const VDBImageLoader &other_loader = (const VDBImageLoader &)other;
  return grid && grid == other_loader.grid;
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

template<typename GridType>
openvdb::GridBase::ConstPtr create_grid(const float *voxels,
                                        const size_t width,
                                        const size_t height,
                                        const size_t depth,
                                        Transform transform_3d,
                                        const float clipping)
{
  using ValueType = typename GridType::ValueType;
  openvdb::GridBase::ConstPtr grid;

  const openvdb::CoordBBox dense_bbox(0, 0, 0, width - 1, height - 1, depth - 1);

  typename GridType::Ptr sparse = GridType::create(ValueType(0.0f));
  if (dense_bbox.empty()) {
    return sparse;
  }

  const openvdb::tools::Dense<const ValueType, openvdb::tools::MemoryLayout::LayoutXYZ> dense(
      dense_bbox, reinterpret_cast<const ValueType *>(voxels));

  openvdb::tools::copyFromDense(dense, *sparse, ValueType(clipping));

  /* Compute index to world matrix. */
  const float3 voxel_size = make_float3(1.0f / width, 1.0f / height, 1.0f / depth);

  transform_3d = transform_inverse(transform_3d);

  const openvdb::Mat4R index_to_world_mat((double)(voxel_size.x * transform_3d[0][0]),
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          (double)(voxel_size.y * transform_3d[1][1]),
                                          0.0,
                                          0.0,
                                          0.0,
                                          0.0,
                                          (double)(voxel_size.z * transform_3d[2][2]),
                                          0.0,
                                          (double)transform_3d[0][3] + voxel_size.x,
                                          (double)transform_3d[1][3] + voxel_size.y,
                                          (double)transform_3d[2][3] + voxel_size.z,
                                          1.0);

  const openvdb::math::Transform::Ptr index_to_world_tfm =
      openvdb::math::Transform::createLinearTransform(index_to_world_mat);

  sparse->setTransform(index_to_world_tfm);

  return sparse;
}
#endif

void VDBImageLoader::grid_from_dense_voxels(const size_t width,
                                            const size_t height,
                                            const size_t depth,
                                            const int channels,
                                            const float *voxels,
                                            Transform transform_3d)
{
#ifdef WITH_OPENVDB
  /* TODO: Create NanoVDB grid directly? */
  if (channels == 1) {
    grid = create_grid<openvdb::FloatGrid>(voxels, width, height, depth, transform_3d, clipping);
  }
  else if (channels == 3) {
    grid = create_grid<openvdb::Vec3fGrid>(voxels, width, height, depth, transform_3d, clipping);
  }
  else if (channels == 4) {
    grid = create_grid<openvdb::Vec4fGrid>(voxels, width, height, depth, transform_3d, clipping);
  }

  /* Clipping already applied, no need to do it again. */
  clipping = 0.0f;
#else
  (void)width;
  (void)height;
  (void)depth;
  (void)channels;
  (void)voxels;
  (void)transform_3d;
#endif
}


#ifdef WITH_NANOVDB
NanoVDBImageLoader::NanoVDBImageLoader(vector<char> &g)
    : nanogrid_data(std::move(g)), VDBImageLoader("")
{
    printf("NanoVDBImageLoader: size in bytes: %lld\n", nanogrid_data.size());
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
    //metadata.depth = dim[2];

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

	printf("NanoVDBMultiResImageLoader: number of levels: %zu\n", levels);

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
    //metadata.depth = dim[2];

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
    printf("RAWImageLoader: size in bytes: %lld\n", grid.size());
}

RAWImageLoader::~RAWImageLoader()
{
}

bool RAWImageLoader::load_metadata(const ImageDeviceFeatures& features, ImageMetaData& metadata)
{
    metadata.channels = channels;
    metadata.width = dimx;
    metadata.height = dimy;
    //metadata.depth = dimz;

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
