/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "scene/image_vdb.h"

#include "util/image_metadata.h"
#include "util/log.h"
#include "util/nanovdb.h"
#include "util/openvdb.h"
#include "util/types_image.h"

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

bool VDBImageLoader::load_metadata(ImageMetaData &metadata)
{
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
    metadata.nanovdb_byte_size = 1;
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

#  if NANOVDB_MAJOR_VERSION_NUMBER > 32 || \
      (NANOVDB_MAJOR_VERSION_NUMBER == 32 && NANOVDB_MINOR_VERSION_NUMBER >= 9)
  /* size() was deprecated in this version. */
  metadata.nanovdb_byte_size = nanogrid.bufferSize();
#  else
  metadata.nanovdb_byte_size = nanogrid.size();
#  endif

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

bool VDBImageLoader::load_pixels(const ImageMetaData &metadata, void *pixels)
{
#ifdef WITH_NANOVDB
  if (metadata.type == IMAGE_DATA_TYPE_NANOVDB_EMPTY) {
    memset(pixels, 0, metadata.nanovdb_byte_size);
    return true;
  }
  if (nanogrid) {
    memcpy(pixels, nanogrid.data(), metadata.nanovdb_byte_size);
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

bool NanoVDBImageLoader::load_metadata(ImageMetaData& metadata)
{
    metadata.channels = (get_nanogrid()->gridType() == nanovdb::GridType::Float) ? 1 : 3; // TODO

    /* Set dimensions. */
    auto bbox = get_nanogrid()->indexBBox();//worldBBox();
    if (bbox.empty()) {
        return false;
    }

    auto dim = bbox.dim();
    metadata.width = dim[0];
    metadata.height = dim[1];
    //metadata.depth = dim[2];

    if (get_nanogrid()) {
        metadata.nanovdb_byte_size = nanogrid_data.size();
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

bool NanoVDBImageLoader::load_pixels(const ImageMetaData&, void* pixels)
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
NanoVDBMultiResImageLoader::NanoVDBMultiResImageLoader(vector<char>& g, NanoVDBMultiResImageLoaderType t)
    : VDBImageLoader(""), type(t)
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

    largest_grid_id = 0;
    size_t max_resolution = 0;  
    //find largest grid id for metadata
    for (int i = 0; i < levels; ++i) {
		nanovdb::CoordBBox cbbox = get_nanogrid(i)->indexBBox();
        nanovdb::Coord dim = cbbox.dim();
        size_t resolution = std::max({dim.x(), dim.y(), dim.z()});
        if (max_resolution == 0 || resolution > max_resolution) {
            max_resolution = resolution;
            largest_grid_id = i;
        }   
	}
    printf("NanoVDBMultiResImageLoader: largest grid id: %zu, resolution: %zu\n", largest_grid_id, max_resolution);
}

NanoVDBMultiResImageLoader::~NanoVDBMultiResImageLoader()
{
}

bool NanoVDBMultiResImageLoader::load_metadata(ImageMetaData& metadata)
{
    metadata.channels = (get_nanogrid(0)->gridType() == nanovdb::GridType::Float) ? 1 : 3; // TODO

    /* Set dimensions. */
    nanovdb::CoordBBox bbox = get_nanogrid(0)->indexBBox();  // worldBBox();
    //auto bbox = get_nanogrid(0)->worldBBox();
    //if (bbox.empty()) {
    //    return false;
    //}
#if 1
  
    for (int i = 0; i < levels; ++i) {
		nanovdb::CoordBBox cbbox = get_nanogrid(i)->indexBBox();
        printf("NanoVDBMultiResImageLoader (indexBBox): level %d, index bbox min: (%d, %d, %d), max: (%d, %d, %d)\n", i,
            cbbox.min().x(), cbbox.min().y(), cbbox.min().z(),
            cbbox.max().x(), cbbox.max().y(), cbbox.max().z());		
	}    
	
    for (int i = 0; i < levels; ++i) {
		nanovdb::Vec3dBBox lbbox = get_nanogrid(i)->worldBBox();
		printf("NanoVDBMultiResImageLoader (worldBBox): level %d, bbox min: (%f, %f, %f), max: (%f, %f, %f)\n", i,
            lbbox.min()[0], lbbox.min()[1], lbbox.min()[2],
            lbbox.max()[0], lbbox.max()[1], lbbox.max()[2]);
	}
#endif    

	for (int i = 1; i < levels; ++i) {
        nanovdb::CoordBBox lbbox = get_nanogrid(i)->indexBBox();  // worldBBox();
		bbox.expand(lbbox);
	}

    auto dim = bbox.dim();
    metadata.width = dim[0];
    metadata.height = dim[1];
    //metadata.depth = dim[2];

    metadata.nanovdb_byte_size = grids.size();
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
    const double* matD = get_nanogrid(largest_grid_id)->map().mMatD;
    const double* vecD = get_nanogrid(largest_grid_id)->map().mVecD;

    // for (int i = 0; i < levels; ++i) {
    //     const double* matD_ = get_nanogrid(i)->map().mMatD;
    //     const double* vecD_ = get_nanogrid(i)->map().mVecD;

    //     printf("NanoVDBMultiResImageLoader: level %d, matD: [%f, %f, %f, %f, %f, %f, %f, %f, %f], vecD: [%f, %f, %f]\n", i,
    //         matD_[0], matD_[1], matD_[2], matD_[3], matD_[4], matD_[5], matD_[6], matD_[7], matD_[8],
    //         vecD_[0], vecD_[1], vecD_[2]);
    // }

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

bool NanoVDBMultiResImageLoader::load_pixels(const ImageMetaData&, void* pixels)
{
    if (grids.size() > 0) {
        memcpy(pixels, grids.data(), grids.size());
    }

    return true;
}

string NanoVDBMultiResImageLoader::name() const
{
    return get_nanogrid(largest_grid_id)->gridName();
}

bool NanoVDBMultiResImageLoader::equals(const ImageLoader& other) const
{
    const NanoVDBMultiResImageLoader& other_loader = (const NanoVDBMultiResImageLoader&)other;
    return get_nanogrid(largest_grid_id) == other_loader.get_nanogrid(other_loader.largest_grid_id);
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
    nanovdb::CoordBBox bbox = get_nanogrid(largest_grid_id)->indexBBox();
    //auto bbox = get_nanogrid(largest_grid_id)->indexBBox();
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
    nanovdb::Vec3d p = get_nanogrid(largest_grid_id)->indexToWorld(nanovdb::Vec3d(in[0], in[1], in[2]));
    return make_float3((float)p[0], (float)p[1], (float)p[2]);
}

///////////////////// NanoVDBDerivatesImageLoader
// New derivative bundle format:
// FileHeader (64 bytes)
// LevelHeader[levelCount]
// GridHeader[gridCount]
// NanoVDB grid payloads (each 32-byte aligned)
NanoVDBDerivatesImageLoader::NanoVDBDerivatesImageLoader(vector<char>& g)
    : VDBImageLoader(""), finest_level_id(0)
{
    bundle_data = std::move(g);

    // Read and validate file header
    if (bundle_data.size() < sizeof(DerivFileHeader)) {
        printf("NanoVDBDerivatesImageLoader: Invalid bundle size\n");
        return;
    }

    const DerivFileHeader* fh = get_file_header();
    
    // Validate magic number
    if (fh->magic != 0x4E56444D) {
        printf("NanoVDBDerivatesImageLoader: Invalid magic number: 0x%08X\n", fh->magic);
        return;
    }

    // Cache file header
    file_header = *fh;

    printf("NanoVDBDerivatesImageLoader: Loaded derivative bundle\n");
    printf("  Version: %u\n", file_header.version);
    printf("  Levels: %u\n", file_header.levelCount);
    printf("  Grids: %u\n", file_header.gridCount);
    printf("  Total size: %llu bytes\n", file_header.totalFileSize);

    // Find finest level (typically level 0, but we check resolution)
    const DerivLevelHeader* level_table = get_level_table();
    const DerivGridHeader* grid_table = get_grid_table();
    
    finest_level_id = 0;
    uint32_t max_resolution = 0;

    for (uint32_t i = 0; i < file_header.levelCount; ++i) {
        const DerivLevelHeader& lh = level_table[i];
        
        if (lh.firstGridIndex < file_header.gridCount) {
            const DerivGridHeader& gh = grid_table[lh.firstGridIndex];
            uint32_t resolution = std::max({gh.dims[0], gh.dims[1], gh.dims[2]});
            
            printf("  Level %u: %u derivatives, resolution %u\n", 
                   (unsigned)lh.levelIndex, (unsigned)lh.derivativeCount, (unsigned)resolution);
            
            if (resolution > max_resolution) {
                max_resolution = resolution;
                finest_level_id = i;
            }
        }
    }

    printf("  Finest level: %u (resolution %u)\n", (unsigned)finest_level_id, max_resolution);

    // Print transform information for each grid
    printf("  Grid transforms:\n");
    for (uint32_t grid_idx = 0; grid_idx < file_header.gridCount; ++grid_idx) {
        nanovdb::NanoGrid<float>* grid = get_grid(grid_idx);
        if (grid) {
            const double* matD = grid->map().mMatD;
            const double* vecD = grid->map().mVecD;
            
            printf("    Grid %u: matD=[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f], vecD=[%.6f, %.6f, %.6f]\n",
                   grid_idx,
                   matD[0], matD[1], matD[2], matD[3], matD[4], matD[5], matD[6], matD[7], matD[8],
                   vecD[0], vecD[1], vecD[2]);
        }
    }

    // Print active voxels for each grid (limited to first 100 per grid)
    printf("  Active voxels (sample):\n");
    for (uint32_t grid_idx = 0; grid_idx < file_header.gridCount; ++grid_idx) {
        nanovdb::NanoGrid<float>* grid = get_grid(grid_idx);
        if (grid) {
            printf("    Grid %u active voxels:\n", grid_idx);
            int count = 0;
            int max_print = 100;
            
            // Loop over child nodes of the root node
            for (auto it2 = grid->tree().root().cbeginChild(); it2 && count < max_print; ++it2) {
                // Loop over child nodes of the upper internal node
                for (auto it1 = it2->cbeginChild(); it1 && count < max_print; ++it1) {
                    // Loop over child nodes of the lower internal node
                    for (auto it0 = it1->cbeginChild(); it0 && count < max_print; ++it0) {
                        // Loop over active values
                        for (auto it = it0->cbeginValueOn(); it && count < max_print; ++it) {
                            float value = *it;
                            nanovdb::Coord coord = it.getCoord();
                            printf("      (%d, %d, %d): %.6f\n", coord[0], coord[1], coord[2], value);
                            count++;
                        }
                    }
                }
            }
            if (count >= max_print) {
                printf("      ... (limited to %d voxels)\n", max_print);
            }
        }
    }
}

NanoVDBDerivatesImageLoader::~NanoVDBDerivatesImageLoader()
{
}

bool NanoVDBDerivatesImageLoader::load_metadata(ImageMetaData& metadata)
{
    if (file_header.magic != 0x4E56444D || file_header.gridCount == 0) {
        return false;
    }

    // All derivative grids are float type
    metadata.channels = 1;
    metadata.type = IMAGE_DATA_TYPE_NANOVDB_DERIVATES;
    metadata.nanovdb_byte_size = bundle_data.size();

    // Get bounding box from finest level
    const DerivLevelHeader* level_table = get_level_table();
    const DerivGridHeader* grid_table = get_grid_table();
    
    const DerivLevelHeader& finest_level = level_table[finest_level_id];
    
    if (finest_level.firstGridIndex >= file_header.gridCount) {
        return false;
    }

    // Use first grid of finest level for metadata
    const DerivGridHeader& first_grid = grid_table[finest_level.firstGridIndex];
    nanovdb::NanoGrid<float>* grid = get_grid(finest_level.firstGridIndex);

    if (!grid) {
        return false;
    }

    // Set dimensions from world bounding box
    auto bbox = grid->indexBBox();  // worldBBox();
    auto dim = bbox.dim();
    metadata.width = dim[0];
    metadata.height = dim[1];

    // Set transform from object space to voxel index
    const double* matD = grid->map().mMatD;
    const double* vecD = grid->map().mVecD;

    Transform index_to_object;
    for (int i = 0; i < 3; ++i) {
        index_to_object[i].x = static_cast<float>(matD[i * 3 + 0]);
        index_to_object[i].y = static_cast<float>(matD[i * 3 + 1]);
        index_to_object[i].z = static_cast<float>(matD[i * 3 + 2]);
        index_to_object[i].w = static_cast<float>(vecD[i]);
    }

    metadata.transform_3d = transform_inverse(index_to_object);
    metadata.use_transform_3d = false;

    return true;
}

bool NanoVDBDerivatesImageLoader::load_pixels(const ImageMetaData&, void* pixels)
{
    if (bundle_data.size() > 0) {
        memcpy(pixels, bundle_data.data(), bundle_data.size());
    }

    return true;
}

string NanoVDBDerivatesImageLoader::name() const
{
    if (file_header.gridCount > 0) {
        // Return a descriptive name based on the bundle
        return string_printf("DerivativeBundle_%uL_%uG", 
                           file_header.levelCount, 
                           file_header.gridCount);
    }
    return "DerivativeBundle";
}

bool NanoVDBDerivatesImageLoader::equals(const ImageLoader& other) const
{
    const NanoVDBDerivatesImageLoader& other_loader = 
        (const NanoVDBDerivatesImageLoader&)other;
    
    if (bundle_data.size() != other_loader.bundle_data.size()) {
        return false;
    }
    
    return !memcmp(bundle_data.data(), 
                   other_loader.bundle_data.data(), 
                   bundle_data.size());
}

void NanoVDBDerivatesImageLoader::cleanup()
{
}

bool NanoVDBDerivatesImageLoader::is_vdb_loader() const
{
    return true;
}

bool NanoVDBDerivatesImageLoader::is_simple_mesh() const
{
    return true;
}

void NanoVDBDerivatesImageLoader::get_bbox(int3 &min_bbox, int3 &max_bbox)
{
    if (file_header.gridCount == 0) {
        min_bbox = max_bbox = make_int3(0, 0, 0);
        return;
    }

    const DerivLevelHeader* level_table = get_level_table();
    const DerivGridHeader* grid_table = get_grid_table();
    
    // Use finest level's first grid as reference coordinate system
    const DerivLevelHeader& finest_level = level_table[finest_level_id];
    if (finest_level.firstGridIndex >= file_header.gridCount) {
        min_bbox = max_bbox = make_int3(0, 0, 0);
        return;
    }
    
    nanovdb::NanoGrid<float>* reference_grid = get_grid(finest_level.firstGridIndex);
    if (!reference_grid) {
        min_bbox = max_bbox = make_int3(0, 0, 0);
        return;
    }
    
    nanovdb::CoordBBox unified_bbox;
    bool first = true;

    // Convert all grids' world bbox to reference grid's index space and union them
    for (uint32_t level_idx = 0; level_idx < file_header.levelCount; ++level_idx) {
        const DerivLevelHeader& lh = level_table[level_idx];
        
        // Iterate over ALL grids in this level (including all derivatives)
        for (uint32_t deriv_idx = 0; deriv_idx < lh.derivativeCount; ++deriv_idx) {
            uint32_t grid_idx = lh.firstGridIndex + deriv_idx;
            
            if (grid_idx < file_header.gridCount) {
                nanovdb::NanoGrid<float>* grid = get_grid(grid_idx);
                if (grid) {
                    // Get world bbox of this grid
                    nanovdb::Vec3dBBox world_bbox = grid->worldBBox();
                    
                    // Convert world bbox corners to reference grid's index space
                    // Need to check all 8 corners since transform might have rotations/scales
                    nanovdb::Vec3d wmin = world_bbox.min();
                    nanovdb::Vec3d wmax = world_bbox.max();
                    
                    nanovdb::CoordBBox index_bbox;
                    for (int i = 0; i < 8; ++i) {
                        nanovdb::Vec3d corner(
                            (i & 1) ? wmax[0] : wmin[0],
                            (i & 2) ? wmax[1] : wmin[1],
                            (i & 4) ? wmax[2] : wmin[2]
                        );
                        nanovdb::Vec3d index_pos = reference_grid->worldToIndex(corner);
                        nanovdb::Coord index_coord(
                            std::floor(index_pos[0]),
                            std::floor(index_pos[1]),
                            std::floor(index_pos[2])
                        );
                        
                        if (i == 0 && first) {
                            index_bbox = nanovdb::CoordBBox(index_coord, index_coord);
                            first = false;
                        } else {
                            index_bbox.expand(index_coord);
                        }
                    }
                    
                    if (!first) {
                        unified_bbox.expand(index_bbox);
                    }
                }
            }
        }
    }

    auto grid_bbox_min = unified_bbox.min();
    auto grid_bbox_max = unified_bbox.max();

    min_bbox = make_int3(grid_bbox_min.x(), grid_bbox_min.y(), grid_bbox_min.z());
    max_bbox = make_int3(grid_bbox_max.x(), grid_bbox_max.y(), grid_bbox_max.z());
}

float3 NanoVDBDerivatesImageLoader::index_to_world(float3 in)
{
    if (file_header.gridCount == 0) {
        return make_float3(0, 0, 0);
    }

    // Use finest level grid for coordinate transformation
    const DerivLevelHeader* level_table = get_level_table();
    const DerivLevelHeader& finest_level = level_table[finest_level_id];
    
    if (finest_level.firstGridIndex < file_header.gridCount) {
        nanovdb::NanoGrid<float>* grid = get_grid(finest_level.firstGridIndex);
        if (grid) {
            nanovdb::Vec3d p = grid->indexToWorld(nanovdb::Vec3d(in[0], in[1], in[2]));
            
            // Get voxel size from grid transform
            const double* matD = grid->map().mMatD;
            nanovdb::Vec3d voxel_size(matD[0], matD[4], matD[8]);
            
            // Offset by half voxel size
            //p[0] -= voxel_size[0] * 0.5;
            //p[1] -= voxel_size[1] * 0.5;
            //p[2] -= voxel_size[2] * 0.5;
            
            return make_float3((float)p[0], (float)p[1], (float)p[2]);
        }
    }

    return make_float3(0, 0, 0);
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

bool RAWImageLoader::load_metadata(ImageMetaData& metadata)
{
    metadata.channels = channels;
    metadata.width = dimx;
    metadata.height = dimy;
    //metadata.depth = dimz;

    metadata.nanovdb_byte_size = grid.size();

    if (metadata.channels == 3) {
        switch (raw_type) {
        case RAWImageLoader::eRawFloat:
            metadata.type = IMAGE_DATA_TYPE_RAW3D_FLOAT3;
            break;
        }
    }
    else if (metadata.channels == 1) {
        switch (raw_type) {
            case RAWImageLoader::eRawFloat:
                metadata.type = IMAGE_DATA_TYPE_RAW3D_FLOAT;
                break;
        }
    }    

    metadata.transform_3d = ccl::transform_scale(ccl::make_float3(1.0f / (float)dimx, 1.0f / (float)dimy, 1.0f / (float)dimz));
    //metadata.transform_3d = ccl::transform_scale(ccl::make_float3(scal_x / (float)dimx, scal_y / (float)dimy, scal_z / (float)dimz));
    metadata.use_transform_3d = true;

    return true;
}

bool RAWImageLoader::load_pixels(const ImageMetaData&, void* pixels)
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
