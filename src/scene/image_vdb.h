/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_OPENVDB
#  include <openvdb/openvdb.h>
#endif
#ifdef WITH_NANOVDB
#  include <nanovdb/NanoVDB.h>
#  if NANOVDB_MAJOR_VERSION_NUMBER > 32 || \
      (NANOVDB_MAJOR_VERSION_NUMBER == 32 && NANOVDB_MINOR_VERSION_NUMBER >= 7)
#    include <nanovdb/GridHandle.h>
#  else
#    include <nanovdb/util/GridHandle.h>
#  endif
#endif

#include "scene/image_loader.h"

#include "util/transform.h"

CCL_NAMESPACE_BEGIN

class VDBImageLoader : public ImageLoader {
 public:
#ifdef WITH_OPENVDB
  VDBImageLoader(openvdb::GridBase::ConstPtr grid_,
                 const string &grid_name,
                 const float clipping = 0.001f);
#endif
  VDBImageLoader(const string &grid_name, const float clipping = 0.001f);
  ~VDBImageLoader() override;

  bool load_metadata(ImageMetaData &metadata) override;

  bool load_pixels(const ImageMetaData &metadata, void *pixels) override;

  virtual string name() const override;

  virtual bool equals(const ImageLoader &other) const override;

  virtual void cleanup() override;

  virtual bool is_vdb_loader() const override;

  virtual bool is_simple_mesh() const;

  virtual void get_bbox(int3& min_bbox, int3& max_bbox);

  virtual float3 index_to_world(float3 in);

#ifdef WITH_OPENVDB
  openvdb::GridBase::ConstPtr get_grid();
#endif

//#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)
//  static nanovdb::GridHandle<> convert(openvdb::GridBase::ConstPtr g, int precision = 32);
//  static void get_texture_info(nanovdb::NanoGrid<float>* ng, size_t ng_size, TextureInfo &info);
//#endif

 protected:
  virtual void load_grid() {}

  void grid_from_dense_voxels(const size_t width,
                              const size_t height,
                              const size_t depth,
                              const int channels,
                              const float *voxels,
                              Transform transform_3d);

  string grid_name;
  float clipping = 0.001f;
#ifdef WITH_OPENVDB
  openvdb::GridBase::ConstPtr grid;
  openvdb::CoordBBox bbox;
#endif
#ifdef WITH_NANOVDB
  nanovdb::GridHandle<> nanogrid;
  int precision = 32;
#endif
};

#ifdef WITH_NANOVDB
class NanoVDBImageLoader : public VDBImageLoader {
public:
    NanoVDBImageLoader(vector<char> &g);
    ~NanoVDBImageLoader();

    virtual bool load_metadata(const ImageDeviceFeatures& features,
        ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata,
        void* pixels,
        const size_t pixels_size,
        const bool associate_alpha) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3& min_bbox, int3& max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
    vector<char> nanogrid_data;
    nanovdb::NanoGrid<float>* get_nanogrid() const {
        return (nanovdb::NanoGrid<float>*) nanogrid_data.data();
    }

};

class NanoVDBMultiResImageLoader : public VDBImageLoader {
public:
    enum NanoVDBMultiResImageLoaderType {
        eMultiResFloat
    };
public:
    NanoVDBMultiResImageLoader(vector<char> &g, NanoVDBMultiResImageLoaderType t);
    ~NanoVDBMultiResImageLoader();

    virtual bool load_metadata(const ImageDeviceFeatures& features,
        ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata,
        void* pixels,
        const size_t pixels_size,
        const bool associate_alpha) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3 &min_bbox, int3 &max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
    size_t levels;
    size_t largest_grid_id;
	
    vector<size_t> grid_offsets;
    vector<char> grids;
    NanoVDBMultiResImageLoaderType type;

    nanovdb::NanoGrid<float>* get_nanogrid(int level) const {
        return (nanovdb::NanoGrid<float>*) (grids.data() + grid_offsets[level]);
    }
};

class NanoVDBDerivatesImageLoader : public VDBImageLoader {
public:
    NanoVDBDerivatesImageLoader(vector<char> &g);
    ~NanoVDBDerivatesImageLoader();

    virtual bool load_metadata(const ImageDeviceFeatures& features,
        ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata,
        void* pixels,
        const size_t pixels_size,
        const bool associate_alpha) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3 &min_bbox, int3 &max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
    // On-disk structures matching the export format
    struct DerivFileHeader {
        uint32_t magic;
        uint32_t version;
        uint32_t payloadAlignment;
        uint32_t levelCount;
        uint32_t gridCount;
        uint32_t reserved1;
        uint64_t levelTableOffset;
        uint64_t gridTableOffset;
        uint64_t payloadBlockOffset;
        uint64_t totalFileSize;
        uint64_t reserved2;
    };

    struct DerivLevelHeader {
        uint32_t levelIndex;
        uint32_t derivativeCount;
        uint32_t firstGridIndex;
        uint32_t reserved;
    };

    struct DerivGridHeader {
        uint32_t levelIndex;
        uint32_t derivativeIndex;
        uint32_t derivativeCountInLevel;
        uint32_t reserved1;
        uint64_t payloadOffset;
        uint64_t payloadSize;
        int32_t  bboxMin[3];
        int32_t  bboxMax[3];
        uint32_t dims[3];
        uint32_t reserved2;
        char     name[56];
    };

    vector<char> bundle_data;
    DerivFileHeader file_header;
    size_t finest_level_id;

    const DerivFileHeader* get_file_header() const {
        return reinterpret_cast<const DerivFileHeader*>(bundle_data.data());
    }

    const DerivLevelHeader* get_level_table() const {
        return reinterpret_cast<const DerivLevelHeader*>(bundle_data.data() + file_header.levelTableOffset);
    }

    const DerivGridHeader* get_grid_table() const {
        return reinterpret_cast<const DerivGridHeader*>(bundle_data.data() + file_header.gridTableOffset);
    }

    nanovdb::NanoGrid<float>* get_grid(uint32_t grid_index) const {
        const DerivGridHeader* grid_headers = get_grid_table();
        return reinterpret_cast<nanovdb::NanoGrid<float>*>(
            const_cast<char*>(bundle_data.data()) + grid_headers[grid_index].payloadOffset);
    }
};

#endif

class RAWImageLoader : public VDBImageLoader {
public:
    enum RAWImageLoaderType {
        eRawFloat
    };

public:
    RAWImageLoader(vector<char> &g, int dx, int dy, int dz, float sx, float sy, float sz, RAWImageLoaderType t, int c);
    ~RAWImageLoader();

    virtual bool load_metadata(const ImageDeviceFeatures& features,
        ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata,
        void* pixels,
        const size_t pixels_size,
        const bool associate_alpha) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3 &min_bbox, int3 &max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
    int dimx, dimy, dimz;
    float scal_x, scal_y, scal_z;
    int channels;
    RAWImageLoaderType raw_type;
    vector<char> grid;
};

CCL_NAMESPACE_END
