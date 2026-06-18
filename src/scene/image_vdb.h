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
#ifdef WITH_ZFP_LOADER
#  include <zfp/array3.hpp>
#  ifdef __CUDACC__
#    include "kernel/util/zfp/array3_device.cuh"
#  endif
#endif

#include "scene/image_loader.h"
#include "scene/image.h"

#include "util/transform.h"
#include "kernel/util/image_3d_derivates.h"

CCL_NAMESPACE_BEGIN

class VDBImageLoader : public ImageLoader {
 public:
#ifdef WITH_OPENVDB
  VDBImageLoader(openvdb::GridBase::ConstPtr grid_,
                 const string &grid_name,
                 const float clipping = 0.f);
#endif
  VDBImageLoader(const string &grid_name, const float clipping = 0.f);
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

  //void set_image_params(const ImageParams &params) {
  //    this->params = params;
  //}

  //void get_image_params(ImageParams &params) const {
  //    params = this->params;
  //}

 protected:
  virtual void load_grid() {}

  void grid_from_dense_voxels(const size_t width,
                              const size_t height,
                              const size_t depth,
                              const int channels,
                              const float *voxels,
                              Transform transform_3d);

  string grid_name;
  float clipping = 0.f;
#ifdef WITH_OPENVDB
  openvdb::GridBase::ConstPtr grid;
  openvdb::CoordBBox bbox;
#endif
#ifdef WITH_NANOVDB
  nanovdb::GridHandle<> nanogrid;
  int precision = 32;
#endif

  //ImageParams params;
};

#ifdef WITH_NANOVDB
class NanoVDBImageLoader : public VDBImageLoader {
public:
    NanoVDBImageLoader(vector<char> &g);
    ~NanoVDBImageLoader();

    virtual bool load_metadata(ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata, void* pixels) override;

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

    virtual bool load_metadata(ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata, void* pixels) override;

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

    virtual bool load_metadata(ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata, void* pixels) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3 &min_bbox, int3 &max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
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
    RAWImageLoader(vector<char> &g, int3 d, float3 s, float3 tr, int3 bmin, int3 bmax, RAWImageLoaderType type, int c);
    ~RAWImageLoader();

    virtual bool load_metadata(ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata, void* pixels) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3 &min_bbox, int3 &max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
    int3 dim;
    float3 scale;
    float3 trans;
    int3 bbox_min;
    int3 bbox_max;
    int channels;
    RAWImageLoaderType raw_type;
    vector<char> grid;
};

#ifdef WITH_ZFP_LOADER
class ZFPImageLoader : public VDBImageLoader {
public:
  ZFPImageLoader(vector<char> &g,
                int3 d,
                float3 s,
                float3 t,
                int3 bmin,
                int3 bmax,
                size_t cache_size_bytes);
    ~ZFPImageLoader();

    virtual bool load_metadata(ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata, void* pixels) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3& min_bbox, int3& max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
    vector<char> zfp_data;
    zfp::array3f* zfp_array;
    int3 dim;
    float3 scale;
    float3 trans;
    int3 bbox_min;
    int3 bbox_max;
    size_t cache_size;
    void* dev_array_storage;

    void deserialize_zfp_array();
};
#endif

class CUBImageLoader : public VDBImageLoader {
public:
    CUBImageLoader(vector<char> &g);
    ~CUBImageLoader();

    virtual bool load_metadata(ImageMetaData& metadata) override;

    virtual bool load_pixels(const ImageMetaData& metadata, void* pixels) override;

    virtual string name() const override;

    virtual bool equals(const ImageLoader& other) const override;

    virtual void cleanup() override;

    virtual bool is_vdb_loader() const override;

    virtual bool is_simple_mesh() const override;

    virtual void get_bbox(int3& min_bbox, int3& max_bbox) override;

    virtual float3 index_to_world(float3 in) override;

protected:
    vector<char> cub_data;
    //void* dev_array_storage;

    void deserialize_cub_array();
    
    // Helper to get header from storage
    //SerializableCUBData* get_header(){
    //  if (cub_data.empty()) {
    //    return nullptr;
    //  }
    //  return (SerializableCUBData *)cub_data.data();
    //}
};

CCL_NAMESPACE_END
