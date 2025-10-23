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

#include "scene/image.h"

CCL_NAMESPACE_BEGIN

class VDBImageLoader : public ImageLoader {
 public:
#ifdef WITH_OPENVDB
  VDBImageLoader(openvdb::GridBase::ConstPtr grid_, const string &grid_name);
#endif
  VDBImageLoader(const string &grid_name);
  ~VDBImageLoader() override;

  bool load_metadata(const ImageDeviceFeatures &features, ImageMetaData &metadata) override;

  bool load_pixels(const ImageMetaData &metadata,
                   void *pixels,
                   const size_t pixels_size,
                   const bool associate_alpha) override;

  string name() const override;

  bool equals(const ImageLoader &other) const override;

  void cleanup() override;

  bool is_vdb_loader() const override;

  virtual bool is_simple_mesh() const;

  virtual void get_bbox(int3& min_bbox, int3& max_bbox);

  virtual float3 index_to_world(float3 in);

#ifdef WITH_OPENVDB
  openvdb::GridBase::ConstPtr get_grid();
#endif

#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)
  static nanovdb::GridHandle<> convert(openvdb::GridBase::ConstPtr g, int precision = 32);
  static void get_texture_info(nanovdb::NanoGrid<float>* ng, size_t ng_size, TextureInfo &info);
#endif

 protected:
  string grid_name;
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
#endif

class RAWImageLoader : public VDBImageLoader {
public:
    enum RAWImageLoaderType {
        eRawFloat,
        eRawByte,
        eRawHalf,
        eRawUShort
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
