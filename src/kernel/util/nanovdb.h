/* SPDX-FileCopyrightText: 2020-2021 Contributors to the OpenVDB Project
 * SPDX-FileCopyrightText: 2023-2025 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * This is an extract from NanoVDB.h, with minimal code needed for kernel side access to grids. The
 * original headers are not compatible with Metal due to missing address space qualifiers. */

#pragma once

#include "util/defines.h"
#include "util/math_base.h"
#include "util/math_float3.h"
#include "util/math_int3.h"
#include "util/types_base.h"
#include "util/types_float3.h"
#include "util/types_int3.h"

#ifndef __KERNEL_GPU__
#  include <climits>
#endif

CCL_NAMESPACE_BEGIN

#define NANOVDB_USE_SINGLE_ROOT_KEY
#define NANOVDB_DATA_ALIGNMENT 32  // NOLINT

namespace ccl_nanovdb {

/* Utilities */

template<typename DstT, typename SrcT>
const ccl_device ccl_global DstT *PtrAdd(const ccl_global SrcT *p, int64_t offset)
{
  return reinterpret_cast<const ccl_global DstT *>(reinterpret_cast<const ccl_global char *>(p) +
                                                   offset);
}

/* The value of an IEEE 754 double, given as its bits, rounded to float. For reading the
 * doubles which a grid stores on devices without a double type (Metal). Values outside
 * of the float range become zero or infinity. */
ccl_device_inline float DoubleBitsAsFloat(const uint64_t bits)
{
  const uint32_t sign = uint32_t(bits >> 63) << 31;
  const int exponent = int((bits >> 52) & 0x7ff) - 1023 + 127;
  if (exponent <= 0) {
    return __uint_as_float(sign);
  }
  if (exponent >= 255) {
    return __uint_as_float(sign | 0x7f800000u);
  }
  const uint32_t mantissa = uint32_t((bits >> 29) & 0x7fffffu);
  const uint32_t round = uint32_t((bits >> 28) & 1u);
  return __uint_as_float((sign | (uint32_t(exponent) << 23) | mantissa) + round);
}

/* Coord */

using Coord = int3;
using PackedCoord = packed_int3;

/* Mask */

template<uint32_t LOG2DIM> struct Mask {
  ccl_static_constexpr uint32_t SIZE = 1U << (3 * LOG2DIM);
  ccl_static_constexpr uint32_t WORD_COUNT = SIZE >> 6;
  uint64_t mWords[WORD_COUNT];

  ccl_device_inline_method bool isOff(const uint32_t n) const ccl_global
  {
    return 0 == (mWords[n >> 6] & (uint64_t(1) << (n & 63)));
  }
};

/* Grid */

template<typename TreeT> struct alignas(NANOVDB_DATA_ALIGNMENT) Grid {
  ccl_static_constexpr int MaxNameSize = 256;
  uint64_t mMagic;
  uint64_t mChecksum;
  uint32_t mVersion;
  uint32_t mFlags;
  uint32_t mGridIndex;
  uint32_t mGridCount;
  uint64_t mGridSize;
  char mGridName[MaxNameSize];
  uint8_t mMap[264];
  uint8_t mWorldBBox[48];  // double[6], but no doubles in Metal
  uint8_t mVoxelSize[24];  // double[3], but no doubles in Metal
  uint32_t mGridClass;
  uint32_t mGridType;
  uint32_t mData0;
  uint64_t mData1, mData2;

  using BuildType = typename TreeT::BuildType;

  const ccl_device_inline_method ccl_global TreeT &tree() const ccl_global
  {
    return *reinterpret_cast<const ccl_global TreeT *>(this + 1);
  }

  /* The index <-> world transform of the grid, in single precision.
   *
   * `mMap` is a `nanovdb::Map`, which stores the transform twice: as floats
   * (`float mMatF[9], mInvMatF[9], mVecF[3], mTaperF`, at the start) and as doubles after
   * them. Only the float copy is readable on Metal, which has no doubles. These are the
   * single precision counterparts of `worldToIndexF`, `indexToWorldF` and `voxelSize` of
   * `nanovdb::Grid`, for kernel code which has to compile on every device. */
  ccl_device_inline_method float mapF(const int i) const ccl_global
  {
    return reinterpret_cast<const ccl_global float *>(mMap)[i];
  }

  ccl_device_inline_method float3 worldToIndexF(const float3 P) const ccl_global
  {
    /* `Map::applyInverseMapF`: mInvMatF * (P - mVecF). */
    const float3 p = P - make_float3(mapF(18), mapF(19), mapF(20));
    return make_float3(p.x * mapF(9) + p.y * mapF(10) + p.z * mapF(11),
                       p.x * mapF(12) + p.y * mapF(13) + p.z * mapF(14),
                       p.x * mapF(15) + p.y * mapF(16) + p.z * mapF(17));
  }

  ccl_device_inline_method float3 indexToWorldF(const float3 P) const ccl_global
  {
    /* `Map::applyMapF`: mMatF * P + mVecF. */
    return make_float3(P.x * mapF(0) + P.y * mapF(1) + P.z * mapF(2) + mapF(18),
                       P.x * mapF(3) + P.y * mapF(4) + P.z * mapF(5) + mapF(19),
                       P.x * mapF(6) + P.y * mapF(7) + P.z * mapF(8) + mapF(20));
  }

  ccl_device_inline_method float3 voxelSizeF() const ccl_global
  {
    /* `mVoxelSize` is `double[3]`. Read as bits and rounded to float. */
    const ccl_global uint64_t *size = reinterpret_cast<const ccl_global uint64_t *>(mVoxelSize);
    return make_float3(DoubleBitsAsFloat(size[0]),
                       DoubleBitsAsFloat(size[1]),
                       DoubleBitsAsFloat(size[2]));
  }
};

/* Tree */

template<typename RootT> struct alignas(NANOVDB_DATA_ALIGNMENT) Tree {
  int64_t mNodeOffset[4];
  uint32_t mNodeCount[3];
  uint32_t mTileCount[3];
  uint64_t mVoxelCount;

  using ValueType = typename RootT::ValueType;
  using BuildType = typename RootT::BuildType;

  const ccl_device_inline_method ccl_global RootT &root() const ccl_global
  {
    return *reinterpret_cast<const ccl_global RootT *>(
        mNodeOffset[3] ? PtrAdd<uint8_t>(this, mNodeOffset[3]) : nullptr);
  }
};

/* RootNode */

template<typename ChildT> struct alignas(NANOVDB_DATA_ALIGNMENT) RootNode {
  using ValueType = typename ChildT::ValueType;
  using BuildType = typename ChildT::BuildType;

#ifdef NANOVDB_USE_SINGLE_ROOT_KEY
  using KeyT = uint64_t;
  static ccl_device_inline_method uint64_t CoordToKey(const Coord ijk)
  {
    return (uint64_t(uint32_t(ijk.z) >> ChildT::TOTAL)) |
           (uint64_t(uint32_t(ijk.y) >> ChildT::TOTAL) << 21) |
           (uint64_t(uint32_t(ijk.x) >> ChildT::TOTAL) << 42);
  }
#else
  using KeyT = Coord;
  static ccl_device_inline_method Coord CoordToKey(const CoordT ijk)
  {
    return ijk & ~ChildT::MASK;
  }
#endif
  PackedCoord mBBox[2];
  uint32_t mTableSize;

  ValueType mBackground;
  ValueType mMinimum;
  ValueType mMaximum;
  float mAverage;
  float mStdDevi;

  struct alignas(NANOVDB_DATA_ALIGNMENT) Tile {
    KeyT key;
    int64_t child;
    uint32_t state;
    ValueType value;
  };

  const ccl_device_inline_method ccl_global Tile *probeTile(const Coord ijk) const ccl_global
  {
    const auto key = CoordToKey(ijk);
    const ccl_global Tile *p = reinterpret_cast<const ccl_global Tile *>(this + 1);
    const ccl_global Tile *q = p + mTableSize;
    for (; p < q; ++p) {
      if (p->key == key) {
        return p;
      }
    }
    return nullptr;
  }

  const ccl_device_inline_method ccl_global ChildT *getChild(const ccl_global Tile *tile) const
      ccl_global
  {
    return PtrAdd<ChildT>(this, tile->child);
  }

  ccl_static_constexpr uint32_t LEVEL = 1 + ChildT::LEVEL;
};

/* InternalNode */

template<typename ChildT, const uint32_t Log2Dim = ChildT::LOG2DIM + 1>
struct alignas(NANOVDB_DATA_ALIGNMENT) InternalNode {
  using ValueType = typename ChildT::ValueType;
  using BuildType = typename ChildT::BuildType;

  union Tile {
    ValueType value;
    int64_t child;
  };

  PackedCoord mBBox[2];
  uint64_t mFlags;
  Mask<Log2Dim> mValueMask;
  Mask<Log2Dim> mChildMask;

  ValueType mMinimum;
  ValueType mMaximum;
  float mAverage;
  float mStdDevi;

  alignas(32) Tile mTable[1u << (3 * Log2Dim)];

  const ccl_device_inline_method ccl_global ChildT *getChild(const uint32_t n) const ccl_global
  {
    return PtrAdd<ChildT>(this, mTable[n].child);
  }

  ccl_static_constexpr uint32_t LOG2DIM = Log2Dim;
  ccl_static_constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL;
  ccl_static_constexpr uint32_t DIM = 1u << TOTAL;
  ccl_static_constexpr uint32_t SIZE = 1u << (3 * LOG2DIM);
  ccl_static_constexpr uint32_t MASK = (1u << TOTAL) - 1u;
  ccl_static_constexpr uint32_t LEVEL = 1 + ChildT::LEVEL;

  static ccl_device_inline_method uint32_t CoordToOffset(const Coord ijk)
  {
    return (((ijk.x & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) |
           (((ijk.y & MASK) >> ChildT::TOTAL) << (LOG2DIM)) | ((ijk.z & MASK) >> ChildT::TOTAL);
  }
};

/* LeafData */

template<typename ValueT, const uint32_t LOG2DIM> struct alignas(NANOVDB_DATA_ALIGNMENT) LeafData {
  using ValueType = ValueT;
  using BuildType = ValueT;

  PackedCoord mBBoxMin;
  uint8_t mBBoxDif[3];
  uint8_t mFlags;
  Mask<LOG2DIM> mValueMask;

  ValueType mMinimum;
  ValueType mMaximum;
  float mAverage;
  float mStdDevi;
  alignas(32) ValueType mValues[1u << 3 * LOG2DIM];

  ccl_device_inline_method ValueType getValue(const uint32_t i) const ccl_global
  {
    return mValues[i];
  }
};

/* LeafFnBase */

template<uint32_t LOG2DIM> struct alignas(NANOVDB_DATA_ALIGNMENT) LeafFnBase {
  PackedCoord mBBoxMin;
  uint8_t mBBoxDif[3];
  uint8_t mFlags;
  Mask<LOG2DIM> mValueMask;

  float mMinimum;
  float mQuantum;
  uint16_t mMin, mMax, mAvg, mDev;
};

/* LeafData<Fp16> */

class Fp16 {};

template<uint32_t LOG2DIM> struct alignas(NANOVDB_DATA_ALIGNMENT) LeafData<Fp16, LOG2DIM> {
  using ValueType = float;
  using BuildType = Fp16;

  LeafFnBase<LOG2DIM> base;
  alignas(32) uint16_t mCode[1u << 3 * LOG2DIM];

  ccl_device_inline_method float getValue(const uint32_t i) const ccl_global
  {
    return mCode[i] * base.mQuantum + base.mMinimum;
  }
};

/* LeafData<FpN> */

class FpN {};

template<uint32_t LOG2DIM> struct alignas(NANOVDB_DATA_ALIGNMENT) LeafData<FpN, LOG2DIM> {
  using ValueType = float;
  using BuildType = FpN;

  LeafFnBase<LOG2DIM> base;

  ccl_device_inline_method float getValue(const uint32_t i) const ccl_global
  {
    const int b = base.mFlags >> 5;
    uint32_t code = reinterpret_cast<const ccl_global uint32_t *>(this + 1)[i >> (5 - b)];
    code >>= (i & ((32 >> b) - 1)) << b;
    code &= (1 << (1 << b)) - 1;
    return float(code) * base.mQuantum + base.mMinimum;
  }
};

/* LeafNode */

template<typename BuildT, const uint32_t Log2Dim = 3>
struct alignas(NANOVDB_DATA_ALIGNMENT) LeafNode {
  using DataType = LeafData<BuildT, Log2Dim>;
  using ValueType = typename DataType::ValueType;
  using BuildType = typename DataType::BuildType;

  DataType data;

  ccl_static_constexpr uint32_t LOG2DIM = Log2Dim;
  ccl_static_constexpr uint32_t TOTAL = LOG2DIM;
  ccl_static_constexpr uint32_t DIM = 1u << TOTAL;
  ccl_static_constexpr uint32_t SIZE = 1u << 3 * LOG2DIM;
  ccl_static_constexpr uint32_t MASK = (1u << LOG2DIM) - 1u;
  ccl_static_constexpr uint32_t LEVEL = 0;

  static ccl_device_inline_method uint32_t CoordToOffset(const Coord ijk)
  {
    return ((ijk.x & MASK) << (2 * LOG2DIM)) | ((ijk.y & MASK) << LOG2DIM) | (ijk.z & MASK);
  }

  ccl_device_inline_method ValueType getValue(const uint32_t offset) const ccl_global
  {
    return data.getValue(offset);
  }

  ccl_device_inline_method ValueType getValue(const Coord ijk) const ccl_global
  {
    return getValue(CoordToOffset(ijk));
  }
};

/* Template Specializations */

template<typename BuildT> using NanoLeaf = LeafNode<BuildT, 3>;
template<typename BuildT> using NanoLower = InternalNode<NanoLeaf<BuildT>, 4>;
template<typename BuildT> using NanoUpper = InternalNode<NanoLower<BuildT>, 5>;
template<typename BuildT> using NanoRoot = RootNode<NanoUpper<BuildT>>;
template<typename BuildT> using NanoTree = Tree<NanoRoot<BuildT>>;
template<typename BuildT> using NanoGrid = Grid<NanoTree<BuildT>>;

/* ReadAccessor */

template<typename BuildT> class ReadAccessor {
  using RootT = NanoRoot<BuildT>;
  using LeafT = NanoLeaf<BuildT>;

  mutable const ccl_global RootT *mRoot;

 public:
  using ValueType = typename RootT::ValueType;

  ccl_device_inline_method ReadAccessor(const ccl_global RootT &root) : mRoot(&root) {}

  ccl_device_inline_method ValueType getValue(const Coord ijk) const
  {
    const ccl_global auto *tile = mRoot->probeTile(ijk);
    if (tile == nullptr) {
      return mRoot->mBackground;
    }
    if (tile->child == 0) {
      return tile->value;
    }

    const ccl_global auto *upper = mRoot->getChild(tile);
    const uint32_t upper_n = upper->CoordToOffset(ijk);
    if (upper->mChildMask.isOff(upper_n)) {
      return upper->mTable[upper_n].value;
    }

    const ccl_global auto *lower = upper->getChild(upper_n);
    const uint32_t lower_n = lower->CoordToOffset(ijk);
    if (lower->mChildMask.isOff(lower_n)) {
      return lower->mTable[lower_n].value;
    }

    const ccl_global LeafT *leaf = lower->getChild(lower_n);
    return leaf->getValue(ijk);
  }
};

template<typename BuildT> class CachedReadAccessor {
  using RootT = NanoRoot<BuildT>;
  using UpperT = NanoUpper<BuildT>;
  using LowerT = NanoLower<BuildT>;
  using LeafT = NanoLeaf<BuildT>;

  mutable Coord mKeys[3] = {make_int3(INT_MAX), make_int3(INT_MAX), make_int3(INT_MAX)};
  mutable const ccl_global RootT *mRoot = nullptr;
  mutable const ccl_global void *mNode[3] = {nullptr, nullptr, nullptr};

 public:
  using ValueType = typename RootT::ValueType;

  ccl_device_inline_method CachedReadAccessor(const ccl_global RootT &root) : mRoot(&root) {}

  template<typename NodeT> ccl_device_inline_method bool isCached(const Coord ijk) const
  {
    return (ijk.x & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL].x &&
           (ijk.y & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL].y &&
           (ijk.z & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL].z;
  }

  ccl_device_inline_method ValueType getValueAndCache(const ccl_global RootT &node,
                                                      const Coord ijk) const
  {
    if (const ccl_global auto *tile = node.probeTile(ijk)) {
      if (tile->child != 0) {
        const ccl_global auto *child = node.getChild(tile);
        insert(ijk, child);
        return getValueAndCache(*child, ijk);
      }
      return tile->value;
    }
    return node.mBackground;
  }

  ccl_device_inline_method ValueType getValueAndCache(const ccl_global LeafT &node,
                                                      const Coord ijk) const
  {
    return node.getValue(ijk);
  }

  template<typename NodeT>
  ccl_device_inline_method ValueType getValueAndCache(const ccl_global NodeT &node,
                                                      const Coord ijk) const
  {
    const uint32_t n = node.CoordToOffset(ijk);
    if (node.mChildMask.isOff(n)) {
      return node.mTable[n].value;
    }
    const ccl_global auto *child = node.getChild(n);
    insert(ijk, child);
    return getValueAndCache(*child, ijk);
  }

  ccl_device_inline_method ValueType getValue(const Coord ijk) const
  {
    if (isCached<LeafT>(ijk)) {
      return getValueAndCache(*((const ccl_global LeafT *)mNode[0]), ijk);
    }
    if (isCached<LowerT>(ijk)) {
      return getValueAndCache(*((const ccl_global LowerT *)mNode[1]), ijk);
    }
    if (isCached<UpperT>(ijk)) {
      return getValueAndCache(*((const ccl_global UpperT *)mNode[2]), ijk);
    }
    return getValueAndCache(*mRoot, ijk);
  }

  template<typename NodeT>
  ccl_device_inline_method void insert(const Coord ijk, const ccl_global NodeT *node) const
  {
    mKeys[NodeT::LEVEL] = ijk & ~NodeT::MASK;
    mNode[NodeT::LEVEL] = node;
  }
};

}  // namespace nanovdb

CCL_NAMESPACE_END
