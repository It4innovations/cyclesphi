// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <map>

#define NANOVDB_USE_OPENVDB

#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

// Include shared derivative format structures
#include "../../src/kernel/util/image_3d_derivates.h"

// ============================================================================
// Binary File Format for NanoVDB Multi-Level Derivative Grids with Vec4 Packing
// ============================================================================
//
// File Layout:
//   1. FileHeader (32-byte aligned)
//   2. LevelHeader[levelCount] (array, naturally aligned)
//   3. GridHeader[gridCount] (array, naturally aligned)
//   4. Payload Block: NanoVDB grid data, each grid 32-byte aligned
//
// Grid naming convention: L-{totalLevels}-{levelIndex}-D-{derivsInLevel}-{derivIndex}
//   Example: L-05-04-D-20-19
//     - totalLevels: 5
//     - levelIndex: 4
//     - derivsInLevel: 20
//     - derivIndex: 19
//
// Grids are sorted by: levelIndex (ascending), then derivativeIndex (ascending)
// ============================================================================

constexpr uint32_t PAYLOAD_ALIGNMENT = 32;
constexpr uint32_t FILE_MAGIC = 0x4E56444D; // "NVDM" = NanoVDB Derivatives Multi-level
constexpr uint32_t FILE_VERSION = 1;

// ============================================================================
// Helper Functions
// ============================================================================

// Align a value up to the given alignment (must be power of 2)
inline uint64_t alignUp(uint64_t value, uint64_t alignment) {
    return (value + (alignment - 1)) & ~(alignment - 1);
}

// Write padding bytes to stream
inline void writePadding(std::ofstream& out, uint64_t currentPos, uint64_t alignment) {
    const uint64_t aligned = alignUp(currentPos, alignment);
    const uint64_t paddingSize = aligned - currentPos;
    if (paddingSize > 0) {
        static const char zeros[256] = {0};
        uint64_t remaining = paddingSize;
        while (remaining > 0) {
            const uint64_t chunk = std::min<uint64_t>(remaining, sizeof(zeros));
            out.write(zeros, chunk);
            remaining -= chunk;
        }
    }
}

// ============================================================================
// On-Disk Structures (portable, fixed-width types)
// ============================================================================

// Note: We do NOT use packed structs for portability.
// Alignment is natural; explicit padding ensures consistent layout.

// Using shared structs from image_3d_derivates.h
// Verify struct sizes match expectations
static_assert(sizeof(DerivFileHeader) == 64, "DerivFileHeader must be 64 bytes");
static_assert(sizeof(DerivLevelHeader) == 16, "DerivLevelHeader must be 16 bytes");
static_assert(sizeof(DerivGridHeader) == 128, "DerivGridHeader must be 128 bytes");

// ============================================================================
// Input Structure for API
// ============================================================================

struct GridInput {
    std::string name;        // Grid name (e.g., "L-05-04-D-20-19")
    const void* data;        // Pointer to NanoVDB grid data
    uint64_t sizeBytes;      // Size of grid data in bytes
    int32_t bboxMin[3];      // Bounding box minimum
    int32_t bboxMax[3];      // Bounding box maximum
    uint32_t dims[3];        // Grid dimensions
    uint32_t gridType;       // Grid type: DERIV_GRID_TYPE_FLOAT/VEC3F/VEC4F
};

// ============================================================================
// Grid Metadata Parsing
// ============================================================================

struct GridMetadata {
    uint32_t totalLevels;
    uint32_t levelIndex;
    uint32_t derivativeCountInLevel;
    uint32_t derivativeIndex;
    bool valid;
};

// Parse grid name like "L-05-04-D-20-19"
// Format: L-{totalLevels}-{levelIndex}-D-{derivCountInLevel}-{derivIndex}
GridMetadata parseGridName(const std::string &name)
{
  GridMetadata meta = {0, 0, 0, 0, false};

    // Expected format: L-<n>-<n>-D-<n>-<n>
    // Examples: L-05-00-D-01-00, L-7-0-D-1-0
    // Minimum valid length is 11 characters: "L-1-0-D-1-0".
    if (name.size() < 11) {
    return meta;
  }

  if (name[0] != 'L' || name[1] != '-') {
    return meta;
  }

  // Find delimiters
  const size_t pos1 = name.find('-', 2);  // after totalLevels
  if (pos1 == std::string::npos) {
    return meta;
  }

  const size_t pos2 = name.find('-', pos1 + 1);  // after levelIndex
  if (pos2 == std::string::npos) {
    return meta;
  }

  const size_t posD = name.find("-D-", pos2);  // marker before derivatives
  if (posD == std::string::npos) {
    return meta;
  }

  const size_t pos3 = name.find('-', posD + 3);  // between derivCount and derivIndex
  if (pos3 == std::string::npos) {
    return meta;
  }

  // Ensure there is no unexpected extra '-'
  if (name.find('-', pos3 + 1) != std::string::npos) {
    return meta;
  }

  try {
    const std::string totalLevelsStr = name.substr(2, pos1 - 2);
    const std::string levelIndexStr = name.substr(pos1 + 1, pos2 - pos1 - 1);
    const std::string derivCountStr = name.substr(posD + 3, pos3 - (posD + 3));
    const std::string derivIndexStr = name.substr(pos3 + 1);

    if (totalLevelsStr.empty() || levelIndexStr.empty() || derivCountStr.empty() ||
        derivIndexStr.empty())
    {
      return meta;
    }

    meta.totalLevels = std::stoul(totalLevelsStr);
    meta.levelIndex = std::stoul(levelIndexStr);
    meta.derivativeCountInLevel = std::stoul(derivCountStr);
    meta.derivativeIndex = std::stoul(derivIndexStr);
    meta.valid = true;
  }
  catch (...) {
    meta = {0, 0, 0, 0, false};
  }

  return meta;
}

// ============================================================================
// Level Aggregation
// ============================================================================

struct LevelInfo {
    uint32_t levelIndex;
    uint32_t derivativeCount;
    uint32_t firstGridIndex;
    std::vector<uint32_t> gridIndices; // Indices into sorted grid array
};

using Vec4fTree = openvdb::tree::Tree4<openvdb::Vec4f, 5, 4, 3>::Type;
using Vec4fGrid = openvdb::Grid<Vec4fTree>;

// Map component count to DerivGridType
DerivGridType getDerivGridType(size_t componentCount)
{
    switch (componentCount) {
        case 1: return DERIV_GRID_TYPE_FLOAT;
        case 2:
        case 3: return DERIV_GRID_TYPE_VEC3F;
        case 4: return DERIV_GRID_TYPE_VEC4F;
        default: return DERIV_GRID_TYPE_FLOAT;
    }
}

const char* packedGridTypeName(size_t componentCount)
{
    switch (componentCount) {
        case 1: return "float";
        case 2: return "vec3f(padded)";
        case 3: return "vec3f";
        case 4: return "vec4f";
        default: return "unknown";
    }
}

template<typename PackedGridT, typename VecT>
openvdb::GridBase::Ptr packIntoVectorGrid(const std::vector<openvdb::FloatGrid::Ptr>& orderedGrids,
                                          size_t startIndex,
                                          size_t componentCount)
{
    auto packedGrid = PackedGridT::create(VecT(0.0f));
    packedGrid->setTransform(orderedGrids[startIndex]->transform().copy());
    packedGrid->setGridClass(orderedGrids[startIndex]->getGridClass());

    auto& tree = packedGrid->tree();
    for (size_t component = 0; component < componentCount; ++component) {
        const auto& src = orderedGrids[startIndex + component];
        for (auto it = src->cbeginValueOn(); it.test(); ++it) {
            const openvdb::Coord ijk = it.getCoord();
            VecT value = tree.getValue(ijk);
            value[component] = *it;
            tree.setValueOn(ijk, value);
        }
    }

    return packedGrid;
}

openvdb::GridBase::Ptr packDerivativeChunk(const std::vector<openvdb::FloatGrid::Ptr>& orderedGrids,
                                           size_t startIndex,
                                           size_t componentCount)
{
    assert(componentCount >= 1 && componentCount <= 4);

    if (componentCount == 1) {
        return orderedGrids[startIndex];
    }
    if (componentCount == 2) {
        // NanoVDB conversion path used here does not support Vec2f grids.
        // Store 2 derivatives in XY of a Vec3f grid; Z stays at zero.
        return packIntoVectorGrid<openvdb::Vec3fGrid, openvdb::Vec3f>(orderedGrids,
                                                                       startIndex,
                                                                       componentCount);
    }
    if (componentCount == 3) {
        return packIntoVectorGrid<openvdb::Vec3fGrid, openvdb::Vec3f>(orderedGrids,
                                                                       startIndex,
                                                                       componentCount);
    }

    return packIntoVectorGrid<Vec4fGrid, openvdb::Vec4f>(orderedGrids,
                                                          startIndex,
                                                          componentCount);
}

// ============================================================================
// Main Export Function
// ============================================================================

bool writeNanoVdbBundle(const std::string& outputPath, const std::vector<GridInput>& grids, bool switchOrderLevels = false) {
    if (grids.empty()) {
        std::cerr << "Error: No grids provided\n";
        return false;
    }
    
    // Step 1: Parse metadata and validate
    std::cout << "Parsing grid metadata...\n";
    
    struct GridWithMetadata {
        GridInput input;
        GridMetadata meta;
        size_t originalIndex;
    };
    
    std::vector<GridWithMetadata> gridsMeta;
    gridsMeta.reserve(grids.size());
    
    uint32_t expectedTotalLevels = 0;
    
    for (size_t i = 0; i < grids.size(); ++i) {
        GridWithMetadata gm;
        gm.input = grids[i];
        gm.meta = parseGridName(grids[i].name);
        gm.originalIndex = i;
        
        if (!gm.meta.valid) {
            std::cerr << "Error: Invalid grid name format: " << grids[i].name << "\n";
            return false;
        }
        
        // Validate totalLevels consistency
        if (i == 0) {
            expectedTotalLevels = gm.meta.totalLevels;
        } else if (gm.meta.totalLevels != expectedTotalLevels) {
            std::cerr << "Error: Inconsistent totalLevels in grid names\n";
            return false;
        }
        
        std::cout << "  Grid " << i << ": " << grids[i].name 
                  << " -> Level " << gm.meta.levelIndex 
                  << ", Deriv " << gm.meta.derivativeIndex 
                  << " (of " << gm.meta.derivativeCountInLevel << ")\n";
        
        gridsMeta.push_back(gm);
    }
    
    // Step 2: Sort grids by levelIndex (ascending), then derivativeIndex (ascending)
    std::cout << "\nSorting grids by level and derivative index...\n";
    std::sort(gridsMeta.begin(), gridsMeta.end(), 
        [](const GridWithMetadata& a, const GridWithMetadata& b) {
            if (a.meta.levelIndex != b.meta.levelIndex) {
                return a.meta.levelIndex < b.meta.levelIndex;
            }
            return a.meta.derivativeIndex < b.meta.derivativeIndex;
        });
    
    // Step 3: Build level table
    std::cout << "Building level table...\n";
    std::vector<LevelInfo> levels;
    
    for (size_t i = 0; i < gridsMeta.size(); ++i) {
        const auto& gm = gridsMeta[i];
        const uint32_t levelIdx = gm.meta.levelIndex;
        
        // Find or create level
        auto it = std::find_if(levels.begin(), levels.end(),
            [levelIdx](const LevelInfo& l) { return l.levelIndex == levelIdx; });
        
        if (it == levels.end()) {
            // New level
            LevelInfo level;
            level.levelIndex = levelIdx;
            level.derivativeCount = gm.meta.derivativeCountInLevel;
            level.firstGridIndex = static_cast<uint32_t>(i);
            level.gridIndices.push_back(static_cast<uint32_t>(i));
            levels.push_back(level);
        } else {
            // Existing level
            it->gridIndices.push_back(static_cast<uint32_t>(i));
            
            // Validate derivative count consistency within level
            if (it->derivativeCount != gm.meta.derivativeCountInLevel) {
                std::cerr << "Error: Inconsistent derivativeCount within level " << levelIdx << "\n";
                return false;
            }
        }
    }
    
    // Sort levels by levelIndex
    std::sort(levels.begin(), levels.end(),
        [](const LevelInfo& a, const LevelInfo& b) {
            return a.levelIndex < b.levelIndex;
        });
    
    // Validate level consistency
    for (const auto& level : levels) {
        if (level.gridIndices.size() != level.derivativeCount) {
            std::cerr << "Error: Level " << level.levelIndex 
                      << " expects " << level.derivativeCount 
                      << " grids but has " << level.gridIndices.size() << "\n";
            return false;
        }
        std::cout << "  Level " << level.levelIndex 
                  << ": " << level.derivativeCount << " derivatives, "
                  << "first grid index = " << level.firstGridIndex << "\n";
    }
    
    const uint32_t levelCount = static_cast<uint32_t>(levels.size());
    const uint32_t gridCount = static_cast<uint32_t>(gridsMeta.size());
    
    std::cout << "\nTotal levels: " << levelCount << "\n";
    std::cout << "Total grids: " << gridCount << "\n";
    
    // Step 4: Calculate offsets
    std::cout << "\nCalculating file layout...\n";
    
    uint64_t currentOffset = 0;
    
    // FileHeader (64 bytes, align to 32)
    const uint64_t fileHeaderOffset = 0;
    currentOffset = sizeof(DerivFileHeader);
    currentOffset = alignUp(currentOffset, PAYLOAD_ALIGNMENT);
    
    // LevelHeader array
    const uint64_t levelTableOffset = currentOffset;
    currentOffset += levelCount * sizeof(DerivLevelHeader);
    
    // GridHeader array
    const uint64_t gridTableOffset = currentOffset;
    currentOffset += gridCount * sizeof(DerivGridHeader);
    
    // Payload block (align to PAYLOAD_ALIGNMENT)
    const uint64_t payloadBlockOffset = alignUp(currentOffset, PAYLOAD_ALIGNMENT);
    currentOffset = payloadBlockOffset;
    
    // Calculate each grid's payload offset
    std::vector<uint64_t> payloadOffsets(gridCount);
    for (uint32_t i = 0; i < gridCount; ++i) {
        payloadOffsets[i] = currentOffset;
        const uint64_t payloadSize = gridsMeta[i].input.sizeBytes;
        currentOffset += payloadSize;
        currentOffset = alignUp(currentOffset, PAYLOAD_ALIGNMENT); // Align next payload
    }
    
    const uint64_t totalFileSize = currentOffset;
    
    std::cout << "  FileHeader offset: " << fileHeaderOffset << "\n";
    std::cout << "  LevelTable offset: " << levelTableOffset << "\n";
    std::cout << "  GridTable offset: " << gridTableOffset << "\n";
    std::cout << "  Payload block offset: " << payloadBlockOffset << "\n";
    std::cout << "  Total file size: " << totalFileSize << " bytes\n";
    
    // Step 5: Validate alignment
    for (uint32_t i = 0; i < gridCount; ++i) {
        if (payloadOffsets[i] % PAYLOAD_ALIGNMENT != 0) {
            std::cerr << "Error: Grid " << i << " payload offset " << payloadOffsets[i] 
                      << " is not aligned to " << PAYLOAD_ALIGNMENT << " bytes\n";
            return false;
        }
    }
    
    // Step 6: Write file
    std::cout << "\nWriting binary file: " << outputPath << "\n";
    std::ofstream out(outputPath, std::ios::binary);
    if (!out) {
        std::cerr << "Error: Failed to open output file\n";
        return false;
    }
    
    // Write DerivFileHeader
    DerivFileHeader fileHeader = {};
    fileHeader.magic = FILE_MAGIC;
    fileHeader.version = FILE_VERSION;
    fileHeader.payloadAlignment = PAYLOAD_ALIGNMENT;
    fileHeader.levelCount = levelCount;
    fileHeader.gridCount = gridCount;
    fileHeader.levelTableOffset = levelTableOffset;
    fileHeader.gridTableOffset = gridTableOffset;
    fileHeader.payloadBlockOffset = payloadBlockOffset;
    fileHeader.totalFileSize = totalFileSize;
    
    out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(DerivFileHeader));
    writePadding(out, sizeof(DerivFileHeader), PAYLOAD_ALIGNMENT);
    
    // Write DerivLevelHeader array
    if (switchOrderLevels) {
        // Write levels in reverse order
        for (auto it = levels.rbegin(); it != levels.rend(); ++it) {
            DerivLevelHeader lh = {};
            lh.levelIndex = it->levelIndex;
            lh.derivativeCount = it->derivativeCount;
            lh.firstGridIndex = it->firstGridIndex;
            out.write(reinterpret_cast<const char*>(&lh), sizeof(DerivLevelHeader));
        }
    } else {
        // Write levels in normal order
        for (const auto& level : levels) {
            DerivLevelHeader lh = {};
            lh.levelIndex = level.levelIndex;
            lh.derivativeCount = level.derivativeCount;
            lh.firstGridIndex = level.firstGridIndex;
            out.write(reinterpret_cast<const char*>(&lh), sizeof(DerivLevelHeader));
        }
    }
    
    // Write DerivGridHeader array
    for (uint32_t i = 0; i < gridCount; ++i) {
        const auto& gm = gridsMeta[i];
        
        DerivGridHeader gh = {};
        gh.levelIndex = gm.meta.levelIndex;
        gh.derivativeIndex = gm.meta.derivativeIndex;
        gh.derivativeCountInLevel = gm.meta.derivativeCountInLevel;
        gh.gridType = gm.input.gridType;  // Use gridType from input
        gh.payloadOffset = payloadOffsets[i];
        gh.payloadSize = gm.input.sizeBytes;
        std::memcpy(gh.bboxMin, gm.input.bboxMin, sizeof(gh.bboxMin));
        std::memcpy(gh.bboxMax, gm.input.bboxMax, sizeof(gh.bboxMax));
        std::memcpy(gh.dims, gm.input.dims, sizeof(gh.dims));
        
        // Copy name (truncate if necessary)
        std::strncpy(gh.name, gm.input.name.c_str(), sizeof(gh.name) - 1);
        gh.name[sizeof(gh.name) - 1] = '\0';
        
        out.write(reinterpret_cast<const char*>(&gh), sizeof(DerivGridHeader));
    }
    
    // Align to payload block
    uint64_t currentPos = gridTableOffset + gridCount * sizeof(DerivGridHeader);
    writePadding(out, currentPos, PAYLOAD_ALIGNMENT);
    
    // Write payloads
    for (uint32_t i = 0; i < gridCount; ++i) {
        const auto& gm = gridsMeta[i];
        
        // Verify we're at the right offset
        currentPos = payloadOffsets[i];
        
        // Write payload
        out.write(reinterpret_cast<const char*>(gm.input.data), gm.input.sizeBytes);
        std::cout << "  Written grid " << i << " (" << gm.input.name << "): " 
                  << gm.input.sizeBytes << " bytes at offset " << payloadOffsets[i] << "\n";
        
        // Align to next payload
        currentPos += gm.input.sizeBytes;
        writePadding(out, currentPos, PAYLOAD_ALIGNMENT);
    }
    
    out.close();
    
    if (!out.good()) {
        std::cerr << "Error: Failed to write file completely\n";
        return false;
    }
    
    std::cout << "\nSuccessfully wrote " << gridCount << " grids in " << levelCount << " levels\n";
    std::cout << "Total file size: " << totalFileSize << " bytes\n";
    
    return true;
}

// ============================================================================
// Main Program: Convert OpenVDB to NanoVDB Bundle
// ============================================================================

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.vdb> <output.bin> [--switch-order-levels]\n";
        std::cerr << "  Reads OpenVDB file containing derivative grids and exports them\n";
        std::cerr << "  to a binary NanoVDB bundle with hierarchical headers.\n";
        std::cerr << "\n";
        std::cerr << "Options:\n";
        std::cerr << "  --switch-order-levels  Reverse the order of levels when writing\n";
        std::cerr << "\n";
        std::cerr << "Grid naming convention: L-{totalLevels}-{levelIdx}-D-{derivCount}-{derivIdx}\n";
        std::cerr << "  Example: L-05-04-D-20-19\n";
        return EXIT_FAILURE;
    }

    const std::string inputFile = argv[1];
    const std::string outputFile = argv[2];
    bool switchOrderLevels = false;

    // Parse optional arguments
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--switch-order-levels") == 0) {
            switchOrderLevels = true;
        }
    }

    try {
        // Initialize OpenVDB
        openvdb::initialize();

        std::cout << "============================================\n";
        std::cout << "NanoVDB Derivative Bundle Exporter\n";
        std::cout << "============================================\n";
        std::cout << "Input:  " << inputFile << "\n";
        std::cout << "Output: " << outputFile << "\n\n";

        // Open the OpenVDB file
        openvdb::io::File file(inputFile);
        file.open(false); // disable delayed loading

        // Get all grids from the file
        auto openvdbGrids = file.getGrids();
        
        if (!openvdbGrids || openvdbGrids->empty()) {
            std::cerr << "Error: No grids found in the OpenVDB file\n";
            return EXIT_FAILURE;
        }

        std::cout << "Found " << openvdbGrids->size() << " OpenVDB grid(s)\n\n";

        // Convert each OpenVDB grid to FloatGrid and parse metadata.
        struct RawDerivativeGrid {
            std::string originalName;
            GridMetadata meta;
            openvdb::FloatGrid::Ptr grid;
        };

        std::map<uint32_t, std::vector<RawDerivativeGrid>> levelDerivatives;
        uint32_t expectedTotalLevels = 0;

        for (auto& openvdbGrid : *openvdbGrids) {
            const std::string gridName = openvdbGrid->getName();
            const GridMetadata meta = parseGridName(gridName);

            if (!meta.valid) {
                std::cerr << "Error: Invalid grid name format: " << gridName << "\n";
                return EXIT_FAILURE;
            }

            if (levelDerivatives.empty()) {
                expectedTotalLevels = meta.totalLevels;
            }
            else if (meta.totalLevels != expectedTotalLevels) {
                std::cerr << "Error: Inconsistent totalLevels in input grids\n";
                return EXIT_FAILURE;
            }

            auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(openvdbGrid);
            if (!floatGrid) {
                std::cerr << "Error: Grid '" << gridName << "' is not a FloatGrid. "
                          << "Only float derivatives are supported for packing.\n";
                return EXIT_FAILURE;
            }

            std::cout << "Read derivative grid: " << gridName
                      << " -> level " << meta.levelIndex
                      << ", derivative " << meta.derivativeIndex
                      << " of " << meta.derivativeCountInLevel << "\n";

            levelDerivatives[meta.levelIndex].push_back({gridName, meta, floatGrid});
        }

        if (levelDerivatives.empty()) {
            std::cerr << "Error: No valid derivative grids found\n";
            return EXIT_FAILURE;
        }

        // Pack derivatives per level into scalar/vector grids with max 4 components.
        std::cout << "\nPacking derivatives by level (max vec4) ...\n";
        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> nanoHandles;
        std::vector<GridInput> gridInputs;

        for (auto& levelEntry : levelDerivatives) {
            const uint32_t levelIndex = levelEntry.first;
            auto& derivatives = levelEntry.second;

            std::sort(derivatives.begin(),
                      derivatives.end(),
                      [](const RawDerivativeGrid& a, const RawDerivativeGrid& b) {
                          return a.meta.derivativeIndex < b.meta.derivativeIndex;
                      });

            const uint32_t rawDerivativeCount = derivatives.front().meta.derivativeCountInLevel;
            for (const auto& derivative : derivatives) {
                if (derivative.meta.derivativeCountInLevel != rawDerivativeCount) {
                    std::cerr << "Error: Inconsistent derivative count for level " << levelIndex
                              << "\n";
                    return EXIT_FAILURE;
                }
            }

            if (derivatives.size() != rawDerivativeCount) {
                std::cerr << "Error: Level " << levelIndex << " expects " << rawDerivativeCount
                          << " derivatives but found " << derivatives.size() << "\n";
                return EXIT_FAILURE;
            }

            for (uint32_t i = 0; i < rawDerivativeCount; ++i) {
                if (derivatives[i].meta.derivativeIndex != i) {
                    std::cerr << "Error: Missing or non-contiguous derivative indices at level "
                              << levelIndex << "\n";
                    return EXIT_FAILURE;
                }
            }

            std::vector<openvdb::FloatGrid::Ptr> orderedGrids;
            orderedGrids.reserve(derivatives.size());
            for (const auto& derivative : derivatives) {
                orderedGrids.push_back(derivative.grid);
            }

            const uint32_t packedGridCount = (rawDerivativeCount + 3u) / 4u;
            std::cout << "Level " << levelIndex << ": " << rawDerivativeCount
                      << " derivatives -> " << packedGridCount << " packed grid(s)\n";

            for (uint32_t packedIndex = 0; packedIndex < packedGridCount; ++packedIndex) {
                const size_t start = static_cast<size_t>(packedIndex) * 4u;
                const size_t componentCount = std::min<size_t>(
                    4u, static_cast<size_t>(rawDerivativeCount) - start);

                auto packedGrid = packDerivativeChunk(orderedGrids, start, componentCount);
                const std::string packedName = "L-" + std::to_string(expectedTotalLevels) +
                                               "-" + std::to_string(levelIndex) +
                                               "-D-" + std::to_string(packedGridCount) +
                                               "-" + std::to_string(packedIndex);
                packedGrid->setName(packedName);

                openvdb::CoordBBox bbox = packedGrid->evalActiveVoxelBoundingBox();
                openvdb::Coord bboxMin = bbox.min();
                openvdb::Coord bboxMax = bbox.max();
                openvdb::Coord dims = bbox.dim();

                auto nanoHandle = nanovdb::tools::openToNanoVDB(packedGrid);
                if (!nanoHandle) {
                    std::cerr << "Error: Failed to convert packed grid '" << packedName << "'\n";
                    return EXIT_FAILURE;
                }

                GridInput gridInput;
                gridInput.name = packedName;
                gridInput.data = nanoHandle.data();
                gridInput.sizeBytes = nanoHandle.size();
                gridInput.gridType = getDerivGridType(componentCount);
                gridInput.bboxMin[0] = bboxMin.x();
                gridInput.bboxMin[1] = bboxMin.y();
                gridInput.bboxMin[2] = bboxMin.z();
                gridInput.bboxMax[0] = bboxMax.x();
                gridInput.bboxMax[1] = bboxMax.y();
                gridInput.bboxMax[2] = bboxMax.z();
                gridInput.dims[0] = static_cast<uint32_t>(dims.x());
                gridInput.dims[1] = static_cast<uint32_t>(dims.y());
                gridInput.dims[2] = static_cast<uint32_t>(dims.z());

                std::cout << "  Packed " << packedName << " as "
                          << packedGridTypeName(componentCount)
                          << " from derivatives [" << start << ", "
                          << (start + componentCount - 1) << "]"
                          << ", NanoVDB size: " << nanoHandle.size() << " bytes\n";

                nanoHandles.push_back(std::move(nanoHandle));
                gridInputs.push_back(gridInput);
            }
        }

        if (gridInputs.empty()) {
            std::cerr << "Error: No packed grids were successfully converted\n";
            return EXIT_FAILURE;
        }

        std::cout << "Successfully created " << gridInputs.size()
                  << " packed NanoVDB grid(s)\n\n";
        std::cout << "============================================\n";

        // Write the bundle using the new format
        if (!writeNanoVdbBundle(outputFile, gridInputs, switchOrderLevels)) {
            std::cerr << "Error: Failed to write NanoVDB bundle\n";
            return EXIT_FAILURE;
        }

        std::cout << "============================================\n";
        std::cout << "Export completed successfully!\n";
        std::cout << "============================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Error: Unknown error occurred\n";
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}