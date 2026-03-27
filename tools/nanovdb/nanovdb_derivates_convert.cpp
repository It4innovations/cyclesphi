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

#define NANOVDB_USE_OPENVDB

#include <nanovdb/io/IO.h> // this is required to read (and write) NanoVDB files on the host
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

// ============================================================================
// Binary File Format for NanoVDB Multi-Level Derivative Grids
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

struct FileHeader {
    uint32_t magic;              // Magic number: 0x4E56444D
    uint32_t version;            // File format version
    uint32_t payloadAlignment;   // Alignment requirement for grid payloads (typically 32)
    uint32_t levelCount;         // Number of levels
    uint32_t gridCount;          // Total number of grids across all levels
    uint32_t reserved1;          // Reserved for future use
    uint64_t levelTableOffset;   // Byte offset to LevelHeader array
    uint64_t gridTableOffset;    // Byte offset to GridHeader array
    uint64_t payloadBlockOffset; // Byte offset to first grid payload
    uint64_t totalFileSize;      // Total file size in bytes
    uint64_t reserved2;          // Reserved for future use
};
static_assert(sizeof(FileHeader) == 64, "FileHeader must be 64 bytes");

struct LevelHeader {
    uint32_t levelIndex;         // Level index (0-based)
    uint32_t derivativeCount;    // Number of derivatives in this level
    uint32_t firstGridIndex;     // Index of first grid in global GridHeader array
    uint32_t reserved;           // Reserved for alignment
};
static_assert(sizeof(LevelHeader) == 16, "LevelHeader must be 16 bytes");

struct GridHeader {
    uint32_t levelIndex;                // Which level this grid belongs to
    uint32_t derivativeIndex;           // Derivative index within this level (0-based)
    uint32_t derivativeCountInLevel;    // Total number of derivatives in this level
    uint32_t reserved1;                 // Reserved for alignment
    uint64_t payloadOffset;             // Byte offset to grid payload (32-byte aligned)
    uint64_t payloadSize;               // Size of grid payload in bytes
    int32_t  bboxMin[3];                // Bounding box min (index space)
    int32_t  bboxMax[3];                // Bounding box max (index space)
    uint32_t dims[3];                   // Grid dimensions
    uint32_t reserved2;                 // Reserved for alignment
    char     name[56];                  // Grid name for debugging (null-terminated)
};
static_assert(sizeof(GridHeader) == 128, "GridHeader must be 128 bytes");

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
GridMetadata parseGridName(const std::string& name) {
    GridMetadata meta = {0, 0, 0, 0, false};
    
    // Expected format: L-XX-XX-D-XX-XX
    if (name.size() < 15) return meta; // Minimum length check
    
    // Find positions of delimiters
    if (name[0] != 'L' || name[1] != '-') return meta;
    
    size_t pos1 = name.find('-', 2);
    if (pos1 == std::string::npos) return meta;
    
    size_t pos2 = name.find('-', pos1 + 1);
    if (pos2 == std::string::npos) return meta;
    
    size_t posD = name.find("-D-", pos2);
    if (posD == std::string::npos) return meta;
    
    size_t pos3 = name.find('-', posD + 3);
    if (pos3 == std::string::npos) return meta;
    
    size_t pos4 = name.find('-', pos3 + 1);
    if (pos4 == std::string::npos) return meta;
    
    try {
        std::string totalLevelsStr = name.substr(2, pos1 - 2);
        std::string levelIndexStr = name.substr(pos1 + 1, pos2 - pos1 - 1);
        std::string derivCountStr = name.substr(posD + 3, pos3 - posD - 3);
        std::string derivIndexStr = name.substr(pos4 + 1);
        
        meta.totalLevels = std::stoul(totalLevelsStr);
        meta.levelIndex = std::stoul(levelIndexStr);
        meta.derivativeCountInLevel = std::stoul(derivCountStr);
        meta.derivativeIndex = std::stoul(derivIndexStr);
        meta.valid = true;
    } catch (...) {
        meta.valid = false;
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

// ============================================================================
// Main Export Function
// ============================================================================

bool writeNanoVdbBundle(const std::string& outputPath, const std::vector<GridInput>& grids) {
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
    currentOffset = sizeof(FileHeader);
    currentOffset = alignUp(currentOffset, PAYLOAD_ALIGNMENT);
    
    // LevelHeader array
    const uint64_t levelTableOffset = currentOffset;
    currentOffset += levelCount * sizeof(LevelHeader);
    
    // GridHeader array
    const uint64_t gridTableOffset = currentOffset;
    currentOffset += gridCount * sizeof(GridHeader);
    
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
    
    // Write FileHeader
    FileHeader fileHeader = {};
    fileHeader.magic = FILE_MAGIC;
    fileHeader.version = FILE_VERSION;
    fileHeader.payloadAlignment = PAYLOAD_ALIGNMENT;
    fileHeader.levelCount = levelCount;
    fileHeader.gridCount = gridCount;
    fileHeader.levelTableOffset = levelTableOffset;
    fileHeader.gridTableOffset = gridTableOffset;
    fileHeader.payloadBlockOffset = payloadBlockOffset;
    fileHeader.totalFileSize = totalFileSize;
    
    out.write(reinterpret_cast<const char*>(&fileHeader), sizeof(FileHeader));
    writePadding(out, sizeof(FileHeader), PAYLOAD_ALIGNMENT);
    
    // Write LevelHeader array
    for (const auto& level : levels) {
        LevelHeader lh = {};
        lh.levelIndex = level.levelIndex;
        lh.derivativeCount = level.derivativeCount;
        lh.firstGridIndex = level.firstGridIndex;
        out.write(reinterpret_cast<const char*>(&lh), sizeof(LevelHeader));
    }
    
    // Write GridHeader array
    for (uint32_t i = 0; i < gridCount; ++i) {
        const auto& gm = gridsMeta[i];
        
        GridHeader gh = {};
        gh.levelIndex = gm.meta.levelIndex;
        gh.derivativeIndex = gm.meta.derivativeIndex;
        gh.derivativeCountInLevel = gm.meta.derivativeCountInLevel;
        gh.payloadOffset = payloadOffsets[i];
        gh.payloadSize = gm.input.sizeBytes;
        std::memcpy(gh.bboxMin, gm.input.bboxMin, sizeof(gh.bboxMin));
        std::memcpy(gh.bboxMax, gm.input.bboxMax, sizeof(gh.bboxMax));
        std::memcpy(gh.dims, gm.input.dims, sizeof(gh.dims));
        
        // Copy name (truncate if necessary)
        std::strncpy(gh.name, gm.input.name.c_str(), sizeof(gh.name) - 1);
        gh.name[sizeof(gh.name) - 1] = '\0';
        
        out.write(reinterpret_cast<const char*>(&gh), sizeof(GridHeader));
    }
    
    // Align to payload block
    uint64_t currentPos = gridTableOffset + gridCount * sizeof(GridHeader);
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
        std::cerr << "Usage: " << argv[0] << " <input.vdb> <output.bin>\n";
        std::cerr << "  Reads OpenVDB file containing derivative grids and exports them\n";
        std::cerr << "  to a binary NanoVDB bundle with hierarchical headers.\n";
        std::cerr << "\n";
        std::cerr << "Grid naming convention: L-{totalLevels}-{levelIdx}-D-{derivCount}-{derivIdx}\n";
        std::cerr << "  Example: L-05-04-D-20-19\n";
        return EXIT_FAILURE;
    }

    const std::string inputFile = argv[1];
    const std::string outputFile = argv[2];

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

        // Convert each OpenVDB grid to NanoVDB and build GridInput array
        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> nanoHandles;
        std::vector<GridInput> gridInputs;
        
        nanoHandles.reserve(openvdbGrids->size());
        gridInputs.reserve(openvdbGrids->size());

        for (auto& openvdbGrid : *openvdbGrids) {
            const std::string gridName = openvdbGrid->getName();
            
            std::cout << "Converting grid: " << gridName << "\n";
            std::cout << "  Type: " << openvdbGrid->type() << "\n";
            
            // Get bounding box and dimensions
            openvdb::CoordBBox bbox = openvdbGrid->evalActiveVoxelBoundingBox();
            openvdb::Coord bboxMin = bbox.min();
            openvdb::Coord bboxMax = bbox.max();
            openvdb::Coord dims = bbox.dim();
            
            std::cout << "  BBox: (" << bboxMin.x() << "," << bboxMin.y() << "," << bboxMin.z() << ") -> "
                      << "(" << bboxMax.x() << "," << bboxMax.y() << "," << bboxMax.z() << ")\n";
            std::cout << "  Dims: " << dims.x() << " x " << dims.y() << " x " << dims.z() << "\n";
            
            // Convert OpenVDB grid to NanoVDB
            auto nanoHandle = nanovdb::tools::openToNanoVDB(openvdbGrid);
            
            if (!nanoHandle) {
                std::cerr << "  Error: Failed to convert grid: " << gridName << "\n";
                continue;
            }
            
            std::cout << "  NanoVDB size: " << nanoHandle.size() << " bytes\n";
            
            // Build GridInput
            GridInput gridInput;
            gridInput.name = gridName;
            gridInput.data = nanoHandle.data();
            gridInput.sizeBytes = nanoHandle.size();
            gridInput.bboxMin[0] = bboxMin.x();
            gridInput.bboxMin[1] = bboxMin.y();
            gridInput.bboxMin[2] = bboxMin.z();
            gridInput.bboxMax[0] = bboxMax.x();
            gridInput.bboxMax[1] = bboxMax.y();
            gridInput.bboxMax[2] = bboxMax.z();
            gridInput.dims[0] = static_cast<uint32_t>(dims.x());
            gridInput.dims[1] = static_cast<uint32_t>(dims.y());
            gridInput.dims[2] = static_cast<uint32_t>(dims.z());
            
            // Store handles (keep data alive) and inputs
            nanoHandles.push_back(std::move(nanoHandle));
            gridInputs.push_back(gridInput);
            
            std::cout << "\n";
        }

        if (gridInputs.empty()) {
            std::cerr << "Error: No grids were successfully converted\n";
            return EXIT_FAILURE;
        }

        std::cout << "Successfully converted " << gridInputs.size() << " grid(s) to NanoVDB\n\n";
        std::cout << "============================================\n";

        // Write the bundle using the new format
        if (!writeNanoVdbBundle(outputFile, gridInputs)) {
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