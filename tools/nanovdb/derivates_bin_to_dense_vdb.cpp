// ============================================================================
// NanoVDB Derivative Bundle to Dense VDB Converter
// ============================================================================
// This program reads a binary file containing NanoVDB grids with multi-level
// derivative data and converts it to a dense OpenVDB float grid.
//
// The input binary format stores Taylor polynomial coefficients in a 
// multi-resolution hierarchy. This program reconstructs the full scalar
// field by evaluating the Taylor polynomial at each voxel location.
// ============================================================================

#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <memory>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// ============================================================================
// NanoVDB Derivative Bundle Format - File Structures
// ============================================================================
// Binary layout:
//   1. FileHeader (64 bytes, aligned to 32)
//   2. LevelHeader[levelCount]
//   3. GridHeader[gridCount]
//   4. NanoVDB grid payloads (each 32-byte aligned)
// ============================================================================

struct DerivFileHeader {
    uint32_t magic;              // Magic number: 0x4E56444D ('NVDM')
    uint32_t version;            // File format version
    uint32_t payloadAlignment;   // Alignment requirement for grid payloads
    uint32_t levelCount;         // Number of resolution levels
    uint32_t gridCount;          // Total number of grids across all levels
    uint32_t reserved1;
    uint64_t levelTableOffset;   // Byte offset to LevelHeader array
    uint64_t gridTableOffset;    // Byte offset to GridHeader array
    uint64_t payloadBlockOffset; // Byte offset to first grid payload
    uint64_t totalFileSize;      // Total file size in bytes
    uint64_t reserved2;
};

struct DerivLevelHeader {
    uint32_t levelIndex;         // Level index (0-based, 0=finest)
    uint32_t derivativeCount;    // Number of derivative grids in this level
    uint32_t firstGridIndex;     // Index of first grid in GridHeader array
    uint32_t reserved;
};

struct DerivGridHeader {
    uint32_t levelIndex;                // Which level this grid belongs to
    uint32_t derivativeIndex;           // Derivative index within level (0-based)
    uint32_t derivativeCountInLevel;    // Total derivatives in this level
    uint32_t reserved1;
    uint64_t payloadOffset;             // Byte offset to grid payload from file start
    uint64_t payloadSize;               // Size of grid payload in bytes
    int32_t  bboxMin[3];                // Bounding box min in index space
    int32_t  bboxMax[3];                // Bounding box max in index space
    uint32_t dims[3];                   // Grid dimensions [x, y, z]
    uint32_t reserved2;
    char     name[56];                  // Grid name for debugging
};

// ============================================================================
// Taylor Polynomial Basis Functions
// ============================================================================
// Computes the basis function value for a given derivative index at a local
// position (px, py, pz) relative to voxel center.
//
// The Taylor series reconstruction is:
//   f(x) = Σ (coefficient[i] * basisValue[i](px, py, pz))
//
// Basis functions include factorial normalization from Taylor expansion:
//   f(x + h) = f(x) + f'(x)*h/1! + f''(x)*h²/2! + f'''(x)*h³/3! + ...
// ============================================================================

inline double derivBasisValue(int derivIdx, double px, double py, double pz)
{
    switch (derivIdx) {
        // ====================================================================
        // 0th order: constant term (function value)
        // ====================================================================
        case 0:  return 1.0;
        
        // ====================================================================
        // 1st order: linear terms (first derivatives)
        // ∂f/∂x, ∂f/∂y, ∂f/∂z
        // ====================================================================
        case 1:  return px;
        case 2:  return py;
        case 3:  return pz;
        
        // ====================================================================
        // 2nd order: pure quadratic terms (second derivatives)
        // ∂²f/∂x², ∂²f/∂y², ∂²f/∂z²
        // Divided by 2! = 2
        // ====================================================================
        case 4:  return px * px * 0.5;
        case 5:  return py * py * 0.5;
        case 6:  return pz * pz * 0.5;
        
        // ====================================================================
        // 2nd order: mixed terms (cross derivatives)
        // ∂²f/∂x∂y, ∂²f/∂x∂z, ∂²f/∂y∂z
        // ====================================================================
        case 7:  return px * py;
        case 8:  return px * pz;
        case 9:  return py * pz;
        
        // ====================================================================
        // 3rd order: pure cubic terms (third derivatives)
        // ∂³f/∂x³, ∂³f/∂y³, ∂³f/∂z³
        // Divided by 3! = 6
        // ====================================================================
        case 10: return px * px * px * (1.0 / 6.0);
        case 11: return py * py * py * (1.0 / 6.0);
        case 12: return pz * pz * pz * (1.0 / 6.0);
        
        // ====================================================================
        // 3rd order: mixed terms with one squared component
        // ∂³f/∂x²∂y, ∂³f/∂x²∂z, ∂³f/∂y²∂x, ∂³f/∂y²∂z, ∂³f/∂z²∂x, ∂³f/∂z²∂y
        // Divided by 2! for the squared term
        // ====================================================================
        case 13: return px * px * py * 0.5;
        case 14: return px * px * pz * 0.5;
        case 15: return py * py * px * 0.5;
        case 16: return py * py * pz * 0.5;
        case 17: return pz * pz * px * 0.5;
        case 18: return pz * pz * py * 0.5;
        
        // ====================================================================
        // 3rd order: fully mixed term
        // ∂³f/∂x∂y∂z
        // ====================================================================
        case 19: return px * py * pz;
        
        // ====================================================================
        // 4th order: pure quartic terms
        // ∂⁴f/∂x⁴, ∂⁴f/∂y⁴, ∂⁴f/∂z⁴
        // Divided by 4! = 24
        // ====================================================================
        case 20: return px * px * px * px / 24.0;
        case 21: return py * py * py * py / 24.0;
        case 22: return pz * pz * pz * pz / 24.0;
        
        // ====================================================================
        // 4th order: mixed cubic-linear terms
        // ∂⁴f/∂x³∂y, ∂⁴f/∂x³∂z, ∂⁴f/∂x∂y³, ∂⁴f/∂y³∂z, ∂⁴f/∂z³∂x, ∂⁴f/∂z³∂y
        // Divided by 3! = 6 for the cubic term
        // ====================================================================
        case 23: return px * px * px * py / 6.0;
        case 24: return px * px * px * pz / 6.0;
        case 25: return px * py * py * py / 6.0; // Note: py³*px
        case 26: return py * py * py * pz / 6.0;
        case 27: return pz * pz * pz * px / 6.0;
        case 28: return pz * pz * pz * py / 6.0;
        
        // ====================================================================
        // 4th order: mixed quadratic-quadratic terms
        // ∂⁴f/∂x²∂y², ∂⁴f/∂x²∂z², ∂⁴f/∂y²∂z²
        // Divided by 2! * 2! = 4
        // ====================================================================
        case 29: return px * px * py * py / 4.0;
        case 30: return px * px * pz * pz / 4.0;
        case 31: return py * py * pz * pz / 4.0;
        
        // ====================================================================
        // 4th order: mixed quadratic-linear-linear terms
        // ∂⁴f/∂x²∂y∂z, ∂⁴f/∂y²∂x∂z, ∂⁴f/∂z²∂x∂y
        // Divided by 2! = 2 for the squared term
        // ====================================================================
        case 32: return px * px * py * pz / 2.0;
        case 33: return py * py * px * pz / 2.0;
        case 34: return pz * pz * px * py / 2.0;
        
        default: return 0.0;
    }
}

// ============================================================================
// Helper: Get NanoVDB Grid Pointer from Binary Data
// ============================================================================
template<typename T>
const nanovdb::NanoGrid<T>* getDerivGridPtr(
    const uint8_t* base, 
    const DerivGridHeader& gh)
{
    return reinterpret_cast<const nanovdb::NanoGrid<T>*>(base + gh.payloadOffset);
}

// ============================================================================
// Main Conversion Function: Reconstruct Value at World Position
// ============================================================================
// This function evaluates the multi-resolution Taylor polynomial at a given
// world-space position (x, y, z). It iterates through all levels, starting
// from the coarsest, and returns the reconstructed value from the first
// level that has non-zero derivative data at that location.
//
// Algorithm:
// 1. Read file header and locate level/grid tables
// 2. For each level (coarsest to finest):
//    a. Convert world position to index space
//    b. Find containing voxel
//    c. Compute local offset from voxel center (normalized by level 0 voxel size)
//    d. Accumulate: sum(coefficient[i] * basisValue[i](offset))
//    e. If any coefficient is non-zero, return the result
// 3. Return 0 if no level has data
// ============================================================================
template<typename T>
float reconstructValueAtPosition(
    const uint8_t* fileData,
    const DerivFileHeader* fh,
    const DerivLevelHeader* levelTable,
    const DerivGridHeader* gridTable,
    double x, double y, double z)
{
    const uint32_t levelCount = fh->levelCount;
    if (levelCount == 0) return 0.0f;

    // ------------------------------------------------------------------------
    // Get reference voxel size from level 0 (finest level)
    // This is used to normalize the local offsets for all levels
    // ------------------------------------------------------------------------
    const DerivLevelHeader& level0Header = levelTable[0];
    const DerivGridHeader& level0Grid0Header = gridTable[level0Header.firstGridIndex];
    const nanovdb::NanoGrid<T>* level0Grid0 = getDerivGridPtr<T>(fileData, level0Grid0Header);
    
    const double level0_voxel_size0 = level0Grid0->voxelSize()[0];
    const double level0_voxel_size1 = level0Grid0->voxelSize()[1];
    const double level0_voxel_size2 = level0Grid0->voxelSize()[2];

    // ------------------------------------------------------------------------
    // Iterate through levels (typically coarsest to finest)
    // First level with non-zero data wins
    // ------------------------------------------------------------------------
    for (uint32_t levelIdx = 0; levelIdx < levelCount; ++levelIdx) {
        const DerivLevelHeader& lh = levelTable[levelIdx];
        
        const uint32_t derivCount = lh.derivativeCount;
        const uint32_t firstGrid = lh.firstGridIndex;

        // Bounds check for safety
        if (firstGrid + derivCount > fh->gridCount) continue;

        // --------------------------------------------------------------------
        // Get first grid to establish index space mapping
        // All grids in a level share the same index space and transform
        // --------------------------------------------------------------------
        const DerivGridHeader& gh0 = gridTable[firstGrid];
        const nanovdb::NanoGrid<T>* grid0 = getDerivGridPtr<T>(fileData, gh0);
        
        // Convert world coordinates to index space (continuous)
        const nanovdb::Vec3d ijk_d = grid0->worldToIndex(nanovdb::Vec3d(x, y, z));
        
        // Floor to get integer voxel coordinate
        const int32_t ix = static_cast<int32_t>(std::floor(ijk_d[0]));
        const int32_t iy = static_cast<int32_t>(std::floor(ijk_d[1]));
        const int32_t iz = static_cast<int32_t>(std::floor(ijk_d[2]));
        const nanovdb::Coord coord(ix, iy, iz);

        // --------------------------------------------------------------------
        // Calculate local offset from voxel center
        // Voxel center in index space is at (ix+0.5, iy+0.5, iz+0.5)
        // --------------------------------------------------------------------
        const nanovdb::Vec3d voxel_center_idx(ix + 0.5, iy + 0.5, iz + 0.5);
        const nanovdb::Vec3d voxel_center_world = grid0->indexToWorld(voxel_center_idx);
        
        // Offset in world space, normalized by level 0 voxel size
        // This normalization ensures Taylor series coefficients are comparable
        const double px = (x - voxel_center_world[0]) / level0_voxel_size0;
        const double py = (y - voxel_center_world[1]) / level0_voxel_size1;
        const double pz = (z - voxel_center_world[2]) / level0_voxel_size2;

        // --------------------------------------------------------------------
        // Accumulate Taylor polynomial: sum(coeff[i] * basis[i](px, py, pz))
        // --------------------------------------------------------------------
        double result = 0.0;
        bool hasNonZero = false;

        for (uint32_t d = 0; d < derivCount; ++d) {
            const DerivGridHeader& gh = gridTable[firstGrid + d];
            const nanovdb::NanoGrid<T>* grid = getDerivGridPtr<T>(fileData, gh);

            // Create accessor for efficient random access
            using AccessorT = nanovdb::ReadAccessor<T>;
            AccessorT acc(grid->tree().root());
            
            // Read coefficient value at this voxel
            const T coeff = acc.getValue(coord);

            // Check if coefficient is non-zero
            if (coeff != T(0.0f)) {
                hasNonZero = true;
                
                // Compute basis function value for this derivative
                const double basis = derivBasisValue(gh.derivativeIndex, px, py, pz);
                
                // Accumulate weighted contribution
                result += static_cast<double>(coeff) * basis;
            }
        }

        // If this level has non-zero data, use it (coarser levels take precedence)
        if (hasNonZero) {
            return static_cast<float>(result);
        }
    }

    // No data found in any level
    return 0.0f;
}

// ============================================================================
// Main Program
// ============================================================================
int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <output.vdb> <reference.vdb> [level_index]\n";
        std::cerr << "\n";
        std::cerr << "Converts NanoVDB derivative bundle to dense OpenVDB grid.\n";
        std::cerr << "  <reference.vdb> : VDB file to read bounding box from\n";
        std::cerr << "  Optional level_index: process only specific level (default: all levels)\n";
        return 1;
    }

    const char* inputFile = argv[1];
    const char* outputFile = argv[2];
    const char* referenceVdbFile = argv[3];
    int specificLevel = -1; // -1 means process all levels
    
    if (argc >= 5) {
        specificLevel = std::atoi(argv[4]);
    }

    try {
        // ====================================================================
        // Step 1: Read entire binary file into memory
        // ====================================================================
        std::cout << "Reading binary file: " << inputFile << std::endl;
        
        std::ifstream infile(inputFile, std::ios::binary | std::ios::ate);
        if (!infile) {
            std::cerr << "Error: Cannot open input file: " << inputFile << std::endl;
            return 1;
        }
        
        // Get file size and allocate buffer
        const std::streamsize fileSize = infile.tellg();
        infile.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> fileData(fileSize);
        if (!infile.read(reinterpret_cast<char*>(fileData.data()), fileSize)) {
            std::cerr << "Error: Failed to read file" << std::endl;
            return 1;
        }
        infile.close();
        
        std::cout << "  File size: " << fileSize << " bytes" << std::endl;

        // ====================================================================
        // Step 2: Parse file header and validate
        // ====================================================================
        const DerivFileHeader* fh = reinterpret_cast<const DerivFileHeader*>(fileData.data());
        
        std::cout << "\nFile Header:" << std::endl;
        std::cout << "  Magic: 0x" << std::hex << fh->magic << std::dec;
        if (fh->magic == 0x4E56444D) {
            std::cout << " (valid)" << std::endl;
        } else {
            std::cout << " (INVALID - expected 0x4E56444D)" << std::endl;
            return 1;
        }
        
        std::cout << "  Version: " << fh->version << std::endl;
        std::cout << "  Payload alignment: " << fh->payloadAlignment << " bytes" << std::endl;
        std::cout << "  Level count: " << fh->levelCount << std::endl;
        std::cout << "  Grid count: " << fh->gridCount << std::endl;
        std::cout << "  Level table offset: " << fh->levelTableOffset << std::endl;
        std::cout << "  Grid table offset: " << fh->gridTableOffset << std::endl;
        std::cout << "  Payload block offset: " << fh->payloadBlockOffset << std::endl;
        std::cout << "  Total file size: " << fh->totalFileSize << std::endl;

        if (fh->levelCount == 0) {
            std::cerr << "Error: No levels in file" << std::endl;
            return 1;
        }

        // ====================================================================
        // Step 3: Parse level and grid tables
        // ====================================================================
        const DerivLevelHeader* levelTable = 
            reinterpret_cast<const DerivLevelHeader*>(fileData.data() + fh->levelTableOffset);
        const DerivGridHeader* gridTable = 
            reinterpret_cast<const DerivGridHeader*>(fileData.data() + fh->gridTableOffset);

        std::cout << "\nLevel Information:" << std::endl;
        for (uint32_t i = 0; i < fh->levelCount; ++i) {
            const DerivLevelHeader& lh = levelTable[i];
            std::cout << "  Level " << lh.levelIndex 
                      << ": " << lh.derivativeCount << " derivatives"
                      << ", first grid index: " << lh.firstGridIndex << std::endl;
        }

        // ====================================================================
        // Step 4: Determine bounding box for output grid
        // Get from level 0 (finest level) or specified level
        // ====================================================================
        uint32_t refLevelIdx = (specificLevel >= 0 && specificLevel < (int)fh->levelCount) 
                               ? specificLevel : 0;
        
        const DerivLevelHeader& refLevel = levelTable[refLevelIdx];
        const DerivGridHeader& refGrid0 = gridTable[refLevel.firstGridIndex];
        const nanovdb::NanoGrid<float>* refNanoGrid = 
            getDerivGridPtr<float>(fileData.data(), refGrid0);

        std::cout << "\nReference Grid (Level " << refLevelIdx << ", Grid 0):" << std::endl;
        std::cout << "  Name: " << refGrid0.name << std::endl;
        std::cout << "  Dimensions: [" << refGrid0.dims[0] << ", " 
                  << refGrid0.dims[1] << ", " << refGrid0.dims[2] << "]" << std::endl;
        std::cout << "  BBox: [" << refGrid0.bboxMin[0] << ", " << refGrid0.bboxMin[1] 
                  << ", " << refGrid0.bboxMin[2] << "] to ["
                  << refGrid0.bboxMax[0] << ", " << refGrid0.bboxMax[1] 
                  << ", " << refGrid0.bboxMax[2] << "]" << std::endl;
        std::cout << "  Voxel size: [" << refNanoGrid->voxelSize()[0] << ", "
                  << refNanoGrid->voxelSize()[1] << ", " 
                  << refNanoGrid->voxelSize()[2] << "]" << std::endl;

        // ====================================================================
        // Step 5: Initialize OpenVDB
        // ====================================================================
        openvdb::initialize();

        // ====================================================================
        // Step 6: Read reference VDB file to get bounding box
        // ====================================================================
        std::cout << "\nReading reference VDB file: " << referenceVdbFile << std::endl;
        
        openvdb::io::File refFile(referenceVdbFile);
        refFile.open();
        
        openvdb::GridPtrVecPtr allGridsPtr = refFile.getGrids();
        refFile.close();
        
        if (!allGridsPtr || allGridsPtr->empty()) {
            std::cerr << "Error: No grids found in reference VDB file" << std::endl;
            return 1;
        }
        
        openvdb::GridBase::Ptr refBaseGrid = (*allGridsPtr)[0];
        std::cout << "  Using grid: " << refBaseGrid->getName() << std::endl;
        
        openvdb::FloatGrid::Ptr refGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(refBaseGrid);
        if (!refGrid) {
            std::cerr << "Error: Reference grid is not a FloatGrid" << std::endl;
            return 1;
        }
        
        // Get active voxel bounding box from reference grid
        auto refBBox = refGrid->evalActiveVoxelBoundingBox();
        
        std::cout << "  Reference grid bbox: " << refBBox << std::endl;
        
        // Compute dimensions from reference bbox
        struct Dims3 {
            int64_t x, y, z;
        };
        
        Dims3 refDims;
        refDims.x = refBBox.max().x() - refBBox.min().x() + 1;
        refDims.y = refBBox.max().y() - refBBox.min().y() + 1;
        refDims.z = refBBox.max().z() - refBBox.min().z() + 1;
        
        std::cout << "  Reference dimensions: [" << refDims.x << ", " << refDims.y << ", " << refDims.z << "]" << std::endl;
        
        // Use reference bbox as output bbox
        openvdb::CoordBBox outputBBox = refBBox;

        std::cout << "\nCreating dense output grid..." << std::endl;
        std::cout << "  Output bbox: " << outputBBox << std::endl;

        // Create dense grid with the same transform as reference grid
        openvdb::FloatGrid::Ptr outputGrid = openvdb::FloatGrid::create(0.0f);
        outputGrid->setName("reconstructed");
        
        // Create affine transform matching reference grid (scale + translation)
        nanovdb::Vec3d nanoOrigin = refNanoGrid->indexToWorld(nanovdb::Vec3d(0, 0, 0));
        openvdb::math::Mat4d mat = openvdb::math::Mat4d::identity();
        // Set scale (diagonal elements)
        mat[0][0] = refNanoGrid->voxelSize()[0];
        mat[1][1] = refNanoGrid->voxelSize()[1];
        mat[2][2] = refNanoGrid->voxelSize()[2];
        // Set translation
        mat.setTranslation(openvdb::Vec3d(nanoOrigin[0], nanoOrigin[1], nanoOrigin[2]));
        outputGrid->setTransform(openvdb::math::Transform::createLinearTransform(mat));

        // Get transform for world coordinate conversion
        const auto& transform = outputGrid->transform();
        const auto& indexToWorld = transform;

        // ====================================================================
        // Step 7: Fill dense grid by evaluating Taylor polynomial at each voxel
        // ====================================================================
        std::cout << "Reconstructing values..." << std::endl;
        
        const openvdb::Coord& bmin = outputBBox.min();
        const openvdb::Coord& bmax = outputBBox.max();
        
        const int64_t dimX = bmax.x() - bmin.x() + 1;
        const int64_t dimY = bmax.y() - bmin.y() + 1;
        const int64_t dimZ = bmax.z() - bmin.z() + 1;
        const int64_t totalVoxels = dimX * dimY * dimZ;
        
        std::cout << "  Grid dimensions: [" << dimX << ", " << dimY << ", " << dimZ << "]" << std::endl;
        std::cout << "  Total voxels to process: " << totalVoxels << std::endl;
        
#ifdef USE_OPENMP
        std::cout << "  Using OpenMP with " << omp_get_max_threads() << " threads" << std::endl;
#endif
        
        // Allocate dense array for reconstruction (XYZ layout)
        std::vector<float> denseData(totalVoxels, 0.0f);
        
        int64_t nonZeroVoxels = 0;
        const int64_t totalZSlices = dimZ;
        int64_t completedZSlices = 0;

#ifdef USE_OPENMP
        // Parallel version with OpenMP - fill dense array
        #pragma omp parallel
        {
            int64_t localNonZero = 0;
            
            #pragma omp for schedule(dynamic, 1)
            for (int64_t iz = 0; iz < dimZ; ++iz) {
                for (int64_t iy = 0; iy < dimY; ++iy) {
                    for (int64_t ix = 0; ix < dimX; ++ix) {
                        // Convert array index to grid index
                        int gridX = bmin.x() + ix;
                        int gridY = bmin.y() + iy;
                        int gridZ = bmin.z() + iz;
                        
                        // Convert index to world coordinates
                        // Use voxel center (add 0.5 to index)
                        openvdb::Vec3d indexPos(gridX + 0.5, gridY + 0.5, gridZ + 0.5);
                        openvdb::Vec3d worldPos = indexToWorld.indexToWorld(indexPos);

                        // Reconstruct value using Taylor polynomial
                        float value = reconstructValueAtPosition<float>(
                            fileData.data(), fh, levelTable, gridTable,
                            worldPos.x(),
                            worldPos.y(),
                            worldPos.z()
                        );

                        // Store in dense array (XYZ layout: x + dimX * (y + dimY * z))
                        int64_t arrayIdx = ix + dimX * (iy + dimY * iz);
                        denseData[arrayIdx] = value;
                        
                        if (value != 0.0f) {
                            localNonZero++;
                        }
                    }
                }
                
                // Progress reporting (every 5% of z-slices)
                #pragma omp critical
                {
                    completedZSlices++;
                    if (totalZSlices >= 20 && completedZSlices % (totalZSlices / 20) == 0) {
                        double percent = 100.0 * completedZSlices / totalZSlices;
                        std::cout << "  Progress: " << percent << "% (" 
                                  << completedZSlices << "/" << totalZSlices 
                                  << " z-slices completed)" << std::endl;
                    }
                }
            }
            
            // Accumulate thread-local non-zero counts
            #pragma omp atomic
            nonZeroVoxels += localNonZero;
        }
#else
        // Serial version without OpenMP - fill dense array
        int64_t processedVoxels = 0;
        const int64_t reportInterval = totalVoxels / 20; // Report every 5%
        
        for (int64_t iz = 0; iz < dimZ; ++iz) {
            for (int64_t iy = 0; iy < dimY; ++iy) {
                for (int64_t ix = 0; ix < dimX; ++ix) {
                    // Convert array index to grid index
                    int gridX = bmin.x() + ix;
                    int gridY = bmin.y() + iy;
                    int gridZ = bmin.z() + iz;
                    
                    // Convert index to world coordinates
                    // Use voxel center (add 0.5 to index)
                    openvdb::Vec3d indexPos(gridX + 0.5, gridY + 0.5, gridZ + 0.5);
                    openvdb::Vec3d worldPos = indexToWorld.indexToWorld(indexPos);

                    // Reconstruct value using Taylor polynomial
                    float value = reconstructValueAtPosition<float>(
                        fileData.data(), fh, levelTable, gridTable,
                        worldPos.x(),
                        worldPos.y(),
                        worldPos.z()
                    );

                    // Store in dense array (XYZ layout: x + dimX * (y + dimY * z))
                    int64_t arrayIdx = ix + dimX * (iy + dimY * iz);
                    denseData[arrayIdx] = value;
                    
                    if (value != 0.0f) {
                        nonZeroVoxels++;
                    }

                    processedVoxels++;
                    
                    // Progress reporting
                    if (reportInterval > 0 && processedVoxels % reportInterval == 0) {
                        double percent = 100.0 * processedVoxels / totalVoxels;
                        std::cout << "  Progress: " << percent << "% (" 
                                  << processedVoxels << "/" << totalVoxels 
                                  << " voxels, " << nonZeroVoxels << " non-zero)" << std::endl;
                    }
                }
            }
        }
#endif

        std::cout << "  Completed: " << totalVoxels << " voxels processed" << std::endl;
        std::cout << "  Non-zero voxels: " << nonZeroVoxels 
                  << " (" << (100.0 * nonZeroVoxels / totalVoxels) << "%)" << std::endl;

        // ====================================================================
        // Step 7b: Copy dense array to OpenVDB grid
        // ====================================================================
        std::cout << "\nCopying dense data to OpenVDB grid..." << std::endl;
        
        // Create bbox starting at origin for dense array
        openvdb::math::CoordBBox denseBBox(
            openvdb::Coord(0, 0, 0), 
            openvdb::Coord(dimX - 1, dimY - 1, dimZ - 1)
        );
        
        // Wrap dense array in OpenVDB Dense wrapper
        openvdb::tools::Dense<const float, openvdb::tools::LayoutXYZ> dense(denseBBox, denseData.data());
        
        // Copy from dense array to grid, shifting by bmin offset
        // We need to create a temporary grid at origin, then copy with offset
        openvdb::FloatGrid::Ptr tempGrid = openvdb::FloatGrid::create(0.0f);
        openvdb::tools::copyFromDense(dense, tempGrid->tree(), 0.0f);
        
        // Now copy to output grid with proper offset
        auto tempAccessor = tempGrid->getAccessor();
        auto outputAccessor = outputGrid->getAccessor();
        
        for (auto iter = tempGrid->cbeginValueOn(); iter; ++iter) {
            openvdb::Coord tempCoord = iter.getCoord();
            openvdb::Coord outputCoord(
                tempCoord.x() + bmin.x(),
                tempCoord.y() + bmin.y(),
                tempCoord.z() + bmin.z()
            );
            outputAccessor.setValue(outputCoord, *iter);
        }
        
        std::cout << "  Dense data copied to grid" << std::endl;

        // ====================================================================
        // Step 8: Write output VDB file
        // ====================================================================
        std::cout << "\nWriting output file: " << outputFile << std::endl;
        
        // Print transformation matrix for verification
        std::cout << "  Output grid transform matrix:" << std::endl;
        openvdb::math::Mat4d transformMat = outputGrid->transform().baseMap()->getAffineMap()->getMat4();
        for (int row = 0; row < 4; ++row) {
            std::cout << "    [";
            for (int col = 0; col < 4; ++col) {
                std::cout << transformMat[row][col];
                if (col < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "  Active voxel bbox: " << outputGrid->evalActiveVoxelBoundingBox() << std::endl;
        
#if 0        
        // Find min/max values in the output grid
        std::cout << "\n  Computing min/max values..." << std::endl;
        float minValue = FLT_MAX, maxValue = -FLT_MAX;
        
        auto minMaxAccessor = outputGrid->getAccessor();
#ifdef USE_OPENMP
        #pragma omp parallel for reduction(min:minValue) reduction(max:maxValue)
#endif
        for (int iz = bmin.z(); iz <= bmax.z(); ++iz) {
            for (int iy = bmin.y(); iy <= bmax.y(); ++iy) {
                for (int ix = bmin.x(); ix <= bmax.x(); ++ix) {
                    float value = minMaxAccessor.getValue(openvdb::Coord(ix, iy, iz));
                    minValue = fminf(minValue, value);
                    maxValue = fmaxf(maxValue, value);
                }
            }
        }
        
        std::cout << "  Min value: " << minValue << std::endl;
        std::cout << "  Max value: " << maxValue << std::endl;
#endif        
        
        openvdb::io::File file(outputFile);
        openvdb::GridPtrVec grids;
        grids.push_back(outputGrid);
        file.write(grids);
        file.close();

        std::cout << "Done!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
