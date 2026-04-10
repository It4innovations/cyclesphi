#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cfloat>
#include <algorithm>

// Global dimensions of the grid
struct Dims3 {
    int x, y, z;
};
Dims3 g_dims;

// Global bounding box offset for VDB grid access
struct Offset3 {
    int x, y, z;
};
Offset3 g_offset;

/**
 * @brief Normalizes a value to [0, 1] range with clamping
 * 
 * Maps a value from [minValue, maxValue] to [0, 1] range.
 * Values outside the range are clamped to [0, 1].
 * 
 * @param value The value to normalize
 * @param minValue Minimum value of the range
 * @param maxValue Maximum value of the range
 * @return Normalized value in [0, 1] range
 */
inline float normalizeToUnitRange(float value, float minValue, float maxValue)
{
    float normalized = (value - minValue) / (maxValue - minValue);
    
    // Clamp to [0, 1] range
    if (normalized < 0.0f)
        normalized = 0.0f;
    if (normalized > 1.0f)
        normalized = 1.0f;
    
    return normalized;
}

/**
 * @brief Normalizes a value to [0, 1] range without clamping
 * 
 * Maps a value from [minValue, maxValue] to [0, 1] range.
 * Does not clamp, so values can be outside [0, 1] if input is outside range.
 * 
 * @param value The value to normalize
 * @param minValue Minimum value of the range
 * @param maxValue Maximum value of the range
 * @return Normalized value (may be outside [0, 1])
 */
inline float normalizeWithoutClamping(float value, float minValue, float maxValue)
{
    return (value - minValue) / (maxValue - minValue);
}

/**
 * @brief Identity mapping - returns the value unchanged
 * 
 * @param value The value to return
 * @param minValue Unused (for interface compatibility)
 * @param maxValue Unused (for interface compatibility)
 * @return The original value unchanged
 */
inline float identityMapping(float value, float minValue, float maxValue)
{
    return value;
}

/**
 * @brief Statistics structure for VDB comparison
 * 
 * Stores min/max values and various quality metrics (MSE, SNR, PSNR, SSIM)
 * for three different normalization strategies.
 */
struct ComparisonStats
{
    // Min/max values from reference and compressed data
    float minValue{FLT_MAX}, maxValue{-FLT_MAX};
    float minVDB{FLT_MAX}, maxVDB{-FLT_MAX};
    
    // Strategy 1: Both normalized to reference range [0, 1] with clamping
    double mse{0.0};    // Mean Squared Error
    double snr{0.0};    // Signal-to-Noise Ratio
    double psnr{0.0};   // Peak Signal-to-Noise Ratio
    double ssim{0.0};   // Structural Similarity Index

    // Strategy 2: Reference normalized to [0, 1], compressed to its own range
    double mse2{0.0};
    double snr2{0.0};
    double psnr2{0.0};

    // Strategy 3: No normalization, raw values
    double mse3{0.0};
    double snr3{0.0};
    double psnr3{0.0};
};

/**
 * @brief Gets a value from a character array at given 3D coordinates
 * 
 * @param data Pointer to float array
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Value at (x, y, z)
 */
inline float getValue(const char* data, int x, int y, int z)
{
    const float* floatData = reinterpret_cast<const float*>(data);
    size_t index = size_t(x) + size_t(y) * g_dims.x + size_t(z) * g_dims.x * g_dims.y;
    return floatData[index];
}

/**
 * @brief Gets a value from an OpenVDB grid at given 3D coordinates
 * 
 * @tparam GridType Type of the OpenVDB grid
 * @param grid OpenVDB grid
 * @param x X coordinate (array index)
 * @param y Y coordinate (array index)
 * @param z Z coordinate (array index)
 * @return Value at (x, y, z)
 * @note Converts array indices to VDB coordinates using global offset
 */
template <typename GridType>
inline float getValue(std::shared_ptr<GridType> grid, int x, int y, int z)
{
    auto acc = grid->getConstAccessor();
    return acc.getValue(openvdb::Coord(x + g_offset.x, y + g_offset.y, z + g_offset.z));
}

/**
 * @brief Applies a 3D uniform box filter (moving average)
 * 
 * Computes the average value in a cubic window around each voxel.
 * Used for SSIM computation.
 * 
 * @param image Input 3D image as flattened 1D array
 * @param win_size Size of the cubic window (must be odd)
 * @return Filtered image as flattened 1D array
 */
std::vector<double> applyUniformFilter(const std::vector<double>& image, int win_size)
{
    std::vector<double> result(image.size(), 0.0);
    int half_win = win_size / 2;
    
    for (int i = half_win; i < g_dims.x - half_win; ++i) {
        for (int j = half_win; j < g_dims.y - half_win; ++j) {
            for (int k = half_win; k < g_dims.z - half_win; ++k) {
                double sum = 0.0;
                
                // Sum all values in the window
                for (int m = -half_win; m <= half_win; ++m) {
                    for (int n = -half_win; n <= half_win; ++n) {
                        for (int u = -half_win; u <= half_win; ++u) {
                            size_t im_id = size_t(i + m) + 
                                          size_t(j + n) * g_dims.x + 
                                          size_t(k + u) * g_dims.x * g_dims.y;
                            sum += image[im_id];
                        }
                    }
                }

                size_t res_id = size_t(i) + size_t(j) * g_dims.x + size_t(k) * g_dims.x * g_dims.y;
                result[res_id] = sum / (win_size * win_size * win_size);
            }
        }
    }
    return result;
}

/**
 * @brief Computes the Structural Similarity Index (SSIM) between two 3D images
 * 
 * SSIM measures the perceived quality difference between two images.
 * It considers luminance, contrast, and structure.
 * 
 * @tparam CompressedType Type of the compressed data (VDB grid or array)
 * @param reference Pointer to reference data
 * @param compressed Compressed data to compare against
 * @param stats Pre-computed statistics containing min/max values
 * @param data_range Dynamic range of the data (default: 1.0 for normalized)
 * @param win_size Window size for local statistics (default: 7)
 * @param K1 Algorithm constant for stability (default: 0.01)
 * @param K2 Algorithm constant for stability (default: 0.03)
 * @return SSIM value (typically in [0, 1], where 1 is identical)
 */
template <typename CompressedType>
double computeSSIM(const char* reference, CompressedType compressed, 
                   const ComparisonStats& stats, double data_range = 1.0, 
                   int win_size = 7, double K1 = 0.01, double K2 = 0.03)
{
    size_t N = g_dims.x * size_t(g_dims.y) * g_dims.z;
    std::vector<double> im1(N);
    std::vector<double> im2(N);

    // Convert both inputs to normalized double arrays
    for (int k = 0; k < g_dims.z; ++k) {
        for (int j = 0; j < g_dims.y; ++j) {
            for (int i = 0; i < g_dims.x; ++i) {
                float value0 = normalizeToUnitRange(getValue(reference, i, j, k), 
                                                     stats.minValue, stats.maxValue);
                float value1 = normalizeToUnitRange(getValue(compressed, i, j, k), 
                                                     stats.minValue, stats.maxValue);
                
                size_t id = size_t(i) + size_t(j) * g_dims.x + size_t(k) * g_dims.x * g_dims.y;
                im1[id] = value0;
                im2[id] = value1;
            }
        }
    }

    // SSIM constants for numerical stability
    double C1 = (K1 * data_range) * (K1 * data_range);
    double C2 = (K2 * data_range) * (K2 * data_range);
    
    // Compute local means
    auto mu1 = applyUniformFilter(im1, win_size);
    auto mu2 = applyUniformFilter(im2, win_size);
    
    // Compute squared means and cross-product
    std::vector<double> mu1_sq(N, 0.0), mu2_sq(N, 0.0), mu1_mu2(N, 0.0);
      
    for (int i = 0; i < g_dims.x; ++i) {
        for (int j = 0; j < g_dims.y; ++j) {
            for (int k = 0; k < g_dims.z; ++k) {
                size_t id = size_t(i) + size_t(j) * g_dims.x + size_t(k) * g_dims.x * g_dims.y;
                mu1_sq[id] = mu1[id] * mu1[id];
                mu2_sq[id] = mu2[id] * mu2[id];
                mu1_mu2[id] = mu1[id] * mu2[id];
            }
        }
    }
    
    // Compute element-wise squares for variance calculation
    std::vector<double> im1_sq(N), im2_sq(N), im1_im2(N);

    for (size_t i = 0; i < N; ++i) {
        im1_sq[i] = im1[i] * im1[i];
        im2_sq[i] = im2[i] * im2[i];
        im1_im2[i] = im1[i] * im2[i];
    }
    
    // Compute local variances and covariance
    auto sigma1_sq = applyUniformFilter(im1_sq, win_size);
    auto sigma2_sq = applyUniformFilter(im2_sq, win_size);
    auto sigma12 = applyUniformFilter(im1_im2, win_size);
      
    for (int i = 0; i < g_dims.x; ++i) {
        for (int j = 0; j < g_dims.y; ++j) {
            for (int k = 0; k < g_dims.z; ++k) {
                size_t id = size_t(i) + size_t(j) * g_dims.x + size_t(k) * g_dims.x * g_dims.y;
                sigma1_sq[id] -= mu1_sq[id];
                sigma2_sq[id] -= mu2_sq[id];
                sigma12[id] -= mu1_mu2[id];
            }
        }
    }
    
    // Compute SSIM for each voxel and average
    double ssim_sum = 0.0;
    size_t count = 0;

    for (int i = win_size / 2; i < g_dims.x - win_size / 2; ++i) {
        for (int j = win_size / 2; j < g_dims.y - win_size / 2; ++j) {
            for (int k = win_size / 2; k < g_dims.z - win_size / 2; ++k) {
                size_t id = size_t(i) + size_t(j) * g_dims.x + size_t(k) * g_dims.x * g_dims.y;

                double numerator = (2 * mu1_mu2[id] + C1) * (2 * sigma12[id] + C2);
                double denominator = (mu1_sq[id] + mu2_sq[id] + C1) * 
                                    (sigma1_sq[id] + sigma2_sq[id] + C2);
                ssim_sum += numerator / denominator;
                count++;
            }
        }
    }
    
    return ssim_sum / count;
}

/**
 * @brief Computes comprehensive comparison statistics between reference and compressed data
 * 
 * Calculates three sets of metrics using different normalization strategies:
 * 1. Both normalized to reference [0,1] range with clamping
 * 2. Each normalized to its own range
 * 3. Raw values without normalization
 * 
 * @tparam CompressedType Type of the compressed data (VDB grid or array)
 * @param reference Pointer to reference data
 * @param compressed Compressed data to compare against
 * @return ComparisonStats structure with all metrics
 */
template <typename CompressedType>
ComparisonStats computeComparisonStats(const char* reference, CompressedType compressed)
{
    ComparisonStats stats;

    // Find min/max values in both datasets
    // Use local variables for OpenMP reduction (struct members can't be reduced directly)
    float minValue = FLT_MAX, maxValue = -FLT_MAX;
    float minVDB = FLT_MAX, maxVDB = -FLT_MAX;
     
    for (int z = 0; z < g_dims.z; ++z) {
        for (int y = 0; y < g_dims.y; ++y) {
            for (int x = 0; x < g_dims.x; ++x) {
                float value0 = getValue(reference, x, y, z);
                minValue = fminf(minValue, value0);
                maxValue = fmaxf(maxValue, value0);

                float value1 = getValue(compressed, x, y, z);
                minVDB = fminf(minVDB, value1);
                maxVDB = fmaxf(maxVDB, value1);
            }
        }
    }
    
    // Assign the reduced values to the stats struct
    stats.minValue = minValue;
    stats.maxValue = maxValue;
    stats.minVDB = minVDB;
    stats.maxVDB = maxVDB;

    // Compute SSIM (uses normalized images, so data_range = 1.0)
    stats.ssim = computeSSIM(reference, compressed, stats, 1.0);

    // Accumulators for three different normalization strategies
    double sumSquared1{0.0}, sumSquaredErr1{0.0};
    double sumSquared2{0.0}, sumSquaredErr2{0.0};
    double sumSquared3{0.0}, sumSquaredErr3{0.0};

    // Compute error metrics for all three strategies
    for (int z = 0; z < g_dims.z; ++z) {
        for (int y = 0; y < g_dims.y; ++y) {
            for (int x = 0; x < g_dims.x; ++x) {
                // Strategy 1: Both normalized to reference [0,1] with clamping
                {
                    float value0 = normalizeToUnitRange(getValue(reference, x, y, z), 
                                                         stats.minValue, stats.maxValue);
                    float value1 = normalizeToUnitRange(getValue(compressed, x, y, z), 
                                                         stats.minValue, stats.maxValue);
                    double sqr = double(value0) * double(value0);
                    double diff = double(value0) - double(value1);
                    sumSquared1 += sqr;
                    sumSquaredErr1 += diff * diff;
                }

                // Strategy 2: Each normalized to its own range without clamping
                {
                    float value0 = normalizeWithoutClamping(getValue(reference, x, y, z), 
                                                            stats.minValue, stats.maxValue);
                    float value1 = normalizeWithoutClamping(getValue(compressed, x, y, z), 
                                                            stats.minVDB, stats.maxVDB);
                    double sqr = double(value0) * double(value0);
                    double diff = double(value0) - double(value1);
                    sumSquared2 += sqr;
                    sumSquaredErr2 += diff * diff;
                }
                
                // Strategy 3: Raw values without normalization
                {
                    float value0 = identityMapping(getValue(reference, x, y, z), 
                                                   stats.minValue, stats.maxValue);
                    float value1 = identityMapping(getValue(compressed, x, y, z), 
                                                   stats.minVDB, stats.maxVDB);
                    double sqr = double(value0) * double(value0);
                    double diff = double(value0) - double(value1);
                    sumSquared3 += sqr;
                    sumSquaredErr3 += diff * diff;
                }
            }
        }
    }
    
    size_t N = g_dims.x * size_t(g_dims.y) * g_dims.z;

    // Compute metrics for Strategy 1
    {
        stats.mse = sumSquaredErr1 / N;
        double signalMean = sumSquared1 / N;
        double noiseMean = stats.mse;
        
        if (noiseMean == 0.0) {
            stats.snr = INFINITY;
            stats.psnr = INFINITY;
        } else {
            stats.snr = 20.0 * log10(sqrt(signalMean) / sqrt(noiseMean));
            stats.psnr = 10.0 * log10(1.0 / noiseMean);
        }
    }

    // Compute metrics for Strategy 2
    {
        stats.mse2 = sumSquaredErr2 / N;
        double signalMean = sumSquared2 / N;
        double noiseMean = stats.mse2;
        
        if (noiseMean == 0.0) {
            stats.snr2 = INFINITY;
            stats.psnr2 = INFINITY;
        } else {
            stats.snr2 = 20.0 * log10(sqrt(signalMean) / sqrt(noiseMean));
            stats.psnr2 = 10.0 * log10(1.0 / noiseMean);
        }
    }

    // Compute metrics for Strategy 3
    {
        stats.mse3 = sumSquaredErr3 / N;
        double signalMean = sumSquared3 / N;
        double noiseMean = stats.mse3;
        
        if (noiseMean == 0.0) {
            stats.snr3 = INFINITY;
            stats.psnr3 = INFINITY;
        } else {
            stats.snr3 = 20.0 * log10(sqrt(signalMean) / sqrt(noiseMean));
            stats.psnr3 = 10.0 * log10(stats.maxValue * stats.maxValue / noiseMean);
        }
    }
    
    return stats;
}

/**
 * @brief Prints comparison statistics in a formatted table
 * 
 * @param stats Statistics structure to print
 * @param useScientific If true, use scientific notation for doubles; otherwise use default format
 */
void printStats(const ComparisonStats& stats, bool useScientific = true)
{
    // Set output format based on parameter
    if (useScientific) {
        std::cout << std::scientific << std::setprecision(6);
    } else {
        std::cout << std::defaultfloat << std::setprecision(6);
    }
    
    std::cout << "\n=== VDB Comparison Statistics ===\n\n";
    
    std::cout << "Data Range:\n";
    std::cout << "  Reference min/max : [" << stats.minValue << ", " << stats.maxValue << "]\n";
    std::cout << "  Compressed min/max: [" << stats.minVDB << ", " << stats.maxVDB << "]\n\n";
    
    std::cout << "Strategy 1 - Both normalized to reference [0,1] with clamping:\n";
    std::cout << "  MSE  : " << stats.mse << '\n';
    std::cout << "  SNR  : " << stats.snr << " dB\n";
    std::cout << "  PSNR : " << stats.psnr << " dB\n\n";
    
    std::cout << "Strategy 2 - Each normalized to its own range:\n";
    std::cout << "  MSE  : " << stats.mse2 << '\n';
    std::cout << "  SNR  : " << stats.snr2 << " dB\n";
    std::cout << "  PSNR : " << stats.psnr2 << " dB\n\n";

    std::cout << "Strategy 3 - Raw values without normalization:\n";
    std::cout << "  MSE  : " << stats.mse3 << '\n';
    std::cout << "  SNR  : " << stats.snr3 << " dB\n";
    std::cout << "  PSNR : " << stats.psnr3 << " dB\n\n";

    std::cout << "Structural Similarity:\n";
    std::cout << "  SSIM : " << stats.ssim << '\n';
    
    std::cout << "\n================================\n";
    
    // Reset to default format
    std::cout << std::defaultfloat;
}

/**
 * @brief Main function - Reads two VDB files and compares them
 */
int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <reference.vdb> <compressed.vdb>\n";
        std::cerr << "\nCompares two VDB files and outputs statistics:\n";
        std::cerr << "  - MSE (Mean Squared Error)\n";
        std::cerr << "  - SNR (Signal-to-Noise Ratio)\n";
        std::cerr << "  - PSNR (Peak Signal-to-Noise Ratio)\n";
        std::cerr << "  - SSIM (Structural Similarity Index)\n";
        return 1;
    }

    // Initialize OpenVDB library
    openvdb::initialize();

    try {
        // Read reference VDB file
        std::cout << "Loading reference: " << argv[1] << " ... " << std::flush;
        openvdb::io::File refFile(argv[1]);
        refFile.open();
        if (!refFile.isOpen()) {
            std::cerr << "\nError: Failed to open reference VDB file: " << argv[1] << std::endl;
            return 1;
        }
        openvdb::GridBase::Ptr baseRefGrid = refFile.readGrid(refFile.beginName().gridName());
        refFile.close();
        auto refGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseRefGrid);
        if (!refGrid) {
            std::cerr << "\nError: Reference grid is not a float grid" << std::endl;
            return 1;
        }
        openvdb::math::Mat4f ref_grid_matrix = refGrid->transform().baseMap()->getAffineMap()->getMat4();
        std::cout << "done\n";
        std::cout << "Reference grid matrix:\n" << ref_grid_matrix << "\n";

        // Read compressed VDB file
        std::cout << "Loading compressed: " << argv[2] << " ... " << std::flush;
        openvdb::io::File compFile(argv[2]);
        compFile.open();
        if (!compFile.isOpen()) {
            std::cerr << "\nError: Failed to open compressed VDB file: " << argv[2] << std::endl;
            return 1;
        }
        openvdb::GridBase::Ptr baseCompGrid = compFile.readGrid(compFile.beginName().gridName());
        compFile.close();
        auto compGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseCompGrid);
        if (!compGrid) {
            std::cerr << "\nError: Compressed grid is not a float grid" << std::endl;
            return 1;
        }
        openvdb::math::Mat4f comp_grid_matrix = compGrid->transform().baseMap()->getAffineMap()->getMat4();
        std::cout << "done\n";
        std::cout << "Compressed grid matrix:\n" << comp_grid_matrix << "\n";

        // Get grid dimensions from bounding boxes
        auto refBBox = refGrid->evalActiveVoxelBoundingBox();
        auto compBBox = compGrid->evalActiveVoxelBoundingBox();
        
        Dims3 refDims, compDims;
        refDims.x = refBBox.max().x() - refBBox.min().x() + 1;
        refDims.y = refBBox.max().y() - refBBox.min().y() + 1;
        refDims.z = refBBox.max().z() - refBBox.min().z() + 1;
        
        compDims.x = compBBox.max().x() - compBBox.min().x() + 1;
        compDims.y = compBBox.max().y() - compBBox.min().y() + 1;
        compDims.z = compBBox.max().z() - compBBox.min().z() + 1;
        
        // Check if dimensions match
        if (refDims.x != compDims.x || refDims.y != compDims.y || refDims.z != compDims.z) {
            std::cerr << "\nError: Grid dimensions do not match!\n";
            std::cerr << "  Reference : " << refDims.x << " x " << refDims.y << " x " << refDims.z << "\n";
            std::cerr << "  Compressed: " << compDims.x << " x " << compDims.y << " x " << compDims.z << "\n";
            return 1;
        }
        
        g_dims = refDims;
        
        // Set global bounding box offset for VDB grid access
        g_offset.x = compBBox.min().x();
        g_offset.y = compBBox.min().y();
        g_offset.z = compBBox.min().z();
        
        std::cout << "Grid dimensions: " << g_dims.x << " x " << g_dims.y << " x " << g_dims.z << "\n";
        std::cout << "Total voxels: " << (g_dims.x * size_t(g_dims.y) * g_dims.z) << "\n";
        std::cout << "Compressed grid offset: (" << g_offset.x << ", " << g_offset.y << ", " << g_offset.z << ")\n";

        // Convert reference VDB to dense float array
        std::cout << "Converting reference to dense array ... " << std::flush;
        size_t N = g_dims.x * size_t(g_dims.y) * g_dims.z;
        std::vector<float> refArray(N);
        auto refAcc = refGrid->getConstAccessor();
        
        for (int z = 0; z < g_dims.z; ++z) {
            for (int y = 0; y < g_dims.y; ++y) {
                for (int x = 0; x < g_dims.x; ++x) {
                    size_t idx = size_t(x) + size_t(y) * g_dims.x + size_t(z) * g_dims.x * g_dims.y;
                    openvdb::Coord coord(x + refBBox.min().x(), y + refBBox.min().y(), z + refBBox.min().z());
                    refArray[idx] = refAcc.getValue(coord);
                }
            }
        }
        std::cout << "done\n";

        // Compute and print statistics
        std::cout << "Computing statistics ... " << std::flush;
        auto stats = computeComparisonStats(reinterpret_cast<const char*>(refArray.data()), compGrid);
        std::cout << "done\n";
        
        printStats(stats);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
