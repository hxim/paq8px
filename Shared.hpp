#pragma once

#include <cstdint>
#include "BlockType.hpp"
#include "RingBuffer.hpp"
#include "SIMDType.hpp"
#include "UpdateBroadcaster.hpp"

// helper #defines to access shared variables
#define INJECT_SHARED_buf   const RingBuffer<uint8_t> &buf=shared->buf;
#define INJECT_SHARED_pos   const uint32_t pos=shared->buf.getpos();
#define INJECT_SHARED_y     const uint8_t  y=shared->State.y;
#define INJECT_SHARED_c0    const uint8_t  c0=shared->State.c0;
#define INJECT_SHARED_c1    const uint8_t  c1=shared->State.c1;
#define INJECT_SHARED_bpos  const uint8_t  bpos=shared->State.bitPosition;
#define INJECT_SHARED_c4    const uint32_t c4=shared->State.c4;
#define INJECT_SHARED_c8    const uint32_t c8=shared->State.c8;

#define INJECT_SHARED_blockType const BlockType blockType=shared->State.blockType;
#define INJECT_SHARED_blockPos const uint32_t blockPos=shared->State.blockPos;

/**
 * Shared information by all the models and some other classes.
 */
struct Shared {
private:
    UpdateBroadcaster updateBroadcaster;

    // compression/decompression options (these flags are stored in the archive)
    static constexpr uint8_t OPTION_MULTIPLE_FILE_MODE = 1U;
    static constexpr uint8_t OPTION_TRAINEXE = 2U;
    static constexpr uint8_t OPTION_TRAINTXT = 4U;
    static constexpr uint8_t OPTION_ADAPTIVE = 8U;
    static constexpr uint8_t OPTION_SKIPRGB = 16U;
    static constexpr uint8_t OPTION_USELSTM = 32U;
    static constexpr uint8_t OPTION_TRAINLSTM = 64U;

    // block detection related options (these flags are not stored in the archive)
    static constexpr uint8_t OPTION_BRUTEFORCE_DEFLATE_DETECTION = 1U;
    static constexpr uint8_t OPTION_SKIP_BLOCK_DETECTION = 2U;

public:

    //Shared state and statistics (global)

    RingBuffer<uint8_t> buf; /**< Rotating input queue set by Predictor */
    uint8_t options = 0; /**< Compression options (bit field) */
    uint8_t detectionOptions = 0; /**< Block detection related options (bit field) */
    SIMDType chosenSimd = SIMDType::SIMD_NONE; /**< default value, will be overridden by the CPU dispatcher, and may be overridden from the command line */
    uint8_t level = 0; /**< level=0: no compression (only transformations), level=1..12 compress using less..more RAM */
    uint64_t mem = 0; /**< pre-calculated value of 65536 * 2^level */
    bool toScreen = true;

    // Getters
    bool GetOptionMultipleFileMode() const { return (options & OPTION_MULTIPLE_FILE_MODE) != 0; }
    bool GetOptionTrainExe() const { return (options & OPTION_TRAINEXE) != 0; }
    bool GetOptionTrainTxt() const { return (options & OPTION_TRAINTXT) != 0; }
    bool GetOptionAdaptiveLearningRate() const { return (options & OPTION_ADAPTIVE) != 0; }
    bool GetOptionSkipRGB() const { return (options & OPTION_SKIPRGB) != 0; }
    bool GetOptionUseLSTM() const { return (options & OPTION_USELSTM) != 0; }
    bool GetOptionTrainLSTM() const { return (options & OPTION_TRAINLSTM) != 0; }

    bool GetOptionBruteforceDeflateDetection() const { return (detectionOptions & OPTION_BRUTEFORCE_DEFLATE_DETECTION) != 0; }
    bool GetOptionSkipBlockDetection() const { return (detectionOptions & OPTION_SKIP_BLOCK_DETECTION) != 0; }

    // Setters
    void SetOptionMultipleFileMode() { options |= OPTION_MULTIPLE_FILE_MODE; }
    void SetOptionTrainExe() { options |= OPTION_TRAINEXE; }
    void SetOptionTrainTxt() { options |= OPTION_TRAINTXT; }
    void SetOptionAdaptiveLearningRate() { options |= OPTION_ADAPTIVE; }
    void SetOptionSkipRGB() { options |= OPTION_SKIPRGB; }
    void SetOptionUseLSTM() { options |= OPTION_USELSTM; }
    void SetOptionTrainLSTM() { options |= OPTION_TRAINLSTM; }

    void SetOptionBruteforceDeflateDetection() { detectionOptions |= OPTION_BRUTEFORCE_DEFLATE_DETECTION; }
    void SetOptionSkipBlockDetection() { detectionOptions |= OPTION_SKIP_BLOCK_DETECTION; }

    struct {

      //
      // Global state, used by most models, updated after every bit by update(y)
      // 

      uint8_t y = 0; /**< Last bit, 0 or 1 */
      uint8_t c0 = 1; /**< Last 0-7 bits of the partial byte with a leading 1 bit (1-255) */
      uint8_t c1 = 0; /**< Last whole byte, equals to c4&0xff or buf(1) */
      uint8_t bitPosition = 0; /**< Bits in c0 (0 to 7), in other words the position of the bit to be predicted (0=MSB) */
      uint32_t c4 = 0; /**< Last 4 whole bytes (buf(4)..buf(1)), packed.  Last byte is bits 0-7. */
      uint32_t c8 = 0; /**< Another 4 bytes (buf(8)..buf(5)) */
      uint32_t misses{}; //updated by the Predictor, used by SSE stage

      //written by BlockModel, used by many models 
      //see PredictorMain, ContextModel
      BlockType blockType; //current blockType
      int blockInfo;       //current block info (or -1 when none)
      uint32_t blockPos;   //relative position in block

      //written and used by BlockModel
      //see PredictorBlock, ContextModelBlock
      uint32_t blockTypeHistory;
      uint8_t blockStateID;

      //
      // State and statistics per model - set by the individual models
      // Order in appearance: models may use information from models that appears above them
      //

      //MatchModel
      struct {
        uint8_t length2;      //used by SSE stage and RecordModel
        uint8_t mode3;        //used by SSE stage 
        uint8_t mode5;        //used by SSE stage 
        uint8_t expectedByte; //used by SSE stage and RecordModel
      } Match{};

      //NormalModel
      struct {
        uint8_t order;
        uint64_t cxt[15]; // context hashes used by NormalModel and MatchModel
      } NormalModel{};

      //image models
      struct {
        struct {
          uint8_t WW, W, NN, N, Wp1, Np1;
        } pixels; //used by SSE stage
        uint8_t plane; //used by SSE stage
        uint8_t ctx; //used by SSE stage
      } Image{};

      //AudioModel
      uint8_t Audio{};

      //JpegModel
      struct {
        std::uint16_t state; // used by SSE stage
      } JPEG;

      //SparseMatchModel
      //SparseModel
      //SparseBitModel
      //ChartModel

      //RecordModel
      uint32_t rLength{};
      
      //CharGroupModel

      //TextModel
      struct {
        uint8_t characterGroup; //used by RecordModel, TextModel - Quantized partial byte as ASCII group
        uint8_t firstLetter; //used by SSE stage
        uint8_t mask; //used by SSE stage
        uint8_t order; //used by SSE stage
      } Text{};

      //WordModel
      struct {
        uint8_t order; //used by SSE stage
      } WordModel{};

      //IndirectModel
      //Dmcforest
      //NestModel
      //XMLModel
      //LinearPredictionModel
      
      //ExeModel
      struct {
        uint8_t state; // used by SSE stage
      } x86_64;

      //DECAlphaModel
      struct {
        uint8_t state; // used by SSE stage
        uint8_t bcount; // used by SSE stage
      } DEC;

    } State{};

    Shared();

    void init(uint8_t level, uint32_t bufMem = 0);
    void update(int y, bool isMissed);
    void reset();
    UpdateBroadcaster *GetUpdateBroadcaster() const;

private:

    /**
     * Copy constructor is private so that it cannot be called
     */
    Shared(Shared const & /*unused*/) {}

    /**
     * Assignment operator is private so that it cannot be called
     */
    auto operator=(Shared const & /*unused*/) -> Shared & { return *this; }

    /**
     * Determine if output is redirected
     * @return
     */
    static auto isOutputRedirected() -> bool;
};
