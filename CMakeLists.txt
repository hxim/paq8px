cmake_minimum_required(VERSION 3.10)
project(paq8px)

find_package(ZLIB)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

option(NATIVECPU "Whether to build for your cpu (vs. the general public)" OFF)
option(DISABLE_ZLIB "Whether to disable zlib" OFF)
option(NDEBUG "Whether to suppress asserts and array bound checks" ON)

if (NATIVECPU)
    add_definitions(-march=native -mtune=native)
else ()
    add_definitions(-march=nocona -mtune=generic)
endif (NATIVECPU)

if (DISABLE_ZLIB)
    add_definitions(-DDISABLE_ZLIB)
endif (DISABLE_ZLIB)

if (NDEBUG)
    add_definitions(-DNDEBUG)
endif (NDEBUG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O3 -floop-strip-mine -funroll-loops -ftree-vectorize -fgcse-sm -falign-loops=16")

add_executable(paq8px
               src/ProgramChecker.cpp
               src/paq8px.cpp
               src/MTFList.cpp
               src/Random.cpp
               src/String.cpp
               src/Predictor.cpp
               src/Models.cpp
               src/model/ExeModel.cpp
               src/APM1.cpp
               src/model/Image1BitModel.cpp
               src/model/Image4BitModel.cpp
               src/model/SparseModel.cpp
               src/Ilog.cpp
               src/model/ContextModel.cpp
               src/model/ContextModelAudio16.cpp
               src/model/ContextModelAudio8.cpp
               src/model/ContextModelDec.cpp
               src/model/ContextModelGeneric.cpp
               src/model/ContextModelImage1.cpp
               src/model/ContextModelImage24.cpp
               src/model/ContextModelImage4.cpp
               src/model/ContextModelImage8.cpp
               src/model/ContextModelJpeg.cpp
               src/model/ContextModelText.cpp
               src/SSE.cpp
               src/UpdateBroadcaster.cpp
               src/model/Audio8BitModel.cpp
               src/Shared.cpp
               src/text/TextModel.cpp
               src/file/File.cpp
               src/file/FileDisk.cpp
               src/file/FileName.cpp
               src/file/FileTmp.cpp
               src/file/ListOfFiles.cpp
               src/file/OpenFromMyFolder.cpp
               src/model/Audio16BitModel.cpp
               src/model/AudioModel.cpp
               src/model/CharGroupModel.cpp
               src/model/DmcForest.cpp
               src/model/DmcModel.cpp
               src/model/DmcNode.cpp
               src/model/Image24BitModel.cpp
               src/model/IndirectModel.cpp
               src/model/JpegModel.cpp
               src/model/LinearPredictionModel.cpp
               src/model/MatchModel.cpp
               src/model/NestModel.cpp
               src/model/NormalModel.cpp
               src/model/RecordModel.cpp
               src/model/SparseMatchModel.cpp
               src/model/WordModel.cpp
               src/model/XMLModel.cpp
               src/text/English.cpp
               src/text/EnglishStemmer.cpp
               src/text/French.cpp
               src/text/FrenchStemmer.cpp
               src/text/German.cpp
               src/text/GermanStemmer.cpp
               src/text/Stemmer.cpp
               src/text/Word.cpp
               src/text/WordEmbeddingDictionary.cpp
               src/StationaryMap.cpp
               src/StateTable.cpp
               src/StateMap.cpp
               src/SmallStationaryContextMap.cpp
               src/MixerFactory.cpp
               src/Mixer.cpp
               src/IndirectMap.cpp
               src/DummyMixer.cpp
               src/ContextMap2.cpp
               src/ContextMap.cpp
               src/APM.cpp
               src/APMPost.cpp
               src/AdaptiveMap.cpp
               src/filter/DecAlpha.cpp
               src/filter/LZWDictionary.cpp
               src/filter/TextParserStateInfo.cpp
               src/model/Image8BitModel.cpp
               src/Encoder.cpp
               src/model/WordModelInfo.cpp
               src/ArithmeticEncoder.cpp
               src/model/SparseBitModel.cpp
               src/model/ChartModel.cpp
               src/LargeStationaryMap.cpp
               src/BitCount.cpp
               src/BlockType.cpp
               src/Squash.cpp
               src/Stretch.cpp
               src/Block.cpp
               src/model/BlockModel.cpp
               src/model/ContextModelBlock.cpp
               src/PredictorBlock.cpp
               src/PredictorMain.cpp
               src/StateMapPair.cpp)
#add_executable(experiment src/test.cpp src/ProgramChecker.cpp src/MTFList.cpp src/Random.cpp src/String.cpp src/Predictor.cpp src/Models.cpp src/model/ExeModel.cpp src/APM1.cpp src/model/Image1BitModel.cpp src/model/Image4BitModel.cpp src/model/SparseModel.cpp src/Ilog.cpp src/model/ContextModel.cpp src/model/ContextModelAudio16.cpp src/model/ContextModelAudio8.cpp src/model/ContextModelDec.cpp src/model/ContextModelGeneric.cpp src/model/ContextModelImage1.cpp src/model/ContextModelImage24.cpp src/model/ContextModelImage4.cpp src/model/ContextModelImage8.cpp src/model/ContextModelJpeg.cpp src/model/ContextModelText.cpp src/SSE.cpp src/UpdateBroadcaster.cpp src/model/Audio8BitModel.cpp src/Shared.cpp src/text/TextModel.cpp src/file/File.cpp src/file/FileDisk.cpp src/file/FileName.cpp src/file/FileTmp.cpp src/file/ListOfFiles.cpp src/file/OpenFromMyFolder.cpp src/model/Audio16BitModel.cpp src/model/AudioModel.cpp src/model/CharGroupModel.cpp src/model/DmcForest.cpp src/model/DmcModel.cpp src/model/DmcNode.cpp src/model/Image24BitModel.cpp src/model/IndirectModel.cpp src/model/JpegModel.cpp src/model/LinearPredictionModel.cpp src/model/MatchModel.cpp src/model/NestModel.cpp src/model/NormalModel.cpp src/model/RecordModel.cpp src/model/SparseMatchModel.cpp src/model/WordModel.cpp src/model/XMLModel.cpp src/text/English.cpp src/text/EnglishStemmer.cpp src/text/French.cpp src/text/FrenchStemmer.cpp src/text/German.cpp src/text/GermanStemmer.cpp src/text/Stemmer.cpp src/text/Word.cpp src/text/WordEmbeddingDictionary.cpp src/StationaryMap.cpp src/StateTable.cpp src/StateMap.cpp src/SmallStationaryContextMap.cpp src/MixerFactory.cpp src/Mixer.cpp src/IndirectMap.cpp src/DummyMixer.cpp src/ContextMap2.cpp src/ContextMap.cpp src/APM.cpp src/APMPost.cpp src/AdaptiveMap.cpp src/filter/DecAlpha.cpp src/filter/LZWDictionary.cpp src/filter/TextParserStateInfo.cpp src/model/Image8BitModel.cpp src/Encoder.cpp src/model/WordModelInfo.cpp src/AritmeticEncoder.cpp src/model/SparseBitModel.cpp src/model/ChartModel.cpp src/LargeStationaryMap.cpp src/BitCount.cpp src/BlockType.cpp src/Squash.cpp src/Stretch.cpp src/Block.cpp src/model/BlockModel.cpp src/model/ContextModelBlock.cpp src/PredictorBlock.cpp src/PredictorMain.cpp src/StateMapPair.cpp)
#add_executable(train_bench src/bench/train.cpp)

if (supported)
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET paq8px PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    #set_property(TARGET experiment PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    #set_property(TARGET train_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif ()
#target_compile_options(paq8px PRIVATE
#        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
#        -Wall -Wextra -Weverything>
#        $<$<CXX_COMPILER_ID:MSVC>:
#        /W4>
#        $<$<CXX_COMPILER_ID:GNU>:
#        -Wall -Wextra>)

target_link_libraries(paq8px ${ZLIB_LIBRARIES})
