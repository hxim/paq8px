cmake_minimum_required(VERSION 3.10)
project(paq8px)

find_package(ZLIB)
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

option(NATIVECPU "Whether to build for your cpu (vs. the general public)" OFF)
option(DISABLE_TEXTMODEL "Whether to disable TextModel and WordModel" OFF)
option(DISABLE_AUDIOMODEL "Whether to disable AudioModels" OFF)
option(DISABLE_ZLIB "Whether to disable zlib" OFF)
option(NDEBUG "Whether to suppress asserts and array bound checks" ON)
option(VERBOSE "Whether to print verbose debug information to screen" OFF)
option(HASHCONFIGCMD "Whether to support custom hash configuration" OFF)

if (NATIVECPU)
    add_definitions(-march=native -mtune=native)
else ()
    add_definitions(-march=nocona -mtune=generic)
endif (NATIVECPU)

if (DISABLE_TEXTMODEL)
    add_definitions(-DDISABLE_TEXTMODEL)
endif (DISABLE_TEXTMODEL)

if (DISABLE_AUDIOMODEL)
    add_definitions(-DDISABLE_AUDIOMODEL)
endif (DISABLE_AUDIOMODEL)

if (DISABLE_ZLIB)
    add_definitions(-DDISABLE_ZLIB)
endif (DISABLE_ZLIB)

if (NDEBUG)
    add_definitions(-DNDEBUG)
endif (NDEBUG)

if (VERBOSE)
    add_definitions(-DVERBOSE)
endif (VERBOSE)

if (HASHCONFIGCMD)
    add_definitions(-DHASHCONFIGCMD)
endif (HASHCONFIGCMD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O3 -floop-strip-mine -funroll-loops -ftree-vectorize -fgcse-sm -falign-loops=16")

add_executable(paq8px ProgramChecker.cpp paq8px.cpp MTFList.cpp Random.cpp String.cpp Predictor.cpp Models.cpp model/ExeModel.cpp APM1.cpp model/Image1BitModel.cpp model/Image4BitModel.cpp model/SparseModel.cpp Ilog.cpp model/ContextModel.cpp SSE.cpp UpdateBroadcaster.cpp model/Audio8BitModel.cpp Shared.cpp text/TextModel.cpp file/File.cpp file/FileDisk.cpp file/FileName.cpp file/FileTmp.cpp file/ListOfFiles.cpp file/OpenFromMyFolder.cpp model/Audio16BitModel.cpp model/AudioModel.cpp model/CharGroupModel.cpp model/DmcForest.cpp model/DmcModel.cpp model/DmcNode.cpp model/Image24BitModel.cpp model/IndirectModel.cpp model/JpegModel.cpp model/LinearPredictionModel.cpp model/MatchModel.cpp model/NestModel.cpp model/NormalModel.cpp model/RecordModel.cpp model/SparseMatchModel.cpp model/WordModel.cpp model/XMLModel.cpp text/English.cpp text/EnglishStemmer.cpp text/French.cpp text/FrenchStemmer.cpp text/German.cpp text/GermanStemmer.cpp text/Stemmer.cpp text/Word.cpp text/WordEmbeddingDictionary.cpp StationaryMap.cpp StateTable.cpp StateMap.cpp SmallStationaryContextMap.cpp MixerFactory.cpp Mixer.cpp IndirectMap.cpp DummyMixer.cpp ContextMap2.cpp ContextMap.cpp APM.cpp AdaptiveMap.cpp filter/DecAlpha.cpp filter/LZWDictionary.cpp filter/TextParserStateInfo.cpp model/Image8BitModel.cpp Encoder.cpp model/Info.cpp ArithmeticEncoder.cpp model/SparseBitModel.cpp LargeStationaryMap.cpp)
#add_executable(experiment test.cpp ProgramChecker.cpp MTFList.cpp Random.cpp String.cpp Predictor.cpp Models.cpp model/ExeModel.cpp APM1.cpp model/Image1BitModel.cpp model/Image4BitModel.cpp model/SparseModel.cpp Ilog.cpp model/ContextModel.cpp SSE.cpp UpdateBroadcaster.cpp model/Audio8BitModel.cpp Shared.cpp text/TextModel.cpp file/File.cpp file/FileDisk.cpp file/FileName.cpp file/FileTmp.cpp file/ListOfFiles.cpp file/OpenFromMyFolder.cpp model/Audio16BitModel.cpp model/AudioModel.cpp model/CharGroupModel.cpp model/DmcForest.cpp model/DmcModel.cpp model/DmcNode.cpp model/Image24BitModel.cpp model/IndirectModel.cpp model/JpegModel.cpp model/LinearPredictionModel.cpp model/MatchModel.cpp model/NestModel.cpp model/NormalModel.cpp model/RecordModel.cpp model/SparseMatchModel.cpp model/WordModel.cpp model/XMLModel.cpp text/English.cpp text/EnglishStemmer.cpp text/French.cpp text/FrenchStemmer.cpp text/German.cpp text/GermanStemmer.cpp text/Stemmer.cpp text/Word.cpp text/WordEmbeddingDictionary.cpp StationaryMap.cpp StateTable.cpp StateMap.cpp SmallStationaryContextMap.cpp MixerFactory.cpp Mixer.cpp IndirectMap.cpp DummyMixer.cpp ContextMap2.cpp ContextMap.cpp APM.cpp AdaptiveMap.cpp filter/LZWDictionary.cpp filter/TextParserStateInfo.cpp model/Image8BitModel.cpp Encoder.cpp model/Info.cpp AritmeticEncoder.cpp model/SparseBitModel.cpp LargeStationaryMap.cpp)
#add_executable(train_bench bench/train.cpp)

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
