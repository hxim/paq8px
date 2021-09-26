#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelGeneric: public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelGeneric(Shared* const sh, Models* const models) : shared(sh), models(models) {
    MixerFactory mf{};
    m = mf.createMixer (
      sh,
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + SparseMatchModel::MIXERINPUTS +
      SparseModel::MIXERINPUTS + SparseBitModel::MIXERINPUTS + ChartModel::MIXERINPUTS +
      RecordModel::MIXERINPUTS + CharGroupModel::MIXERINPUTS +
      TextModel::MIXERINPUTS + WordModel::MIXERINPUTS_BIN + IndirectModel::MIXERINPUTS +
      DmcForest::MIXERINPUTS + NestModel::MIXERINPUTS + XMLModel::MIXERINPUTS +
      LinearPredictionModel::MIXERINPUTS + ExeModel::MIXERINPUTS +
      LstmModel<>::MIXERINPUTS
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + NormalModel::MIXERCONTEXTS_POST + SparseMatchModel::MIXERCONTEXTS +
      SparseModel::MIXERCONTEXTS + SparseBitModel::MIXERCONTEXTS + ChartModel::MIXERCONTEXTS +
      RecordModel::MIXERCONTEXTS + CharGroupModel::MIXERCONTEXTS +
      TextModel::MIXERCONTEXTS + WordModel::MIXERCONTEXTS + IndirectModel::MIXERCONTEXTS +
      DmcForest::MIXERCONTEXTS + NestModel::MIXERCONTEXTS + XMLModel::MIXERCONTEXTS +
      LinearPredictionModel::MIXERCONTEXTS + ExeModel::MIXERCONTEXTS + 
      LstmModel<>::MIXERCONTEXTS
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + NormalModel::MIXERCONTEXTSETS_POST + SparseMatchModel::MIXERCONTEXTSETS +
      SparseModel::MIXERCONTEXTSETS + SparseBitModel::MIXERCONTEXTSETS + ChartModel::MIXERCONTEXTSETS +
      RecordModel::MIXERCONTEXTSETS + CharGroupModel::MIXERCONTEXTSETS +
      TextModel::MIXERCONTEXTSETS + WordModel::MIXERCONTEXTSETS + IndirectModel::MIXERCONTEXTSETS +
      DmcForest::MIXERCONTEXTSETS + NestModel::MIXERCONTEXTSETS + XMLModel::MIXERCONTEXTSETS +
      LinearPredictionModel::MIXERCONTEXTSETS + ExeModel::MIXERCONTEXTSETS + 
      LstmModel<>::MIXERCONTEXTSETS
    );
    m->setScaleFactor(980, 90);
  }


  int p() {

    m->add(256); //network bias

    NormalModel& normalModel = models->normalModel();
    normalModel.mix(*m);
    normalModel.mixPost(*m);

    MatchModel& matchModel = models->matchModel();
    matchModel.mix(*m);

    if ((shared->options & OPTION_LSTM) != 0u) {
      LstmModel<>& lstmModel = models->lstmModel();
      lstmModel.mix(*m);
    }

    SparseMatchModel& sparseMatchModel = models->sparseMatchModel();
    sparseMatchModel.mix(*m);
    SparseBitModel& sparseBitModel = models->sparseBitModel();
    sparseBitModel.mix(*m);
    SparseModel& sparseModel = models->sparseModel();
    sparseModel.mix(*m);
    ChartModel& chartModel = models->chartModel();
    chartModel.mix(*m);
    RecordModel& recordModel = models->recordModel();
    recordModel.mix(*m);
    CharGroupModel& charGroupModel = models->charGroupModel();
    charGroupModel.mix(*m);
    TextModel& textModel = models->textModel();
    textModel.mix(*m);
    WordModel& wordModel = models->wordModel();
    wordModel.mix(*m);
    IndirectModel& indirectModel = models->indirectModel();
    indirectModel.mix(*m);
    DmcForest& dmcForest = models->dmcForest();
    dmcForest.mix(*m);
    NestModel& nestModel = models->nestModel();
    nestModel.mix(*m);
    XMLModel& xmlModel = models->xmlModel();
    xmlModel.mix(*m);

    LinearPredictionModel& linearPredictionModel = models->linearPredictionModel();
    linearPredictionModel.mix(*m);

    //exemodel must be the last
    ExeModel& exeModel = models->exeModel();
    exeModel.mix(*m);

    return m->p();
  }

  ~ContextModelGeneric() {
    delete m;
  }

};