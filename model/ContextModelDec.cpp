#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelDec {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelDec(Shared* const sh, Models* const models) : shared(sh), models(models) {
    MixerFactory mf{};
    m = mf.createMixer (
      sh,
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + SparseMatchModel::MIXERINPUTS +
      SparseModel::MIXERINPUTS + SparseBitModel::MIXERINPUTS + ChartModel::MIXERINPUTS + RecordModel::MIXERINPUTS +
      TextModel::MIXERINPUTS + WordModel::MIXERINPUTS + IndirectModel::MIXERINPUTS +
      ExeModel::MIXERINPUTS +
      DECAlphaModel::MIXERINPUTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + NormalModel::MIXERCONTEXTS_POST + SparseMatchModel::MIXERCONTEXTS +
      SparseModel::MIXERCONTEXTS + SparseBitModel::MIXERCONTEXTS + ChartModel::MIXERCONTEXTS + RecordModel::MIXERCONTEXTS +
      TextModel::MIXERCONTEXTS + WordModel::MIXERCONTEXTS + IndirectModel::MIXERCONTEXTS +
      DECAlphaModel::MIXERCONTEXTS +
      ExeModel::MIXERCONTEXTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + NormalModel::MIXERCONTEXTSETS_POST + SparseMatchModel::MIXERCONTEXTSETS +
      SparseModel::MIXERCONTEXTSETS + SparseBitModel::MIXERCONTEXTSETS + ChartModel::MIXERCONTEXTSETS + RecordModel::MIXERCONTEXTSETS +
      TextModel::MIXERCONTEXTSETS + WordModel::MIXERCONTEXTSETS + IndirectModel::MIXERCONTEXTSETS +
      DECAlphaModel::MIXERCONTEXTSETS +
      ExeModel::MIXERCONTEXTSETS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTSETS : 0)
    );
    m->setScaleFactor(1800, 60);
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
    shared->State.rLength = 16;
    RecordModel& recordModel = models->recordModel();
    recordModel.mix(*m);
    TextModel& textModel = models->textModel();
    textModel.mix(*m);
    WordModel& wordModel = models->wordModel();
    wordModel.mix(*m);
    IndirectModel& indirectModel = models->indirectModel();
    indirectModel.mix(*m);
    DECAlphaModel& decAlphaModel = models->decAlphaModel();
    decAlphaModel.mix(*m);
    ExeModel& exeModel = models->exeModel();
    exeModel.mix(*m);

    return m->p();
  }

  ~ContextModelDec() {
    delete m;
  }

};