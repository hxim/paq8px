#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelDec : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelDec(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models) {
    const bool useLSTM = shared->GetOptionUseLSTM();
    m = mf->createMixer (
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + SparseMatchModel::MIXERINPUTS +
      SparseModel::MIXERINPUTS + SparseBitModel::MIXERINPUTS + ChartModel::MIXERINPUTS + RecordModel::MIXERINPUTS +
      TextModel::MIXERINPUTS + WordModel::MIXERINPUTS_BIN + IndirectModel::MIXERINPUTS +
      ExeModel::MIXERINPUTS +
      DECAlphaModel::MIXERINPUTS +
      (useLSTM ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + NormalModel::MIXERCONTEXTS_POST + SparseMatchModel::MIXERCONTEXTS +
      SparseModel::MIXERCONTEXTS + SparseBitModel::MIXERCONTEXTS + ChartModel::MIXERCONTEXTS + RecordModel::MIXERCONTEXTS +
      TextModel::MIXERCONTEXTS + WordModel::MIXERCONTEXTS + IndirectModel::MIXERCONTEXTS +
      DECAlphaModel::MIXERCONTEXTS +
      ExeModel::MIXERCONTEXTS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + NormalModel::MIXERCONTEXTSETS_POST + SparseMatchModel::MIXERCONTEXTSETS +
      SparseModel::MIXERCONTEXTSETS + SparseBitModel::MIXERCONTEXTSETS + ChartModel::MIXERCONTEXTSETS + RecordModel::MIXERCONTEXTSETS +
      TextModel::MIXERCONTEXTSETS + WordModel::MIXERCONTEXTSETS + IndirectModel::MIXERCONTEXTSETS +
      DECAlphaModel::MIXERCONTEXTSETS +
      ExeModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTSETS : 0)
      ,
      (useLSTM ? 1 : 0)
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

    const bool useLSTM = shared->GetOptionUseLSTM();
    if (useLSTM) {
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
    TextModel& textModel = models->textModel();
    textModel.mix(*m);
    WordModel& wordModel = models->wordModel();
    wordModel.mix(*m);
    IndirectModel& indirectModel = models->indirectModel();
    indirectModel.mix(*m);
    DECAlphaModel& decAlphaModel = models->decAlphaModel();
    decAlphaModel.mix(*m);

    //exemodel must be the last one
    ExeModel& exeModel = models->exeModel();
    exeModel.mix(*m);

    return m->p();
  }

  ~ContextModelDec() {
    delete m;
  }

};