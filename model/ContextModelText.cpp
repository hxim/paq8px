#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelText {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelText(Shared* const sh, Models* const models) : shared(sh), models(models) {
    MixerFactory mf{};
    m = mf.createMixer (
      sh,
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + SparseMatchModel::MIXERINPUTS +
      SparseModel::MIXERINPUTS + SparseBitModel::MIXERINPUTS + RecordModel::MIXERINPUTS + CharGroupModel::MIXERINPUTS +
      TextModel::MIXERINPUTS + WordModel::MIXERINPUTS + IndirectModel::MIXERINPUTS +
      DmcForest::MIXERINPUTS + NestModel::MIXERINPUTS + XMLModel::MIXERINPUTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + NormalModel::MIXERCONTEXTS_POST + SparseMatchModel::MIXERCONTEXTS +
      SparseModel::MIXERCONTEXTS + SparseBitModel::MIXERCONTEXTS + RecordModel::MIXERCONTEXTS + CharGroupModel::MIXERCONTEXTS +
      TextModel::MIXERCONTEXTS + WordModel::MIXERCONTEXTS + IndirectModel::MIXERCONTEXTS +
      DmcForest::MIXERCONTEXTS + NestModel::MIXERCONTEXTS + XMLModel::MIXERCONTEXTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + NormalModel::MIXERCONTEXTSETS_POST + SparseMatchModel::MIXERCONTEXTSETS +
      SparseModel::MIXERCONTEXTSETS + SparseBitModel::MIXERCONTEXTSETS + RecordModel::MIXERCONTEXTSETS + CharGroupModel::MIXERCONTEXTSETS +
      TextModel::MIXERCONTEXTSETS + WordModel::MIXERCONTEXTSETS + IndirectModel::MIXERCONTEXTSETS +
      DmcForest::MIXERCONTEXTSETS + NestModel::MIXERCONTEXTSETS + XMLModel::MIXERCONTEXTSETS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTSETS : 0)
    );
    m->setScaleFactor(940, 60);
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

    return m->p();
  }

  ~ContextModelText() {
    delete m;
  }

};