#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelJpeg : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelJpeg(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models) {
    const bool useLSTM = shared->GetOptionUseLSTM();
    m = mf->createMixer (
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + 
      JpegModel::MIXERINPUTS +
      SparseMatchModel::MIXERINPUTS +
      SparseModel::MIXERINPUTS + SparseBitModel::MIXERINPUTS + RecordModel::MIXERINPUTS + CharGroupModel::MIXERINPUTS +
      TextModel::MIXERINPUTS + WordModel::MIXERINPUTS_BIN + 
      LinearPredictionModel::MIXERINPUTS +
      (useLSTM ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + 
      JpegModel::MIXERCONTEXTS +
      SparseMatchModel::MIXERCONTEXTS +
      SparseModel::MIXERCONTEXTS + SparseBitModel::MIXERCONTEXTS + RecordModel::MIXERCONTEXTS + CharGroupModel::MIXERCONTEXTS +
      TextModel::MIXERCONTEXTS + WordModel::MIXERCONTEXTS + 
      LinearPredictionModel::MIXERCONTEXTS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      JpegModel::MIXERCONTEXTSETS +
      SparseMatchModel::MIXERCONTEXTSETS +
      SparseModel::MIXERCONTEXTSETS + SparseBitModel::MIXERCONTEXTSETS + RecordModel::MIXERCONTEXTSETS + CharGroupModel::MIXERCONTEXTSETS +
      TextModel::MIXERCONTEXTSETS + WordModel::MIXERCONTEXTSETS +
      LinearPredictionModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTSETS : 0)
      ,
      useLSTM ? 1 : 0
    );
  }

  int p() {

    m->add(256); //network bias

    NormalModel& normalModel = models->normalModel();
    normalModel.mix(*m);

    MatchModel& matchModel = models->matchModel();
    matchModel.mix(*m);

    //is it useful?
    const bool useLSTM = shared->GetOptionUseLSTM();
    if (useLSTM) {
      LstmModel<>& lstmModel = models->lstmModel();
      lstmModel.mix(*m);
    }

    JpegModel& jpegModel = models->jpegModel();
    if (jpegModel.mix(*m) != 0) {
      m->setScaleFactor(1024, 256); //850 for larger files, 1400 for smaller files - very sensitive
      return m->p();
    }
    else {
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
      LinearPredictionModel& linearPredictionModel = models->linearPredictionModel();
      linearPredictionModel.mix(*m);
      
      m->setScaleFactor(1200, 120); 
      return m->p();
    }

  }

  ~ContextModelJpeg() {
    delete m;
  }

};