#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelAudio8 : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelAudio8(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models) {
    const bool useLSTM = shared->GetOptionUseLSTM();
    m = mf->createMixer (
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS +
      Audio8BitModel::MIXERINPUTS+
      RecordModel::MIXERINPUTS + 
      (useLSTM ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + 
      Audio8BitModel::MIXERCONTEXTS +
      RecordModel::MIXERCONTEXTS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Audio8BitModel::MIXERCONTEXTSETS +
      RecordModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTSETS : 0)
      ,
      (useLSTM ? 1 : 0)
    );
    m->setScaleFactor(850, 140); //800-900, 140
  }

  void setParam(int blockInfo) {
    Audio8BitModel& audio8BitModel = models->audio8BitModel();
    audio8BitModel.setParam(blockInfo);
    uint32_t fixedRecordLenght = audio8BitModel.stereo + 1;
    RecordModel& recordModel = models->recordModel();
    recordModel.setParam(fixedRecordLenght);  // 1 or 2
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

    Audio8BitModel& audio8BitModel = models->audio8BitModel();
    audio8BitModel.mix(*m);

    RecordModel& recordModel = models->recordModel();
    recordModel.mix(*m);

    return m->p();
  }

  ~ContextModelAudio8() {
    delete m;
  }

};