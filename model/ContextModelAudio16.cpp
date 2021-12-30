#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelAudio16 : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelAudio16(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models){
    const bool useLSTM = shared->GetOptionUseLSTM();
    m = mf->createMixer (
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS +
      Audio16BitModel::MIXERINPUTS +
      RecordModel::MIXERINPUTS + 
      (useLSTM ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + 
      Audio16BitModel::MIXERCONTEXTS +
      RecordModel::MIXERCONTEXTS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Audio16BitModel::MIXERCONTEXTSETS +
      RecordModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTSETS : 0)
      ,
      (useLSTM ? 1 : 0)
    );
    m->setScaleFactor(1024, 128);
  }

  void setParam(int blockInfo) {
    Audio16BitModel& audio16BitModel = models->audio16BitModel();
    audio16BitModel.setParam(blockInfo);
    uint32_t fixedRecordLenght = (audio16BitModel.stereo + 1) * 2;
    RecordModel& recordModel = models->recordModel();
    recordModel.setParam(fixedRecordLenght); // 2 or 4
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

    Audio16BitModel& audio16BitModel = models->audio16BitModel();
    audio16BitModel.mix(*m);

    RecordModel& recordModel = models->recordModel();
    recordModel.mix(*m);

    return m->p();

  }

  ~ContextModelAudio16() {
    delete m;
  }

};