#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelAudio8 {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelAudio8(Shared* const sh, Models* const models) : shared(sh), models(models) {
    MixerFactory mf{};
    m = mf.createMixer (
      sh,
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS +
      Audio8BitModel::MIXERINPUTS+
      RecordModel::MIXERINPUTS + 
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + 
      Audio8BitModel::MIXERCONTEXTS +
      RecordModel::MIXERCONTEXTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Audio8BitModel::MIXERCONTEXTSETS +
      RecordModel::MIXERCONTEXTSETS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTSETS : 0)
    );
    m->setScaleFactor(850, 140); //800-900, 140
  }


  int p(int blockInfo) {

    m->add(256); //network bias

    NormalModel& normalModel = models->normalModel();
    normalModel.mix(*m);

    MatchModel& matchModel = models->matchModel();
    matchModel.mix(*m);

    //??
    if ((shared->options & OPTION_LSTM) != 0u) {
      LstmModel<>& lstmModel = models->lstmModel();
      lstmModel.mix(*m);
    }

    Audio8BitModel& audio8BitModel = models->audio8BitModel();
    audio8BitModel.setParam(blockInfo);
    audio8BitModel.mix(*m);

    int isStereo = (blockInfo & 1);
    shared->State.rLength = isStereo + 1;
    RecordModel& recordModel = models->recordModel();
    recordModel.mix(*m);

    return m->p();
  }

  ~ContextModelAudio8() {
    delete m;
  }

};