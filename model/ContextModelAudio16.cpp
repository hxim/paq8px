#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelAudio16 {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelAudio16(Shared* const sh, Models* const models) : shared(sh), models(models) {
    MixerFactory mf{};
    m = mf.createMixer (
      sh,
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS +
      Audio16BitModel::MIXERINPUTS +
      RecordModel::MIXERINPUTS + 
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + 
      Audio16BitModel::MIXERCONTEXTS +
      RecordModel::MIXERCONTEXTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Audio16BitModel::MIXERCONTEXTSETS +
      RecordModel::MIXERCONTEXTSETS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTSETS : 0)
    );
    m->setScaleFactor(1024, 128);
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

    Audio16BitModel& audio16BitModel = models->audio16BitModel();
    audio16BitModel.setParam(blockInfo);
    audio16BitModel.mix(*m);

    int isStereo = (blockInfo & 1);
    shared->State.rLength = (isStereo + 1) * 2;
    RecordModel& recordModel = models->recordModel();
    recordModel.mix(*m);

    return m->p();

  }

  ~ContextModelAudio16() {
    delete m;
  }

};