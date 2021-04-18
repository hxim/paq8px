#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelImage24 {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelImage24(Shared* const sh, Models* const models) : shared(sh), models(models) {
    MixerFactory mf{};
    m = mf.createMixer (
      sh,
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + 
      Image24BitModel::MIXERINPUTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE +
      Image24BitModel::MIXERCONTEXTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Image24BitModel::MIXERCONTEXTSETS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTSETS : 0)
      
    );
  }


  int p(int blockInfo, int isAlpha, int isPNG) {

    m->add(256); //network bias

    NormalModel& normalModel = models->normalModel();
    normalModel.mix(*m);

    MatchModel& matchModel = models->matchModel();
    matchModel.mix(*m);

    //?
    if ((shared->options & OPTION_LSTM) != 0u) {
      LstmModel<>& lstmModel = models->lstmModel();
      lstmModel.mix(*m);
    }

    Image24BitModel& image24BitModel = models->image24BitModel();
    image24BitModel.setParam(blockInfo, isAlpha, isPNG);
    image24BitModel.mix(*m);

    if(isAlpha)
      m->setScaleFactor(2048, 128);
    else 
      m->setScaleFactor(1024, 100); // 800-1300, 90-110

    return m->p();
  }

  ~ContextModelImage24() {
    delete m;
  }

};