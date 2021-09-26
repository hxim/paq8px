#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelImage24 : public IContextModel {

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

  void setParam(int width, int isAlpha, int isPNG) {
    Image24BitModel& image24BitModel = models->image24BitModel();
    image24BitModel.setParam(width, isAlpha, isPNG);
    if (isAlpha)
      m->setScaleFactor(2048, 128);
    else
      m->setScaleFactor(1024, 100); // 800-1300, 90-110
  }

  int p() {

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
    image24BitModel.mix(*m);

    return m->p();
  }

  ~ContextModelImage24() {
    delete m;
  }

};