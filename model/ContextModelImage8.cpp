#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelImage8 : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelImage8(Shared* const sh, Models* const models) : shared(sh), models(models) {
    MixerFactory mf{};
    m = mf.createMixer (
      sh,
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + 
      Image8BitModel::MIXERINPUTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE +
      Image8BitModel::MIXERCONTEXTS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Image8BitModel::MIXERCONTEXTSETS +
      (((shared->options & OPTION_LSTM) != 0u) ? LstmModel<>::MIXERCONTEXTSETS : 0)
      
    );
  }

  void setParam(int width, uint32_t isGray, uint32_t isPNG) {
    Image8BitModel& image8BitModel = models->image8BitModel();
    image8BitModel.setParam(width, isGray, isPNG);
    if (isGray)
      m->setScaleFactor(1300, 100); // 1100-1400, 90-110
    else
      m->setScaleFactor(1600, 110); // 1500-1800, 100-128
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

    Image8BitModel& image8BitModel = models->image8BitModel();
    image8BitModel.mix(*m);

    return m->p();
  }

  ~ContextModelImage8() {
    delete m;
  }

};