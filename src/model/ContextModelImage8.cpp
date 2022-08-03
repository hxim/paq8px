#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelImage8 : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelImage8(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models) {
    const bool useLSTM = shared->GetOptionUseLSTM();
    m = mf->createMixer (
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + 
      Image8BitModel::MIXERINPUTS +
      (useLSTM ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE +
      Image8BitModel::MIXERCONTEXTS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Image8BitModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTSETS : 0)
      ,
      (useLSTM ? 1 : 0)
    );
  }

  void setParam(int width, uint32_t isGray) {
    Image8BitModel& image8BitModel = models->image8BitModel();
    image8BitModel.setParam(width, isGray);
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

    //is it useful?
    const bool useLSTM = shared->GetOptionUseLSTM();
    if (useLSTM) {
      LstmModel<>& lstmModel = models->lstmModelImage8();
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