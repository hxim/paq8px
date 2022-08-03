#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelImage1 : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelImage1(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models) {
    const bool useLSTM = shared->GetOptionUseLSTM();
    m = mf->createMixer (
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + 
      Image1BitModel::MIXERINPUTS +
      (useLSTM ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE + NormalModel::MIXERCONTEXTS_POST +
      Image1BitModel::MIXERCONTEXTS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + NormalModel::MIXERCONTEXTSETS_POST +
      Image1BitModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTSETS : 0)
      ,
      (useLSTM ? 1 : 0)
    );
    m->setScaleFactor(1200, 130);
  }

  void setParam(int width) {
    Image1BitModel& image1BitModel = models->image1BitModel();
    image1BitModel.setParam(width);
  }

  int p() {

    m->add(256); //network bias

    NormalModel& normalModel = models->normalModel();
    normalModel.mix(*m);
    normalModel.mixPost(*m);

    MatchModel& matchModel = models->matchModel();
    matchModel.mix(*m);

    //is it useful?
    const bool useLSTM = shared->GetOptionUseLSTM();
    if (useLSTM) {
      LstmModel<>& lstmModel = models->lstmModelImage1();
      lstmModel.mix(*m);
    }

    Image1BitModel& image1BitModel = models->image1BitModel();
    image1BitModel.mix(*m);

    return m->p();
  }

  ~ContextModelImage1() {
    delete m;
  }

};