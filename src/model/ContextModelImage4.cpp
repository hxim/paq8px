#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelImage4 : public IContextModel {

private:
  Shared* const shared;
  Models* const models;
  Mixer* m;

public:
  ContextModelImage4(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models) {
    const bool useLSTM = shared->GetOptionUseLSTM();
    m = mf->createMixer (
      1 +  //bias
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + 
      Image4BitModel::MIXERINPUTS +
      (useLSTM ? LstmModel<>::MIXERINPUTS : 0)
      ,
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE +
      Image4BitModel::MIXERCONTEXTS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTS : 0)
      ,
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE + 
      Image4BitModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModel<>::MIXERCONTEXTSETS : 0)
      ,
      (useLSTM ? 1 : 0)
    );
    m->setScaleFactor(2048, 256);
  }

  void setParam(int imageWidthInBytes) {
    Image4BitModel& image4BitModel = models->image4BitModel();
    image4BitModel.setParam(imageWidthInBytes);
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
      LstmModel<>& lstmModel = models->lstmModelImage4();
      lstmModel.mix(*m);
    }

    Image4BitModel& image4BitModel = models->image4BitModel();
    image4BitModel.mix(*m);

    return m->p();
  }

  ~ContextModelImage4() {
    delete m;
  }

};