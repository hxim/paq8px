#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelImage24 : public IContextModel
{

private:
  Shared* const shared;
  Models* const models;
  Mixer* m0;
  Mixer* m1;
  Mixer* m2;

public:
  ContextModelImage24(Shared* const sh, Models* const models, const MixerFactory* const mf) : shared(sh), models(models) {
    const bool useLSTM = shared->GetOptionUseLSTM();

    int mixerinputs =
      MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS +
      Image24BitModel::MIXERINPUTS +
      (useLSTM ? LstmModelContainer::MIXERINPUTS : 0);
    int mixerContexts =
      MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS_PRE +
      Image24BitModel::MIXERCONTEXTS +
      (useLSTM ? LstmModelContainer::MIXERCONTEXTS : 0);
    int mixerContextSets =
      MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS_PRE +
      Image24BitModel::MIXERCONTEXTSETS +
      (useLSTM ? LstmModelContainer::MIXERCONTEXTSETS : 0);
    int promotedInputs = (useLSTM ? 1 : 0);

    m0 = mf->createMixer(mixerinputs, mixerContexts, mixerContextSets, promotedInputs);
    m1 = mf->createMixer(mixerinputs, mixerContexts, mixerContextSets, promotedInputs);
    m2 = mf->createMixer(mixerinputs, mixerContexts, mixerContextSets, promotedInputs);

    m0->setScaleFactor(490, 130);
    m1->setScaleFactor(620, 135);
    m2->setScaleFactor(770, 140);

    m0->setLowerLimitOfLearningRate(5, 1);
    m1->setLowerLimitOfLearningRate(5, 1);
    m2->setLowerLimitOfLearningRate(5, 1);
  }

  void setParam(int width, int isAlpha) {
    Image24BitModel& image24BitModel = models->image24BitModel();
    image24BitModel.setParam(width, isAlpha);
  }

  int p() {

    Image24BitModel& image24BitModel = models->image24BitModel();
    image24BitModel.update();

    int color = image24BitModel.color;
    Mixer* m =
      color == 0 ? m0 :
      color == 1 ? m1 :
      color == 2 ? m2 : m2;

    NormalModel& normalModel = models->normalModel();
    normalModel.mix(*m);

    MatchModel& matchModel = models->matchModel();
    matchModel.mix(*m);

    const bool useLSTM = shared->GetOptionUseLSTM();
    if (useLSTM) {
      LstmModelContainer& lstmModel = models->lstmModelImage24();
      lstmModel.mix(*m);
    }

    image24BitModel.mix(*m);

    return m->p();
  }

  ~ContextModelImage24() {
    delete m0;
    delete m1;
    delete m2;
  }

};
