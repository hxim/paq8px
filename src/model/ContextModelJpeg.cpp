#include "../MixerFactory.hpp"
#include "../Models.hpp"

class ContextModelJpeg : public IContextModel
{

private:
  const Shared* const shared;
  Models* const models;
  Mixer* mEntropyCoded; // for the entropy coded image data, where JpegModel makes predictions
  Mixer* mSpecial;      // for bits whose value follows from the JPEG format (stuffing bytes, markers, restart intervals)
  Mixer* mNoJpeg;       // for all other bits

public:
  ContextModelJpeg(Shared* const sh, Models* const models, const MixerFactory* const mf) :
    shared(sh), models(models)
  {

    mEntropyCoded = mf->createMixer(
      1 + //bias
      NormalModel::MIXERINPUTS + MatchModel::MIXERINPUTS +
      JpegModel::MIXERINPUTS
      ,
      NormalModel::MIXERCONTEXTS_PRE + MatchModel::MIXERCONTEXTS +
      JpegModel::MIXERCONTEXTS
      ,
      NormalModel::MIXERCONTEXTSETS_PRE + MatchModel::MIXERCONTEXTSETS +
      JpegModel::MIXERCONTEXTSETS
      ,
      0
    );
    mEntropyCoded->setScaleFactor(1024, 160, 128);

    mSpecial = mf->createMixer(
      1 + //bias
      NormalModel::MIXERINPUTS + MatchModel::MIXERINPUTS +
      JpegModel::SPECIALMIXERINPUTS // the fixed prediction from JpegModel
      ,
      NormalModel::MIXERCONTEXTS_PRE + MatchModel::MIXERCONTEXTS +
      JpegModel::SPECIALMIXERCONTEXTS
      ,
      NormalModel::MIXERCONTEXTSETS_PRE + MatchModel::MIXERCONTEXTSETS +
      JpegModel::SPECIALMIXERCONTEXTSETS
      ,
      0
    );
    mSpecial->setScaleFactor(4096, 1024, 512); // this mixer only sees stuffing and marker bits

    mNoJpeg = mf->createMixer(
      1 + //bias
      NormalModel::MIXERINPUTS + MatchModel::MIXERINPUTS +
      SparseMatchModel::MIXERINPUTS +
      SparseModel::MIXERINPUTS + SparseBitModel::MIXERINPUTS + RecordModel::MIXERINPUTS + CharGroupModel::MIXERINPUTS +
      TextModel::MIXERINPUTS + WordModel::MIXERINPUTS_BIN +
      LinearPredictionModel::MIXERINPUTS
      ,
      NormalModel::MIXERCONTEXTS_PRE + MatchModel::MIXERCONTEXTS +
      SparseMatchModel::MIXERCONTEXTS +
      SparseModel::MIXERCONTEXTS + SparseBitModel::MIXERCONTEXTS + RecordModel::MIXERCONTEXTS + CharGroupModel::MIXERCONTEXTS +
      TextModel::MIXERCONTEXTS + WordModel::MIXERCONTEXTS +
      LinearPredictionModel::MIXERCONTEXTS
      ,
      NormalModel::MIXERCONTEXTSETS_PRE + MatchModel::MIXERCONTEXTSETS +
      SparseMatchModel::MIXERCONTEXTSETS +
      SparseModel::MIXERCONTEXTSETS + SparseBitModel::MIXERCONTEXTSETS + RecordModel::MIXERCONTEXTSETS + CharGroupModel::MIXERCONTEXTSETS +
      TextModel::MIXERCONTEXTSETS + WordModel::MIXERCONTEXTSETS +
      LinearPredictionModel::MIXERCONTEXTSETS
      ,
      0
    );
    mNoJpeg->setScaleFactor(1500, 150, 120);
  }

  int p() {
    // jpegModel runs first, before any model adds an input:
    // - it sets shared->State.JPEG.state, which NormalModel::updateHashes() and
    //   the SSE stage read;
    // - its result tells which mixer to use for this bit.
    JpegModel& jpegModel = models->jpegModel();
    const JpegResult jpegResult = jpegModel.update();

    Mixer* const m =
      jpegResult == JpegResult::EntropyCoded ? mEntropyCoded :
      jpegResult == JpegResult::Special ? mSpecial :
      mNoJpeg;

    m->add(256); // bias

    NormalModel& normalModel = models->normalModel();
    normalModel.mix(*m);

    MatchModel& matchModel = models->matchModel();
    matchModel.mix(*m);

    switch (jpegResult) {
      case JpegResult::EntropyCoded: {
        jpegModel.mix(*m);
        break;
      }
      case JpegResult::Special: {
        m->add(jpegModel.specialInput());
        m->set(jpegModel.specialContext(), JpegModel::SPECIALMIXERCONTEXTS);
        break;
      }
      case JpegResult::NoJpeg: {
        SparseMatchModel& sparseMatchModel = models->sparseMatchModel();
        sparseMatchModel.mix(*m);
        SparseBitModel& sparseBitModel = models->sparseBitModel();
        sparseBitModel.mix(*m);
        SparseModel& sparseModel = models->sparseModel();
        sparseModel.mix(*m);
        RecordModel& recordModel = models->recordModel();
        recordModel.mix(*m);
        CharGroupModel& charGroupModel = models->charGroupModel();
        charGroupModel.mix(*m);
        TextModel& textModel = models->textModel();
        textModel.mix(*m);
        WordModel& wordModel = models->wordModel();
        wordModel.mix(*m);
        LinearPredictionModel& linearPredictionModel = models->linearPredictionModel();
        linearPredictionModel.mix(*m);
        break;
      }
    }

    return m->p();
  }

  ~ContextModelJpeg() {
    delete mEntropyCoded;
    delete mSpecial;
    delete mNoJpeg;
  }

};
