#pragma once

#include "IPredictor.hpp"
#include "Shared.hpp"

struct ErrorInfo
{
  uint32_t data[2], sum, mask, collected;

  void reset() {
    memset(this, 0, sizeof(*this));
  }
};

class Mixer : protected IPredictor
{
protected:
  static constexpr int MAX_LEARNING_RATE = 8 * 65536 - 1;
  static constexpr int MIN_LEARNING_RATE_S1 = 2 * 65536 - 1;
  static constexpr int MIN_LEARNING_RATE_SN = 6 * 65536 - 1;

  const Shared* const shared;
  const uint32_t n; /**< max inputs */
  const uint32_t m; /**< max contexts */
  const uint32_t s; /**< max context sets */
  const int lowerLimitOfLearningRate; /**< for linear learning rate decay */
  const bool isAdaptiveLearningRate; /**< linked to command line option '-a' */
  int scaleFactor; /**< scale factor for dot product */
  Array<short, 64> tx; /**< n inputs from add() */
  Array<short, 64> wx; /**< n*m weights */
  Array<uint32_t> cxt; /**< s contexts */
  Array<ErrorInfo> info; /**< stats for the adaptive learning rates  */
  Array<int> rates; /**< learning rates */
  uint32_t numContexts{}; /**< number of contexts (0 to s)  */
  uint32_t base{}; /**< offset of next context */
  uint32_t nx{}; /**< number of inputs in tx, 0 to n */
  Array<int> pr; /**< last result (scaled 12 bits) */
  Mixer* mp; /**< points to a second-layer Mixer to combine results, or nullptr */
  const int simdWidth; /**< number of shorts per SIMD lane, used for input padding */

  /**
    * Computes the dot product of input vector tx and weight vector @p w
    * over @p n elements.
    * @param w weight vector (aligned, padded to simdWidth)
    * @param n number of elements (multiple of simdWidth)
    * @return dot product (scaled by 1/256)
    */
  virtual int dotProduct(const short* w, const size_t n) = 0;

  /**
    * Adjusts weight vector @p w by adding a term proportional to
    * input vector tx and error @p e (gradient descent step).
    * @param w weight vector to update in place
    * @param n number of elements (multiple of simdWidth)
    * @param e scaled error term
    */
  virtual void train(short* w, const size_t n, const int e) = 0;

  /**
    * Must be called at the end of each derived constructor.
    * Creates the second-layer Mixer (when s > 1) using the same SIMD
    * implementation as @p this, so that virtual dispatch is consistent.
    * @param promoted extra inputs forwarded to the second-layer mixer
    */
  virtual void initSecondLayer(int promoted) = 0;

public:
  /**
    * Mixer(n, m, s) combines models using @ref m neural networks with
    * @ref n inputs each, of which up to @ref s may be selected.  If s > 1 then
    * the outputs of these neural networks are combined using another
    * neural network (with arguments s, 1, 1). If s = 1 then the
    * output is direct.
    * @param sh shared context
    * @param n max inputs (will be rounded up to a multiple of simdWidth)
    * @param m max contexts
    * @param s max context sets
    * @param simdWidth number of shorts per SIMD lane (controls input padding)
    */
  Mixer(const Shared* sh, int n, int m, int s, int simdWidth);

  ~Mixer() override;

  /**
    * Predicts the next bit as a 12-bit probability (0 to 4095).
    * Subscribes to the UpdateBroadcaster on first call.
    * @return prediction
    */
  virtual int p();

  virtual void setScaleFactor(int sf0, int sf1);
  virtual void promote(int x);

  /**
    * Adjusts weights to minimize the coding cost of the last prediction.
    * Trains the network where the expected output is the last bit (shared y).
    */
  virtual void update() ;

  /**
    * Inputs x (call up to n times).
    * m.add(stretch(p)) inputs a prediction from one of n models.  The
    * prediction should be positive to predict a 1 bit, negative for 0,
    * nominally +-256 to +-2K.  The maximum allowed value is +-32K but
    * using such large values may cause overflow if n is large.
    * @param x
    */
  void add(int x);

  /**
    * Selects @ref cx as one of @ref range neural networks to use.
    * 0 <= cx < range. Should be called up to @ref s times such
    * that the total of the ranges is <= @ref m.
    * @param cx
    * @param range
    */
  void set(uint32_t cx, uint32_t range);

  void reset();
};
