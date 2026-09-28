#pragma once

#include "IPredictor.hpp"
#include "Shared.hpp"


/**
  * Combines model predictions with a three-layer network, called a Mixer.
  *
  * The Mixer receives @p n stretched probabilities from the different models.
  * It combines them using adaptive weights and produces a single probability
  * representing the expected probability of the next bit being 1. The Mixer
  * output is passed to the SSE stage and then to the arithmetic coder.
  *
  * The Mixer uses integer arithmetic. Probabilities and weights are represented
  * using fixed-point numbers.
  *
  * The models supply the Mixer with a fixed number of mixer contexts (small
  * integers). Each mixer context selects one weight vector from its context set.
  * This lets the Mixer use different weights in different situations, such as
  * different file types or parts of a file. The mixer contexts are chosen by
  * the models rather than learned by the Mixer.
  *
  * 
  * Layers
  * ---
  * input      Receives @p n stretched probabilities from the models. 
  *            For each of the @p s context sets, one of the @p m weight vectors
  *            is selected. This produces @p s stretched outputs.
  *
  * middle     Receives the @p s outputs from the input layer and @p promoted
  *            inputs from strong models (so-called promoted inputs). It uses
  *            the same context selections as the input layer, so each output
  *            has its own set of context-dependent weights. This produces
  *            another @p s stretched outputs.
  *
  * combining  Receives the @p s outputs from the middle layer and the @p s
  *            outputs from the input layer directly (the skip connection),
  *            as well as the promoted inputs. It uses one weight vector to
  *            combine them into the final output.
  *
  * In addition to passing its stretched output probabilities forward,
  * each layer also stores a squashed version that update() can use to adjust
  * the weights based on how accurate the prediction was.
  * This helps the Mixer learn how to weight its inputs in each context,
  * which improves prediction accuracy over time.
  *
  * 
  * Weight scales
  * ---
  * The input layer receives raw model outputs, while the middle and combining
  * layers receive more stable, mature signals produced by mixing, as well as
  * promoted inputs from strong models. These signals have different magnitudes,
  * so the 3 layers need different weight scales.
  *
  * Each layer therefore scales its dot product — the sum of its inputs
  * multiplied by their weights — by its own predefined factor. This keeps the
  * weights in a useful range and gives all weights in the layer a similar
  * effective learning rate.
  *
  *
  * Input layout for each layer
  * ---
  * Each layer's input vector is padded to a multiple of the SIMD width so it
  * can be processed efficiently with SIMD instructions. Padding slots are kept
  * at zero and therefore have no effect on the dot product.
  *
  * The models fill the input layer by appending inputs with add() for each bit.
  * The input slots are refilled from the start for each new bit.
  *
  * input       model inputs            [ 0 .. n               )
  *             SIMD padding            [ n .. padded size     )
  *
  * middle      input layer outputs     [ 0 .. s               )
  *             promoted inputs         [ s .. s+promoted      )
  *             SIMD padding            [ s+promoted .. padded size )
  *
  * combining   middle layer outputs    [ 0 .. s               )
  *             skip connection         [ s .. 2s              )
  *             promoted inputs         [ 2s .. 2s+promoted     )
  *             SIMD padding            [ 2s+promoted .. padded size )
  *
  * 
  * Weight initialization
  * ---
  * The weights are initialized to a small positive value instead of zero. This
  * gives each input a small initial contribution while leaving room for training
  * to increase or decrease the weight. Because the inputs are often correlated,
  * training may eventually make some weights very small or even negative to
  * cancel information that is already represented by another input.
  *
  * Starting with a small positive value therefore gives the weights a useful
  * neutral starting point: some can grow from there, while others can shrink
  * and eventually become negative.
  *
  * 
  * Skip connection
  * ---
  * The combining layer sees the input layer's outputs directly, not only through
  * the middle layer. The corresponding input and middle outputs are strongly
  * correlated because they are based on overlapping information and are trained
  * toward the same target.
  *
  * Feeding both correlated signals directly to the combining layer makes learning
  * less efficient. Instead, the skip input is represented as the difference
  * between the input and middle outputs. This gives the combining layer one
  * signal representing the middle layer's prediction and another representing
  * how the input layer differs from it. The difference is typically smaller and
  * less correlated with the middle output, which makes the two signals easier to
  * learn from together.
  *
  * This is why slot s+i carries the difference between input output i and middle
  * output i.
  *
  * Using the difference does not limit what the combining layer can represent.
  * With w_m as the weight for the middle output and w_s as the weight for the
  * skip slot,
  *
  * w_m*mid + w_s*(in - mid) == (w_m - w_s)*mid + w_s*in
  *
  * so any combination of the middle and input outputs can still be represented.
  *
  * Unlike the other weights, the skip weights are initialized to zero. A
  * difference has no preferred direction: a positive weight favors cases where
  * the input output is higher than the middle output, while a negative weight
  * favors the opposite. Zero is therefore a natural neutral starting point for
  * this signal. It also leaves the initial prediction unchanged until training
  * learns a useful correction.
  *
  * This initialization was chosen based on compression results: initializing
  * the skip weights like the other weights improved large files but hurt small
  * files, while starting them at zero gave better overall behavior.
  *
  */

class Mixer : protected IPredictor
{
protected:

  // Fixed-point scale used for the learning rates.
  static constexpr int LR_SCALE = 1 << (16 + 2);
  static constexpr int MIN_LEARNING_RATE_UNITS = 1;
  static constexpr int MAX_LEARNING_RATE_UNITS = 8;
  static constexpr int MAX_LEARNING_RATE = MAX_LEARNING_RATE_UNITS * LR_SCALE - 1;

  // Converts a learning rate expressed in integer units to the fixed-point
  // representation used by the training code. The -1 prevents overflow.
  static constexpr int learningRate(const int units) { return units * LR_SCALE - 1; }

  // Default minimum learning rates reached by the learning rate
  // decay in each layer.
  static constexpr int DEFAULT_END_RATE_INPUT = 6;
  static constexpr int DEFAULT_END_RATE_MIDDLE = 2;
  static constexpr int DEFAULT_END_RATE_COMBINING = 1;

  /**
  * Context selections made for one bit.
  *
  * The input and middle layers share the same selections, so output i in both
  * layers corresponds to the same mixer context. Each selection identifies
  * one weight vector within the available vectors.
  */
  struct ContextSet
  {
    const uint32_t m;    /**< number of weight vectors available */
    const uint32_t s;    /**< number of mixer contexts */
    Array<uint32_t> cxt; /**< selected weight-vector indices */
    size_t count;        /**< number of selections made so far this bit (0..s) */
    uint32_t base;       /**< start of the next range of weight vectors */

    ContextSet(const uint32_t m, const uint32_t s) : m(m), s(s), cxt(s), count(0), base(0) {}
    ContextSet(const ContextSet&) = delete;
    ContextSet& operator=(const ContextSet&) = delete;

    /** Selects context @p cx from the next @p range weight vectors. */
    void select(const uint32_t cx, const uint32_t range) {
      assert(count < s);
      assert(cx < range);
      assert(base + range <= m);
      cxt[count++] = base + cx;
      base += range;
    }

    /** Starts a new set of context selections. */
    void reset() {
      count = 0;
      base = 0;
    }
  };

  /** A contiguous range of slots in a layer's input vector. */
  struct Block
  {
    size_t offset;
    size_t count;
    size_t end() const { return offset + count; }
  };

  /**
  * All parameters, weights, predictions and temporary data for one layer.
  */
  struct Layer
  {
    ContextSet& contexts;  /**< context selections used to choose weight vectors */
    const size_t n;        /**< number of input slots, including SIMD padding */
    const int maxInput;    /**< maximum allowed absolute input value */
    const int errorLimit;  /**< training is skipped when the absolute error is within this limit */

    int scaleFactor;                /**< scale applied to the layer's dot product */
    int rate;                       /**< current learning rate */
    int lowerLimitOfLearningRate;   /**< minimum learning rate reached by decay */

    Array<short, 64> tx;   /**< input vector, padded with zero slots for SIMD */
    Array<short, 64> wx;   /**< weight vectors, one for each context */
    Array<short> pr;       /**< squashed 12-bit predictions used by update() */
    Array<short> out;      /**< stretched outputs routed to later layers */

    Block routed;          /**< slots receiving outputs from the preceding layer */
    Block skip;            /**< slots receiving the skip connection */
    Block promoted;        /**< slots receiving promoted inputs */

    Layer(ContextSet& contexts, size_t n, int maxInput, int errorLimit, short initialWeight,
      int lowerLimit) :
      contexts(contexts), n(n),
      maxInput(maxInput),
      errorLimit(errorLimit),
      scaleFactor(0),
      rate(MAX_LEARNING_RATE),
      lowerLimitOfLearningRate(lowerLimit),
      tx(n), wx(n* contexts.m), pr(contexts.s), out(contexts.s),
      routed{}, skip{}, promoted{} {
      for (uint32_t i = 0; i < contexts.s; ++i) {
        pr[i] = 2048; // initial p = 0.5
        out[i] = 0;
      }
      for (size_t i = 0; i < n * contexts.m; ++i) {
        wx[i] = initialWeight;
      }
    }

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
  };

  const Shared* const shared;
  const size_t numInputs;            /**< number of add() calls per bit; inputLayer.n rounds this up */
  const size_t numPromotedInputs;    /**< number of slots available to promote() */
  size_t numPromoted;                /**< promote() calls since the last reset */
  size_t numAdded;                   /**< add() calls since the last reset */

  ContextSet bitContexts;   /**< the contexts selected per bit for the input and middle layers */
  ContextSet singleContext; /**< the single context used by the combining layer */

  Layer inputLayer;       /**< combines the model inputs */
  Layer middleLayer;      /**< remixes input-layer outputs and promoted inputs */
  Layer combiningLayer;   /**< combines middle, skip, and promoted inputs */

  /**
  * SIMD implementations used by all layers.
  *
  * dotProduct computes one dot product, dotProduct2 computes two dot products
  * over the same input vector, and train() updates one weight vector.
  */
  virtual int dotProduct(const short* t, const short* w, size_t n) = 0;
  virtual int dotProduct2(const short* t, const short* w0, const short* w1, size_t n, int& sum1) = 0;
  virtual void train(const short* t, short* w, size_t n, int e) = 0;

  /**
  * Checks that @p layer has valid dimensions for the SIMD kernels and that
  * its dot products fit safely in their accumulators.
  * Used in asserts only.
  */
  static bool isWellFormed(const Layer& layer, int simdWidth);

  /** Sets every weight in @p block of @p layer to @p value. */
  static void setWeightBlock(Layer& layer, const Block& block, short value);

  /**
  * Computes the outputs of @p layer using the selected weight vectors.
  * The stretched outputs are stored in @p layer.out and their squashed
  * probabilities in @p layer.pr.
  */
  void forward(Layer& layer);

  /** Copies the stretched outputs of @p from into @p block of @p to. */
  static void routeOutputs(const Layer& from, Layer& to, const Block& block);

  /**
  * Fills the combining layer's skip block with the difference between the input layer
  * and middle layer output.
  */
  void routeSkipConnection();

public:
  /**
  * @param n inputs to the input layer, i.e. add() calls per bit. The input
  *          vector is rounded up to a multiple of @p simdWidth.
  * @param m number of weight vectors available to the input and middle layers
  * @param s number of mixer contexts selected per bit
  * @param promoted number of extra inputs available through promote()
  * @param simdWidth number of shorts in one SIMD vector
  */
  Mixer(const Shared* sh, int n, int m, int s, int promoted, int simdWidth);

  // The layers refer to context sets inside *this*.
  Mixer(const Mixer&) = delete;
  Mixer& operator=(const Mixer&) = delete;
  Mixer(Mixer&&) = delete;
  Mixer& operator=(Mixer&&) = delete;

  ~Mixer() override;

  /**
  * Computes the final 12-bit probability for the next bit (1..4095).
  * All inputs and context selections for the current bit must have been
  * supplied before calling p().
  */
  virtual int p();

  /** Sets the dot-product scale factors for the input, middle, and combining layers. */
  void setScaleFactor(int sf0, int sf1, int sf2);

  /**
  * Sets the minimum learning rates reached by decay in the input, middle,
  * and combining layers.
  */
  void setLowerLimitOfLearningRate(int lr0, int lr1, int lr2);

  /**
  * Adds one input to the promoted blocks of both the middle and combining
  * layers. The value must be a stretched prediction in the range -2047..2047.
  */
  void promote(int x);

  /** Updates the weights of all layers using the error from the last prediction. */
  void update() override;

  /**
  * Adds one model prediction to the input layer. Call n times per bit.
  * The value must be a stretched prediction in the range -2047..2047.
  */
  void add(int x);

  /**
  * Selects context @p cx from a range of @p range weight vectors. Call @p s
  * times per bit; the selections are used by both the input and middle layers.
  */
  void set(uint32_t cx, uint32_t range);

  /** Starts a new bit by resetting the per-bit input, promotion, and context-selection state. */
  void reset();
};
