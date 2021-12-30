#pragma once

#include <cmath>
#include <cstdint>

#include "Shared.hpp"

/**
 * Ordinary Least Squares predictor
 * @tparam F
 * @tparam T
 * @tparam hasZeroMean
 */
template<typename F, typename T, const bool hasZeroMean = true>
class OLS {
    static constexpr F ftol = 1E-8;
    static constexpr F sub = F(int64_t(!hasZeroMean) << (8 * sizeof(T) - 1));

private:
    const Shared* const shared;
    int n, kMax, km, index;
    F lambda, nu;
    Array<F,32> x, w, b;
    Array<F,32> mCovariance;
    Array<F,32> mCholesky;

    auto factor() -> int {
      // copy the matrix
      memcpy(&mCholesky[0], &mCovariance[0], n * n * sizeof(F));

      for( int i = 0; i < n; i++ ) {
        mCholesky[i*n+i] += nu; //main diagonal
      }
      for( int i = 0; i < n; i++ ) {
        for( int j = 0; j < i; j++ ) {
          F sum = mCholesky[i*n+j];
          for( int k = 0; k < j; k++ ) {
            sum -= (mCholesky[i*n+k] * mCholesky[j*n+k]);
          }
          mCholesky[i*n+j] = sum / mCholesky[j*n+j];
        }
        F sum = mCholesky[i*n+i];
        for( int k = 0; k < i; k++ ) {
          sum -= (mCholesky[i*n+k] * mCholesky[i*n+k]);
        }
        if( sum > ftol ) {
          mCholesky[i*n+i] = sqrt(sum); //main diagonal
        } else {
          return 1;
        }
      }
      return 0;
    }

    void solve() {
      for( int i = 0; i < n; i++ ) {
        F sum = b[i];
        for( int j = 0; j < i; j++ ) {
          sum -= (mCholesky[i*n+j] * w[j]);
        }
        w[i] = sum / mCholesky[i*n+i];
      }
      for( int i = n - 1; i >= 0; i-- ) {
        F sum = w[i];
        for( int j = i + 1; j < n; j++ ) {
          sum -= (mCholesky[j*n+i] * w[j]);
        }
        w[i] = sum / mCholesky[i*n+i];
      }
    }

public:
    OLS(const Shared* const sh, int n, int kMax = 1, F lambda = 0.998, F nu = 0.001) : shared(sh),
      n(n), kMax(kMax), lambda(lambda), nu(nu), 
      x(n), w(n), b(n),
      mCovariance(n*n), mCholesky(n*n) {
        km = index = 0;
        for( int i = 0; i < n; i++ ) {
          x[i] = w[i] = b[i] = 0.0;
          for( int j = 0; j < n; j++ ) {
            mCovariance[i*n+j] = mCholesky[i*n+j] = 0.0;
          }
        }
      }

    void add(const T val) {
      assert(index < n);
      x[index++] = F(val) - sub;
    }

    void addFloat(const F val) {
      assert(index < n);
      x[index++] = val - sub;
    }

    auto predict(const T **p) -> F {
      F sum = 0.;
      for( int i = 0; i < n; i++ ) {
        sum += w[i] * (x[i] = F(*p[i]) - sub);
      }
      return sum + sub;
    }

    auto predict() -> F {
      assert(index == n);
      index = 0;
      F sum = 0.;
      for( int i = 0; i < n; i++ ) {
        sum += w[i] * x[i];
      }
      return sum + sub;
    }

    inline void update(const T val) {
#ifdef __GNUC__
      if( shared->chosenSimd == SIMDType::SIMD_AVX2 ) {
        updateAVX2(val);
      } else
#endif
      {
        updateUnrolled(val);
      }
    }

#ifdef __GNUC__
#ifdef __AVX2__
    __attribute__((target("avx2")))
#endif
#endif
    void updateAVX2(const T val) {
      F mul = 1.0 - lambda;
      for( int j = 0; j < n; j++ ) {
        for( int i = 0; i < n; i++ ) {
          mCovariance[j*n+i] = lambda * mCovariance[j*n+i] + mul * (x[j] * x[i]);
        }
      }
      mul *= (F(val) - sub);
      for( int i = 0; i < n; i++ ) {
        b[i] = lambda * b[i] + mul * x[i];
      }
      km++;
      if( km >= kMax ) {
        if( !factor()) {
          solve();
        }
        km = 0;
      }
    }

    void updateUnrolled(const T val) {
      F mul = 1.0 - lambda;
      int l = n - (n & 3);
      int i = 0;
      for( int j = 0; j < n; j++ ) {
        for( i = 0; i < l; i += 4 ) {
          mCovariance[j*n+i] = lambda * mCovariance[j*n+i] + mul * (x[j] * x[i]);
          mCovariance[j*n+i + 1] = lambda * mCovariance[j*n+i + 1] + mul * (x[j] * x[i + 1]);
          mCovariance[j*n+i + 2] = lambda * mCovariance[j*n+i + 2] + mul * (x[j] * x[i + 2]);
          mCovariance[j*n+i + 3] = lambda * mCovariance[j*n+i + 3] + mul * (x[j] * x[i + 3]);
        }
        for( ; i < n; i++ ) {
          mCovariance[j*n+i] = lambda * mCovariance[j*n+i] + mul * (x[j] * x[i]);
        }
      }
      mul *= (F(val) - sub);
      for( i = 0; i < l; i += 4 ) {
        b[i] = lambda * b[i] + mul * x[i];
        b[i + 1] = lambda * b[i + 1] + mul * x[i + 1];
        b[i + 2] = lambda * b[i + 2] + mul * x[i + 2];
        b[i + 3] = lambda * b[i + 3] + mul * x[i + 3];
      }
      for( ; i < n; i++ ) {
        b[i] = lambda * b[i] + mul * x[i];
      }
      km++;
      if( km >= kMax ) {
        if( !factor()) {
          solve();
        }
        km = 0;
      }
    }
};
