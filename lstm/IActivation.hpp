#ifndef PAQ8PX_IACTIVATION_HPP
#define PAQ8PX_IACTIVATION_HPP

#include "../Utils.hpp"

class IActivation {
public:
  virtual ~IActivation() = default;
  virtual void Run(float* f, std::size_t const len) const = 0;
};

#endif //PAQ8PX_IACTIVATION_HPP
