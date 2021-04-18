#ifndef PAQ8PX_IDECAY_HPP
#define PAQ8PX_IDECAY_HPP

#include <cstdint>

class IDecay {
public:
  virtual ~IDecay() = default;
  virtual void Apply(float& rate, std::uint64_t const time_step) const = 0;
};

#endif //PAQ8PX_IDECAY_HPP
