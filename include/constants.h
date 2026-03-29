#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// This header file defines global constants to:
//  - Avoid hardcoding values directly in the code.
//  - Facilitate future modifications and updates.

#include "data_type.h"


namespace constant {

  constexpr size_type NEQ = 5;
  constexpr size_type NG = 2;

  constexpr value_type PI = 3.14159265358979323846;

  constexpr size_type THREADS_PER_BLOCK_X = 8;
  constexpr size_type THREADS_PER_BLOCK_Y = 8;
  constexpr size_type THREADS_PER_BLOCK_Z = 4;

}

#endif /* _CONSTANTS_H_ */
