#ifndef    RENDER_PROPERTIES_HH
# define   RENDER_PROPERTIES_HH

namespace mandelbulb {

  struct RenderProperties {
    unsigned iterations; ///< The number of iterations to compute for each pixel
                         ///< of the fractal. The larger this value the more data
                         ///< we can accumulate and thus the more accurate the
                         ///< final image will be but the more time it will take
                         ///< to perform the rendering.
    unsigned accuracy;   ///< For each iteration, individual points are checked
                         ///< to determine whether they belong to the fractal or
                         ///< not. This test involves computing a certain number
                         ///< of terms of a series: this larger this value the
                         ///< more terms will be computed, yielding more accurate
                         ///< results but also taking longer to compute.

    float exponent;      ///< For each individual pixel, the terms of the series
                         ///< involve an exponentiation: this value indicates the
                         ///< degree of the exponentation to apply. The larger
                         ///< this value the more `bulbs` will be present in the
                         ///< final fractal object.
  };

}

#endif    /* RENDER_PROPERTIES_HH */
