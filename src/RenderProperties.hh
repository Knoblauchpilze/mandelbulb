#ifndef    RENDER_PROPERTIES_HH
# define   RENDER_PROPERTIES_HH

# include <sdl_engine/Gradient.hh>
namespace mandelbulb {

  struct RenderProperties {
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

    float hitThreshold;  ///< The distance under which a hit is considered on the
                         ///< surface of the fractal. The smaller this value the
                         ///< more accurate the intersections will be but the more
                         ///< time it will take to reach the fractal.

    unsigned raySteps;   ///< The maximum number of steps to perform for a single
                         ///< ray before considering that it has either diverged
                         ///< or converged.

    float bailout;       ///< The bailout distance above which the ray is said to
                         ///< diverge when computing the series for the fractal.
                         ///< The more accurate this value the quicker we can get
                         ///< out of computing terms which are not converging.
  };

}

#endif    /* RENDER_PROPERTIES_HH */
