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

  struct ShadingProperties {
    sdl::core::engine::Color fColor;      ///< The color to use to represent the
                                          ///< fractal. Represents the base color
                                          ///< of the object without any lights'
                                          ///< modification.

    sdl::core::engine::Color noDataColor; ///< The color to use to represent the
                                          ///< regions where rays do not hit the
                                          ///< fractal (usually black).

    float exposure;                       ///< A value used to multiply luminance
                                          ///< of pixels before applying the tone
                                          ///< mapping: allows to artificially do
                                          ///< some exposure correction on pixels
                                          ///< of the scene.

    float burnout;                        ///< A factor allowing to slightly burn
                                          ///< very white areas of the final image
                                          ///< so that we get a nicer look.
  };

}

#endif    /* RENDER_PROPERTIES_HH */
