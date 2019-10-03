
# include "Content.hh"

namespace mandelbulb {
  namespace lib {

    Content::Content(const std::string& name,
                     const utils::Sizef& sizeHint,
                     const sdl::core::engine::Color& color):
      sdl::core::SdlWidget(name, sizeHint, nullptr, color)
    {
      // Empty implementation.
    }

  }
}
