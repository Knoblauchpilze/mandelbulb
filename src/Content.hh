#ifndef    CONTENT_HH
# define   CONTENT_HH

# include <maths_utils/Box.hh>
# include <sdl_core/SdlWidget.hh>

namespace mandelbulb {
  namespace lib {

    class Content: public sdl::core::SdlWidget {
      public:

        Content(const std::string& name,
                const utils::Sizef& sizeHint = utils::Sizef(),
                const sdl::core::engine::Color& color = sdl::core::engine::Color());

        ~Content() = default;
    };

  }
}

#endif    /* CONTENT_HH */
