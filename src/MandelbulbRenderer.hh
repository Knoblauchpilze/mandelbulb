#ifndef    MANDELBULB_RENDERER_HH
# define   MANDELBULB_RENDERER_HH

# include <mutex>
# include <sdl_graphic/ScrollableWidget.hh>

namespace mandelbulb {

  class MandelbulbRenderer: public sdl::graphic::ScrollableWidget {
    public:

      MandelbulbRenderer(const utils::Sizef& sizeHint = utils::Sizef(),
                         sdl::core::SdlWidget* parent = nullptr);

      ~MandelbulbRenderer();

    private:

      void
      build();

    private:

      /**
       * @brief - A mutex allowing to protect this widget from concurrent accesses.
       */
      mutable std::mutex m_propsLocker;
  };

}

# include "MandelbulbRenderer.hxx"

#endif    /* MANDELBULB_RENDERER_HH */
