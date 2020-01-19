#ifndef    CAMERA_HH
# define   CAMERA_HH

# include <memory>
# include <vector>
# include <maths_utils/Size.hh>
# include <maths_utils/Vector3.hh>
# include <core_utils/CoreObject.hh>

namespace mandelbulb {

  class Camera: public utils::CoreObject {
    public:

      Camera(const utils::Sizei& dims,
             float fov,
             const std::vector<float>& transform);

      ~Camera() = default;

      utils::Vector3f
      getEye() const noexcept;

      utils::Vector3f
      getU() const noexcept;

      utils::Vector3f
      getV() const noexcept;

      utils::Vector3f
      getW() const noexcept;

    private:

      float m_fov;

      utils::Sizei m_dims;
      float m_focal;

      utils::Vector3f m_eye;
      utils::Vector3f m_u;
      utils::Vector3f m_v;
      utils::Vector3f m_w;
  };

  using CameraShPtr = std::shared_ptr<Camera>;
}

# include "Camera.hxx"

#endif    /* CAMERA_HH */
