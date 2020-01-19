#ifndef    CAMERA_HXX
# define   CAMERA_HXX

# include "Camera.hh"

namespace mandelbulb {

  inline
  utils::Vector3f
  Camera::getEye() const noexcept {
    return m_eye;
  }

  inline
  utils::Vector3f
  Camera::getU() const noexcept {
    return m_u;
  }

  inline
  utils::Vector3f
  Camera::getV() const noexcept {
    return m_v;
  }

  inline
  utils::Vector3f
  Camera::getW() const noexcept {
    return m_w;
  }

}

#endif    /* CAMERA_HXX */
