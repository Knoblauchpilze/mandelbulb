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

  inline
  utils::Vector3f
  Camera::getDirection(const utils::Vector2f& perc) {
    // Use the internal `u/v/w` vectors and normalize the
    // result.
    utils::Vector3f rawDir =
      perc.x() * m_u +
      perc.y() * m_v +
      m_w
    ;

    return rawDir.normalized();
  }

  inline
  bool
  Camera::setDims(const utils::Sizei& dims) {
    // Check whether the dimensions are actually different.
    if (m_dims == dims) {
      return false;
    }

    // Update the internal properties from the new dimensions.
    m_dims = dims;

    // Recompute the internal vectors.
    updateEUVW();

    return true;
  }

  inline
  float
  Camera::getDistance() const noexcept {
    return m_distance;
  }

  inline
  bool
  Camera::setDistance(float distance) {
    // Consistency check.
    if (distance < 0.0f) {
      warn("Could not set distance to " + std::to_string(distance) + " for camera, keeping " + std::to_string(m_distance));
      return false;
    }

    // Check whether the distance is different from the current one.
    if (utils::fuzzyEqual(distance, m_distance)) {
      return false;
    }

    // Update the distance to the origin.
    m_distance = distance;

    // Update internal vectors.
    updateEUVW();

    return true;
  }

  inline
  bool
  Camera::rotate(const utils::Vector2f& angle) {
    // Check whether the delta to apply on any of the angle is valid.
    if (utils::fuzzyEqual(angle.x(), 0.0f) &&
        utils::fuzzyEqual(angle.y(), 0.0f))
    {
      return false;
    }

    // Update the internal rotations. Don't forget the flipover prevention
    // along the vertical axis.
    utils::Vector2f save = m_rotations;

    m_rotations.x() += angle.x();
    m_rotations.y() += angle.y();
    m_rotations.y() = std::min(
      3.1415926535f / 2.0f - getVerticalThreshold(),
      std::max(
        -3.1415926535f / 2.0f + getVerticalThreshold(),
        m_rotations.y()
      )
    );

    // Check whether we actually changed anything.
    if (utils::fuzzyEqual(m_rotations.x(), save.x()) &&
        utils::fuzzyEqual(m_rotations.y(), save.y()))
    {
      return false;
    }

    // Update the internal angles.
    updateEUVW();

    return true;
  }

  inline
  float
  Camera::getVerticalThreshold() noexcept {
    return 0.001f;
  }

}

#endif    /* CAMERA_HXX */
