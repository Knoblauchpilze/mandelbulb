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
  bool
  Camera::setDims(const utils::Sizei& dims) {
    // Check whether the dimensions are actually different.
    if (m_dims == dims) {
      return false;
    }

    // Update the internal properties from the new dimensions.
    updateDims(dims);
  }

  inline
  void
  Camera::copyTransform(const std::vector<float>& rawTransform) {
    // Resize the local attribute to hold all the values. As we will
    // be directly copying from the input vector we need to clear any
    // existing values.
    m_transform.clear();

    // Copy the rotation and the translation, that is all values except
    // the last row.
    m_transform.insert(
      m_transform.begin(),
      rawTransform.begin(),
      rawTransform.begin() + 12u
    );
  }

}

#endif    /* CAMERA_HXX */
