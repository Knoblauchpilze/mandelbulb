
# include "Camera.hh"

namespace mandelbulb {

  Camera::Camera(const utils::Sizei& dims,
                 float fov,
                 const std::vector<float>& transform):
    utils::CoreObject(std::string("cam_") + dims.toString() + "_fov_" + std::to_string(fov)),

    m_fov(fov),
    m_dims(),
    m_transform(),

    m_focal(0.0f),

    m_eye(),
    m_u(),
    m_v(),
    m_w()
  {
    setService("camera");

    // Check consistency.
    if (m_fov <= 0.0f) {
      error(
        std::string("Cannot create camera with dimensions ") + dims.toString(),
        std::string("Invalid fov ") + std::to_string(m_fov)
      );
    }
    if (transform.size() != 16u) {
      error(
        std::string("Cannot create camera with dimensions ") + m_dims.toString(),
        std::string("Transformation matrix only has ") + std::to_string(transform.size()) + " element(s) (16 expected)"
      );
    }

    // Copy the relevant values of the transform.
    copyTransform(transform);

    // Update the camera from dimensions.
    setDims(dims);
  }

  void
  Camera::updateDims(const utils::Sizei& dims) {
    // Update the dimensions to their provided value.
    m_dims = dims;

    // Determine the focal from the fov.
    m_focal = (m_dims.w() / 2.0f) / std::tan(0.5f * m_fov * 3.1415926535f / 180.0f);
    if (m_focal <= 0.0f) {
      error(
        std::string("Could not create camera with dimensions ") + m_dims.toString() + " and fov " + std::to_string(m_fov),
        std::string("Invalid focal ") + std::to_string(m_focal)
      );
    }

    // Compute the position of the eye from the transformation matrix. We assume
    // a row major description of the input `transform` vector.
    m_eye = utils::Vector3f(
      m_transform[0u * 4u + 3u],
      m_transform[1u * 4u + 3u],
      m_transform[2u * 4u + 3u]
    );

    float d = m_eye.length();

    // Compute UVW vectors for camera setup.
    float mul = d * m_dims.w() / (2.0f * m_focal);
    m_u = utils::Vector3f(
      mul * m_transform[0u * 4u + 0u],
      mul * m_transform[1u * 4u + 0u],
      mul * m_transform[2u * 4u + 0u]
    );

    mul = d * m_dims.h() / (2.0f * m_focal);
    m_u = utils::Vector3f(
      mul * m_transform[0u * 4u + 1u],
      mul * m_transform[1u * 4u + 1u],
      mul * m_transform[2u * 4u + 1u]
    );

    m_w = utils::Vector3f(
      -d * m_transform[0u * 4u + 2u],
      -d * m_transform[1u * 4u + 2u],
      -d * m_transform[2u * 4u + 2u] 
    );

    log(
      std::string("Created camera with eye: ") + m_eye.toString() + ", u: " + m_u.toString() +
      ", v: " + m_v.toString() + ", w: " + m_w.toString()
    );
  }

}
