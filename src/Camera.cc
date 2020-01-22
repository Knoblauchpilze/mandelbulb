
# include "Camera.hh"

namespace mandelbulb {

  Camera::Camera(const utils::Sizei& dims,
                 float fov,
                 float distance,
                 const utils::Vector2f& rotations):
    utils::CoreObject(std::string("cam_") + dims.toString() + "_fov_" + std::to_string(fov)),

    m_fov(fov),
    m_dims(),
    m_focal(0.0f),
    m_distance(),
    m_rotations(),

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

    // Update the camera from provided properties.
    setDims(dims);
    setDistance(distance);
    rotate(rotations);
  }

  void
  Camera::updateEUVW() {
    // We will apply both the distance and the rotations at the
    // same time. As defined in the following artical:
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system
    // The `x` coordinate of `m_rotations` will be any value while the
    // `y` coordinate will be in the range `]-pi/2, pi/2[` effectively
    // making us use the elevation semantic (latitude).
    // Note that as we want initially the camera to be pointing
    float theta = m_rotations.x();
    float phi = m_rotations.y();

    m_eye.x() = m_distance * std::sin(theta) * std::cos(phi);
    m_eye.y() = -m_distance * std::cos(theta) * std::cos(phi);
    m_eye.z() = m_distance * std::sin(phi);

    float d = m_eye.length();

    utils::Vector3f forward = -m_eye;
    forward.normalize();

    // The `lateral` vec can be approximated as the cross product between
    // the `forward` vector and the canonical `z` up vector. We will derive
    // the real `up` vector from it and then orthogonalize the base using
    // Gram-Schmidt.
    utils::Vector3f lateral(forward.y(), -forward.x(), 0.0f);
    lateral.normalize();

    // We can now compute the real up vector from the lateral (which should
    // be correct) and the forward vector.
    utils::Vector3f up = lateral ^ forward;

    // Correct the `lateral` with the real up.
    lateral = forward ^ up;

    // Correct the base with Gram-Schmidt assuming the `forward` vector is
    // the only certain vector.
    lateral -= (forward * lateral) * forward;
    lateral.normalize();

    up -= (forward * up) * forward;
    up -= (lateral * up) * lateral;
    up.normalize();

    // Compute UVW vectors for camera setup. In order to do that we
    // need to compute the transformation of the base coordinate
    // frame of the camera.
    float mul = d * m_dims.w() / (2.0f * m_focal);
    m_u = mul * lateral;

    mul = d * m_dims.h() / (2.0f * m_focal);
    m_v = mul * up;

    m_w = d * forward;

    log(
      std::string("Created camera with eye: ") + m_eye.toString() + ", u: " + m_u.toString() +
      ", v: " + m_v.toString() + ", w: " + m_w.toString()
    );
  }

}
