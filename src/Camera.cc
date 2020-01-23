
# include "Camera.hh"

namespace mandelbulb {

  Camera::Camera(const utils::Sizei& dims,
                 float fov,
                 float distance,
                 const utils::Vector2f& rotations):
    utils::CoreObject(std::string("cam_") + dims.toString() + "_fov_" + std::to_string(fov)),

    m_fov(fov),
    m_dims(),
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
    // We know that we want the `m_w` vector to be the normalized
    // version of the `forward` vector. That leaves with computing
    // the `m_u` and `m_v`. We know that the aspect ratio of the
    // image should be preserved and that the field of view should
    // also be applied.
    // For the `m_u` vector, the situation looks like below:
    //
    //         eye
    //   u <----+
    //         /|\                avoid multi line comment
    //    fov /_|_\               avoid multi line comment
    //       /  |  \              avoid multi line comment
    //      /   |<--\--- 1
    //     /____|____\            avoid multi line comment
    //        aspect
    //
    // We can easily compute the that `u = tan(fov/2) * aspect/2`
    // and in this case we define a fov of `1`. To accomodate for
    // other values we will scale the `u` vector by the tangent
    // as defined.
    //
    // For the `v` vector we have the following situation:
    //
    //         eye
    //   v <----+
    //         /|\                avoid multi line comment
    //  alpha /_|_\               Actually this should be rotated by
    //       /  |  \              pi/2 radians.
    //      /   |<--\--- 1
    //     /____|____\            avoid multi line comment
    //          1
    //
    // Similarly ideally we have `v = tan(alpha/2) * 1`. We will
    // accomodate for the field of view in the same way than for
    // the `u` vector.
    float aspect = 1.0f * m_dims.w() / m_dims.h();

    m_u = (aspect * std::tan(m_fov / 2.0f)) * lateral;
    m_v = std::tan(m_fov / 2.0f) * up;
    m_w = forward;

    log(
      std::string("Created camera with eye: ") + m_eye.toString() + ", u: " + m_u.toString() +
      ", v: " + m_v.toString() + ", w: " + m_w.toString() + " (aspect: " + std::to_string(aspect) + ")",
      utils::Level::Verbose
    );
  }

}
