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

      /**
       * @brief - Create a camera with the specified dimensions, fov, distance
       *          and rotations.
       *          The default camera is aligned with the canonical coordinate
       *          frame (meaning that the `forward` vector is `y`, the lateral
       *          vector is `x` and the up vector is `z`). The rotations are
       *          applied using the Euler convention `zxz`.
       * @param dims - the dimensions of the camera plane.
       * @param fov - the field of view to use: larger values will lead to some
       *              sort of fisheye effect while smaller value will give a
       *              zoom effect.
       * @param distance - the distance from the origin. The rotations apply
       *                   first and then the distance is counted negatively
       *                   along the forward vector.
       * @param rotations - the initial rotations to apply to the camera. Note
       *                    that two values are used, one representing the
       *                    longitudinal rotation and the latitudinal rotation.
       */
      Camera(const utils::Sizei& dims,
             float fov,
             float distance,
             const utils::Vector2f& rotations);

      ~Camera() = default;

      utils::Vector3f
      getEye() const noexcept;

      utils::Vector3f
      getU() const noexcept;

      utils::Vector3f
      getV() const noexcept;

      utils::Vector3f
      getW() const noexcept;

      /**
       * @brief - Used to retrieve the real world direction of a ray starting
       *          at the specified percentage of the camera plane. Each axis
       *          of the `perc` value should range in `[-0.5; 0.5]` but it is
       *          not checked (and the formula stays valid with values outside
       *          of this range).
       * @param perc - a vector representing the position of the ray of the
       *               camera plane for this the real world direction is to be
       *               computed.
       * @return - a direction representing this ray in real world coordinate.
       */
      utils::Vector3f
      getDirection(const utils::Vector2f& perc);

      bool
      setDims(const utils::Sizei& dims);

      /**
       * @brief - Assign a new distance from the origin to the camera. The rest
       *          of the properties are assumed to stay the same (including the
       *          rotations). The new vector describing the camera need to be
       *          fetched through the `getU/V/W` interface after calling this
       *          method.
       * @param distance - the new distance to the camera.
       * @return - `true` if the distance has been updated.
       */
      bool
      setDistance(float distance);

      /**
       * @brief - Apply the rotation angle as an addition to the internal rotation
       *          along the specified angle. Just like the `setDistance` method all
       *          other properties defining the camera are assumed to stay the same.
       * @param angle - a delta that should be added to the existing angle along
       *                the rotation axis. Two values are provided for each of the
       *                possible rotation axes: either `z` or `x`.
       * @return - `true` if at least one of the internal rotation has been updated.
       */
      bool
      rotate(const utils::Vector2f& angle);

    private:

      /**
       * @brief - Used as a threshold below which vertical motions are disabled to
       *          avoid the flip over of the camera. Typically when the rotation
       *          angle about the `x` axis reaches `pi/2 - threshold` any further
       *          rotation will be disabled.
       * @return - a threshold preventing camera flip over.
       */
      static
      float
      getVerticalThreshold() noexcept;

      /**
       * @brief - Used to update the value of the eye position and the `m_u/v/w`
       *          vectors according to the latest values of the fov, dimensions
       *          and distance/rotation.
       *          No check is performed to verify whether it is actually needed.
       */
      void
      updateEUVW();

    private:

      /**
       * @brief - The field of view of the camera.
       */
      float m_fov;

      /**
       * @brief - The dimensions of the focal plane of this camera.
       *          Used as a base to update the `m_u/v/w` vector.
       */
      utils::Sizei m_dims;

      /**
       * @brief - The distance from the eye of the camera and the origin.
       *          Should never be negative.
       */
      float m_distance;

      /**
       * @brief - The rotations currently applied to the initial orientation
       *          of the camera facing the `y` axis and with the `x` axis as
       *          lateral vector.
       */
      utils::Vector2f m_rotations;

      /**
       * @brief- Derived values allowing to ease the process of deriving
       *         a ray from a camera position and orientation.
       */
      utils::Vector3f m_eye;
      utils::Vector3f m_u;
      utils::Vector3f m_v;
      utils::Vector3f m_w;
  };

  using CameraShPtr = std::shared_ptr<Camera>;
}

# include "Camera.hxx"

#endif    /* CAMERA_HH */
