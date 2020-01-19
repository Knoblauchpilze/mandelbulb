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

      bool
      setDims(const utils::Sizei& dims);

    private:

      /**
       * @brief - Used to copy the transform to the local data by stripping
       *          the unneeded values.
       * @param transform - the data to copy.
       */
      void
      copyTransform(const std::vector<float>& rawTransform);

      /**
       * @brief - Used to update the internal properties assuming that the dimensions
       *          should be set to the specified value. We consider that the rest of
       *          the `base` properties should stay the same.
       *          Note that we don't perform any check on the input dimensions to see
       *          whether they are valid. If this is not the case UB may arise.
       * @param dims - the new dimensions of the camera.
       */
      void
      updateDims(const utils::Sizei& dims);

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
       * @brief - The transformation applied to the camera. Stored as
       *          a row major `4x3` matrix where the layout is described
       *          below:
       *
       *          | r00 r10 r20 t0 |
       *          | r01 r11 r21 t1 |
       *          | r02 r12 r22 t2 |
       */
      std::vector<float> m_transform;

      /**
       * @brief - Derived properties computed from the dimensions of
       *          the focal plane and the field of view.
       */
      float m_focal;

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
