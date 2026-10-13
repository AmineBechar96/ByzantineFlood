# Physically based water overlay (tasks 18-21) -- to build

Renders a horizontal water plane at a chosen height h on a real dry fisheye
frame; the label is h, exact by construction.

1. Ground segmentation per camera (SAM or similar), once.
2. Ground plane from mounting height (5.1-7.6 m), tilt and fisheye
   intrinsics (OpenCV fisheye model).
3. Waterline: where the plane at height h cuts kerbs, poles, walls, cars.
4. Below the waterline: mirrored reflection, Fresnel blend by viewing
   angle, darkened and tinted submerged content, low-amplitude ripple noise.
5. Validate against known heights (kerb, pole) visible in frame.

Levels follow Table II of the Section II draft (reference-object criteria).
Overlay images are training data only.
