# Video Converter border rendering V2

**Draw black border** expands the alpha silhouette outward using a 4× supersampled mask. The selected color is composited behind the source through straight-alpha blending. A fixed transparent gutter of `border width + 2` pixels is added to every frame so the new contour is not clipped and animation dimensions stay consistent. The default border remains 1 px. With smoothing disabled, opaque source RGB is unchanged.

**Smooth border transition** uses the existing **Outline contour smoothing** slider as an inward transition depth (1–16 output pixels), measured from the inner edge of the outward outline. The outward contour and alpha stay identical at all smoothing settings. A low-strength, tapered blend of the chosen border color affects only this inner band. Only the blend weight is softened; artwork RGB is never blurred. Pixels already close to the border color, such as dark lineart with a black border, receive little or no additional coloring. Both selected-frame preview and full export use the same finishing function.

## Real animation comparison

Input: `feet2_extra_przerywnik.mp4`, Drawn/comics + white background, 1 px black border, smoothing setting 4. In the earlier V2 comparison on frames 0 and 75, the old inward implementation changed respectively 11,494 and 12,802 opaque artwork pixels relative to a borderless result. The outward-only implementation changed none. The current optional transition intentionally blends a narrow inner band, with a lower maximum strength than the old implementation; it does not extend the border farther outside the silhouette.

With the updated transition, preview frames 0 and 75 both kept exactly the same alpha mask as the unsmoothed outward border. The inward blend changed 8,636 and 9,672 opaque pixels respectively, with maximum single-channel differences of 80 and 81 levels. The interior beyond the configured depth is untouched. The processed-frame cache version was advanced so older preview/export results cannot mask this change.

With the inward transition enabled, a six-frame PNG export from the actual MP4 completed with 560×746 RGBA frames (from a 554×740 source) and preserved source delays `[33, 34, 33, 33, 34, 33]` ms. Preview frame 0 matched its exported PNG byte for byte. The entire 150-frame animation was not exported in this validation.
