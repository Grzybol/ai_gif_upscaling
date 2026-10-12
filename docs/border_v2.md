# Video Converter border rendering V2

**Draw black border** now expands the alpha silhouette outward using a 4× supersampled mask. The selected color is composited behind the source through straight-alpha blending; opaque source RGB is unchanged. A fixed transparent gutter of `border width + 2` pixels is added to every frame so the new contour is not clipped and animation dimensions stay consistent. The default border remains 1 px. Existing dark artwork lines are not darkened a second time when the border is enabled.

**Smooth border transition** now feathers only the supersampled contour geometry. Its existing slider controls a subpixel feather, up to one output pixel at the maximum setting. It no longer blends black through multiple inward rings of artwork. Both selected-frame preview and full export use the same finishing function.

## Real animation comparison

Input: `feet2_extra_przerywnik.mp4`, Drawn/comics + white background, 1 px black border, smoothing setting 4. On frames 0 and 75, the previous inward implementation changed respectively 11,494 and 12,802 opaque artwork pixels relative to a borderless white-background result. V2 changed **zero opaque artwork RGB pixels**; its color appears in newly covered alpha outside the original silhouette. Close-up inspection of hair, fingers, toes, and chair edges showed less dark intrusion into the artwork. Some source-keying specks remain visible because the border pass deliberately does not alter the interior.

A six-frame export from the actual MP4 completed in 9.55 s. It produced 560×746 RGBA frames (from a 554×740 source), a VP9 WebM with six frames and the source delays `[33, 34, 33, 33, 34, 33]` ms, and preserved alpha. Preview frames 0 and 4 matched the corresponding exported PNGs byte for byte. The entire 150-frame animation was not exported in this validation.
