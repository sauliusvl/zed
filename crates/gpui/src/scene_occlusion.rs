//! CPU occlusion culling: drops scene primitives that are entirely hidden
//! behind an opaque quad, so no backend spends time drawing them.

use crate::{Bounds, DrawOrder, Point, Quad, ScaledPixels, Scene, Size, TransformationMatrix};

/// Occluders are capped to keep the pass O(primitives × MAX_OCCLUDERS).
const MAX_OCCLUDERS: usize = 16;

/// Occluders smaller than this (in square scaled pixels) aren't worth testing
/// against: they cost more than they could ever save.
const MIN_OCCLUDER_AREA: f32 = 128.0 * 128.0;

impl Scene {
    /// Removes primitives entirely covered by an opaque quad with a higher draw
    /// order. Conservative (whole primitives only, single occluder), but cheap
    /// and platform-agnostic. Call before [`Scene::finish`] so the sort shrinks.
    pub fn cull_occluded(&mut self) {
        let area = |bounds: &Bounds<ScaledPixels>| bounds.size.width.0 * bounds.size.height.0;

        let mut occluders: Vec<(DrawOrder, Bounds<ScaledPixels>)> = self
            .quads
            .iter()
            .filter(|quad| area(&quad.bounds) >= MIN_OCCLUDER_AREA && is_opaque_quad(quad))
            .map(|quad| (quad.order, quad.bounds))
            .collect();
        if occluders.is_empty() {
            return;
        }
        // Largest occluders first; they're the most likely to cover something.
        occluders.sort_unstable_by(|a, b| {
            area(&b.1)
                .partial_cmp(&area(&a.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        occluders.truncate(MAX_OCCLUDERS);

        retain_visible(
            &mut self.quads,
            &occluders,
            |q| q.order,
            |q| q.bounds.intersect(&q.content_mask.bounds),
        );
        retain_visible(
            &mut self.shadows,
            &occluders,
            |s| s.order,
            |s| {
                s.bounds
                    .dilate(s.blur_radius * 3.0)
                    .intersect(&s.content_mask.bounds)
            },
        );
        retain_visible(
            &mut self.underlines,
            &occluders,
            |u| u.order,
            |u| u.bounds.intersect(&u.content_mask.bounds),
        );
        retain_visible(
            &mut self.monochrome_sprites,
            &occluders,
            |s| s.order,
            |s| transformed_bounds(s.bounds, &s.transformation).intersect(&s.content_mask.bounds),
        );
        retain_visible(
            &mut self.subpixel_sprites,
            &occluders,
            |s| s.order,
            |s| transformed_bounds(s.bounds, &s.transformation).intersect(&s.content_mask.bounds),
        );
        retain_visible(
            &mut self.polychrome_sprites,
            &occluders,
            |s| s.order,
            |s| s.bounds.intersect(&s.content_mask.bounds),
        );
        retain_visible(
            &mut self.paths,
            &occluders,
            |p| p.order,
            |p| p.clipped_bounds(),
        );
        retain_visible(
            &mut self.surfaces,
            &occluders,
            |s| s.order,
            |s| s.bounds.intersect(&s.content_mask.bounds),
        );
    }
}

/// A quad usable as an occluder: a fully opaque solid fill with no rounding,
/// border, or clipping, so everything behind it is truly hidden.
fn is_opaque_quad(quad: &Quad) -> bool {
    let solid_opaque = quad
        .background
        .as_solid()
        .is_some_and(|color| color.is_opaque());
    if !solid_opaque {
        return false;
    }

    let unrounded = quad.corner_radii.top_left.0 == 0.0
        && quad.corner_radii.top_right.0 == 0.0
        && quad.corner_radii.bottom_left.0 == 0.0
        && quad.corner_radii.bottom_right.0 == 0.0;
    let unbordered = quad.border_widths.top.0 == 0.0
        && quad.border_widths.right.0 == 0.0
        && quad.border_widths.bottom.0 == 0.0
        && quad.border_widths.left.0 == 0.0;

    unrounded && unbordered && covers(&quad.content_mask.bounds, &quad.bounds)
}

/// Whether `outer` fully contains `inner`.
fn covers(outer: &Bounds<ScaledPixels>, inner: &Bounds<ScaledPixels>) -> bool {
    outer.origin.x.0 <= inner.origin.x.0
        && outer.origin.y.0 <= inner.origin.y.0
        && outer.origin.x.0 + outer.size.width.0 >= inner.origin.x.0 + inner.size.width.0
        && outer.origin.y.0 + outer.size.height.0 >= inner.origin.y.0 + inner.size.height.0
}

/// Drops primitives fully covered by an occluder with a strictly higher order.
fn retain_visible<T>(
    primitives: &mut Vec<T>,
    occluders: &[(DrawOrder, Bounds<ScaledPixels>)],
    order_of: impl Fn(&T) -> DrawOrder,
    bounds_of: impl Fn(&T) -> Bounds<ScaledPixels>,
) {
    primitives.retain(|primitive| {
        let order = order_of(primitive);
        let bounds = bounds_of(primitive);
        !occluders
            .iter()
            .any(|(occluder_order, occluder)| *occluder_order > order && covers(occluder, &bounds))
    });
}

/// Axis-aligned bounds of a sprite after its transformation.
fn transformed_bounds(
    bounds: Bounds<ScaledPixels>,
    transform: &TransformationMatrix,
) -> Bounds<ScaledPixels> {
    let rs = transform.rotation_scale;
    let t = transform.translation;
    let x0 = bounds.origin.x.0;
    let y0 = bounds.origin.y.0;
    let x1 = x0 + bounds.size.width.0;
    let y1 = y0 + bounds.size.height.0;
    let mut min = (f32::MAX, f32::MAX);
    let mut max = (f32::MIN, f32::MIN);
    for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
        // Matches the shader: transpose(rotation_scale) * position + translation.
        let tx = rs[0][0] * x + rs[1][0] * y + t[0];
        let ty = rs[0][1] * x + rs[1][1] * y + t[1];
        min.0 = min.0.min(tx);
        min.1 = min.1.min(ty);
        max.0 = max.0.max(tx);
        max.1 = max.1.max(ty);
    }
    Bounds {
        origin: Point {
            x: ScaledPixels(min.0),
            y: ScaledPixels(min.1),
        },
        size: Size {
            width: ScaledPixels(max.0 - min.0),
            height: ScaledPixels(max.1 - min.1),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentMask, Hsla};

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: ScaledPixels(x),
                y: ScaledPixels(y),
            },
            size: Size {
                width: ScaledPixels(w),
                height: ScaledPixels(h),
            },
        }
    }

    fn quad(bounds: Bounds<ScaledPixels>, lightness: f32) -> Quad {
        Quad {
            bounds,
            content_mask: ContentMask {
                bounds: rect(0., 0., 1000., 1000.),
            },
            background: Hsla {
                h: 0.,
                s: 0.,
                l: lightness,
                a: 1.,
            }
            .into(),
            ..Default::default()
        }
    }

    #[test]
    fn culls_quad_fully_covered_by_higher_order_opaque_quad() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(rect(10., 10., 50., 50.), 0.3));
        // Overlaps the first quad, so it gets a higher draw order.
        scene.insert_primitive(quad(rect(0., 0., 500., 500.), 0.6));
        scene.cull_occluded();
        assert_eq!(scene.quads.len(), 1);
        assert_eq!(scene.quads[0].bounds, rect(0., 0., 500., 500.));
    }

    #[test]
    fn keeps_partially_covered_quad() {
        let mut scene = Scene::default();
        // Extends past the occluder's right/bottom edge.
        scene.insert_primitive(quad(rect(400., 400., 200., 200.), 0.3));
        scene.insert_primitive(quad(rect(0., 0., 500., 500.), 0.6));
        scene.cull_occluded();
        assert_eq!(scene.quads.len(), 2);
    }

    #[test]
    fn keeps_quad_in_front_of_occluder() {
        let mut scene = Scene::default();
        // The big opaque quad is painted first, so it's *behind* the small one.
        scene.insert_primitive(quad(rect(0., 0., 500., 500.), 0.6));
        scene.insert_primitive(quad(rect(10., 10., 50., 50.), 0.3));
        scene.cull_occluded();
        assert_eq!(scene.quads.len(), 2);
    }

    #[test]
    fn transparent_or_small_quads_do_not_occlude() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(rect(10., 10., 50., 50.), 0.3));
        // Fully covers, but is translucent.
        let mut translucent = quad(rect(0., 0., 500., 500.), 0.6);
        translucent.background = Hsla {
            h: 0.,
            s: 0.,
            l: 0.6,
            a: 0.5,
        }
        .into();
        scene.insert_primitive(translucent);
        // Fully covers and is opaque, but is below the size threshold.
        scene.insert_primitive(quad(rect(0., 0., 100., 100.), 0.9));
        scene.cull_occluded();
        assert_eq!(scene.quads.len(), 3);
    }
}
