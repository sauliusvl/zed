//! Platform-agnostic helpers for damage tracking, occlusion culling, and
//! opaque-quad classification.
//!
//! These let a GPU backend redraw only the region that changed between frames
//! (damage tracking) and avoid overdraw (occlusion culling or an opaque depth
//! pre-pass), without reimplementing the `Scene`-level logic per platform. A
//! backend supplies its own GPU glue (scissoring, depth attachment,
//! presentation); everything here is pure CPU-side analysis of the `Scene`.

use crate::{
    Bounds, DrawOrder, Path, Point, Quad, ScaledPixels, Scene, Size, TransformationMatrix,
};

/// The region of the framebuffer that changed between two frames.
#[derive(Clone, Copy, Debug)]
pub enum SceneDamage {
    /// Everything must be redrawn (e.g. the previous contents are invalid).
    Full,
    /// Only this rectangle changed, in scaled/device pixels.
    Rect(Bounds<ScaledPixels>),
    /// Nothing changed; the frame's presentation can be skipped.
    Unchanged,
}

impl SceneDamage {
    /// Computes the region of `next` that differs from `prev`.
    ///
    /// The diff is sound: it never reports a changed region as unchanged (so it
    /// can't cause stale pixels), though it may over-report when many distant
    /// primitives change. Both scenes must be finished (sorted), which is the
    /// case for any scene handed to a renderer.
    pub fn between(prev: &Scene, next: &Scene) -> SceneDamage {
        let mut acc: Option<Bounds<ScaledPixels>> = None;
        diff_primitives(&prev.quads, &next.quads, |q| q.bounds, &mut acc);
        diff_primitives(
            &prev.shadows,
            &next.shadows,
            |s| s.bounds.dilate(s.blur_radius * 3.0),
            &mut acc,
        );
        diff_primitives(&prev.underlines, &next.underlines, |u| u.bounds, &mut acc);
        diff_primitives(
            &prev.monochrome_sprites,
            &next.monochrome_sprites,
            |s| transformed_bounds(s.bounds, &s.transformation),
            &mut acc,
        );
        diff_primitives(
            &prev.subpixel_sprites,
            &next.subpixel_sprites,
            |s| transformed_bounds(s.bounds, &s.transformation),
            &mut acc,
        );
        diff_primitives(
            &prev.polychrome_sprites,
            &next.polychrome_sprites,
            |s| s.bounds,
            &mut acc,
        );
        diff_paths(&prev.paths, &next.paths, &mut acc);

        match acc {
            Some(rect) => SceneDamage::Rect(rect),
            None => SceneDamage::Unchanged,
        }
    }

    /// Combines two damage regions: the result covers both.
    ///
    /// Renderers use this to accumulate damage across frames whose presentation
    /// failed or was skipped, so no change is lost before it reaches the screen.
    pub fn union(self, other: SceneDamage) -> SceneDamage {
        match (self, other) {
            (SceneDamage::Full, _) | (_, SceneDamage::Full) => SceneDamage::Full,
            (SceneDamage::Unchanged, damage) | (damage, SceneDamage::Unchanged) => damage,
            (SceneDamage::Rect(a), SceneDamage::Rect(b)) => SceneDamage::Rect(a.union(&b)),
        }
    }
}

impl Scene {
    /// The highest draw order across all primitives, or 0 if the scene is empty.
    ///
    /// A renderer can map each primitive's `order` to a depth value relative to
    /// this maximum to drive an opaque depth pre-pass.
    pub fn max_order(&self) -> DrawOrder {
        // Primitive vectors are sorted by order in `Scene::finish`, so each
        // type's maximum is its last element.
        [
            self.quads.last().map(|p| p.order),
            self.shadows.last().map(|p| p.order),
            self.paths.last().map(|p| p.order),
            self.underlines.last().map(|p| p.order),
            self.monochrome_sprites.last().map(|p| p.order),
            self.subpixel_sprites.last().map(|p| p.order),
            self.polychrome_sprites.last().map(|p| p.order),
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(0)
    }

    /// Quads eligible for an opaque depth pre-pass: fully opaque solid fills with
    /// no rounding, border, or clipping, so they can be drawn without a fragment
    /// shader to occlude whatever is behind them.
    ///
    /// Yielded in scene order (ascending by `order`); a back-to-front pre-pass
    /// can `.rev()` this.
    pub fn opaque_quads(&self) -> impl DoubleEndedIterator<Item = &Quad> {
        self.quads.iter().filter(|quad| is_opaque_quad(quad))
    }
}

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

/// Maximum number of occluders considered by [`Scene::cull_occluded`], keeping
/// the cull pass O(primitives × MAX_OCCLUDERS).
const MAX_OCCLUDERS: usize = 16;

/// Occluders smaller than this (in square scaled pixels) are ignored; tiny
/// occluders cost more to test against than they could ever save.
const MIN_OCCLUDER_AREA: f32 = 128.0 * 128.0;

/// Whether CPU occlusion culling is enabled (via `GPUI_OCCLUSION_CULL`).
pub(crate) fn occlusion_cull_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GPUI_OCCLUSION_CULL").is_ok())
}

impl Scene {
    /// Removes primitives that are entirely hidden behind a single opaque quad
    /// with a higher draw order, so no backend spends GPU time on them.
    ///
    /// This is the CPU counterpart to a GPU depth pre-pass: coarser (whole
    /// primitives instead of fragments; a union of occluders doesn't count) but
    /// platform-agnostic and also saves vertex and upload work. Call before
    /// [`Scene::finish`] so the sort shrinks too.
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
        // Test the largest occluders first; they're the most likely to cover.
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

/// Keeps only primitives whose visible bounds aren't fully covered by an
/// occluder with a strictly higher draw order.
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

fn bytes_eq<T: Copy>(a: &T, b: &T) -> bool {
    let size = std::mem::size_of::<T>();
    let a = unsafe { std::slice::from_raw_parts(a as *const T as *const u8, size) };
    let b = unsafe { std::slice::from_raw_parts(b as *const T as *const u8, size) };
    a == b
}

fn slice_bytes_eq<T>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len()
        && unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, std::mem::size_of_val(a)) }
            == unsafe {
                std::slice::from_raw_parts(b.as_ptr() as *const u8, std::mem::size_of_val(b))
            }
}

fn union_into(acc: &mut Option<Bounds<ScaledPixels>>, b: Bounds<ScaledPixels>) {
    *acc = Some(match acc.take() {
        Some(a) => a.union(&b),
        None => b,
    });
}

fn diff_primitives<T: Copy>(
    prev: &[T],
    cur: &[T],
    bounds_of: impl Fn(&T) -> Bounds<ScaledPixels>,
    acc: &mut Option<Bounds<ScaledPixels>>,
) {
    diff_with(prev, cur, |a, b| bytes_eq(a, b), bounds_of, acc);
}

fn diff_paths(
    prev: &[Path<ScaledPixels>],
    cur: &[Path<ScaledPixels>],
    acc: &mut Option<Bounds<ScaledPixels>>,
) {
    diff_with(
        prev,
        cur,
        |a, b| {
            a.order == b.order
                && bytes_eq(&a.bounds, &b.bounds)
                && bytes_eq(&a.color, &b.color)
                && slice_bytes_eq(&a.vertices, &b.vertices)
        },
        |p| p.bounds,
        acc,
    );
}

/// Diffs two primitive slices using a common-prefix / common-suffix comparison.
///
/// Unlike a naive index-by-index diff, this isolates a single inserted or removed
/// element (e.g. the cursor blinking in and out, which shifts every following
/// element) to just the changed window instead of treating everything after it as
/// different. The bounds of every element in the differing window of both frames
/// are accumulated into `acc`.
fn diff_with<T>(
    prev: &[T],
    cur: &[T],
    eq: impl Fn(&T, &T) -> bool,
    bounds_of: impl Fn(&T) -> Bounds<ScaledPixels>,
    acc: &mut Option<Bounds<ScaledPixels>>,
) {
    let max_common = prev.len().min(cur.len());

    let mut prefix = 0;
    while prefix < max_common && eq(&prev[prefix], &cur[prefix]) {
        prefix += 1;
    }
    if prefix == prev.len() && prefix == cur.len() {
        return; // Identical.
    }

    // Match from the end, without overlapping the prefix in either slice.
    let mut suffix = 0;
    while suffix < max_common - prefix
        && eq(&prev[prev.len() - 1 - suffix], &cur[cur.len() - 1 - suffix])
    {
        suffix += 1;
    }

    for p in &prev[prefix..prev.len() - suffix] {
        union_into(acc, bounds_of(p));
    }
    for c in &cur[prefix..cur.len() - suffix] {
        union_into(acc, bounds_of(c));
    }
}

/// Axis-aligned bounds of a sprite after its transformation, so rotated sprites
/// damage their full painted extent.
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

    fn scene_of(quads: &[Quad]) -> Scene {
        let mut scene = Scene::default();
        for q in quads {
            scene.insert_primitive(*q);
        }
        scene.finish();
        scene
    }

    #[test]
    fn identical_scenes_are_unchanged() {
        let a = scene_of(&[quad(rect(0., 0., 100., 100.), 0.5)]);
        let b = scene_of(&[quad(rect(0., 0., 100., 100.), 0.5)]);
        assert!(matches!(
            SceneDamage::between(&a, &b),
            SceneDamage::Unchanged
        ));
    }

    #[test]
    fn changed_quad_damages_its_bounds() {
        let unchanged = quad(rect(0., 0., 100., 100.), 0.1);
        let before = scene_of(&[unchanged, quad(rect(200., 200., 10., 20.), 0.5)]);
        let after = scene_of(&[unchanged, quad(rect(200., 200., 10., 20.), 0.9)]);
        match SceneDamage::between(&before, &after) {
            SceneDamage::Rect(damage) => assert_eq!(damage, rect(200., 200., 10., 20.)),
            other => panic!("expected rect damage, got {other:?}"),
        }
    }

    #[test]
    fn inserted_quad_damages_only_itself() {
        // Mirrors the cursor blinking on: one primitive appears mid-scene,
        // shifting the index of everything after it.
        let a = quad(rect(0., 0., 50., 50.), 0.1);
        let cursor = quad(rect(60., 0., 2., 20.), 0.5);
        let c = quad(rect(100., 0., 50., 50.), 0.9);
        let before = scene_of(&[a, c]);
        let after = scene_of(&[a, cursor, c]);
        match SceneDamage::between(&before, &after) {
            SceneDamage::Rect(damage) => assert_eq!(damage, rect(60., 0., 2., 20.)),
            other => panic!("expected rect damage, got {other:?}"),
        }
    }

    #[test]
    fn removed_quad_damages_only_itself() {
        // The cursor blinking off.
        let a = quad(rect(0., 0., 50., 50.), 0.1);
        let cursor = quad(rect(60., 0., 2., 20.), 0.5);
        let c = quad(rect(100., 0., 50., 50.), 0.9);
        let before = scene_of(&[a, cursor, c]);
        let after = scene_of(&[a, c]);
        match SceneDamage::between(&before, &after) {
            SceneDamage::Rect(damage) => assert_eq!(damage, rect(60., 0., 2., 20.)),
            other => panic!("expected rect damage, got {other:?}"),
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

    #[test]
    fn union_combines_damage() {
        assert!(matches!(
            SceneDamage::Full.union(SceneDamage::Rect(rect(0., 0., 1., 1.))),
            SceneDamage::Full
        ));
        assert!(matches!(
            SceneDamage::Unchanged.union(SceneDamage::Unchanged),
            SceneDamage::Unchanged
        ));
        match SceneDamage::Rect(rect(0., 0., 10., 10.))
            .union(SceneDamage::Rect(rect(20., 20., 10., 10.)))
        {
            SceneDamage::Rect(damage) => assert_eq!(damage, rect(0., 0., 30., 30.)),
            other => panic!("expected rect damage, got {other:?}"),
        }
    }
}
