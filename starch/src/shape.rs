use std::rc::Rc;

use crate::tools::{Color, Image, MutImage, Res, Vec2, UNSET_COLOR};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Pixel {
    pub x: i32,
    pub y: i32,
    pub color: Color,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub pixels: Rc<Vec<Pixel>>, // Always sorted and relative to the bounding box.
    pub bb: Rect,               // Bounding box.
    pub has_relative_colors: bool, // Color numbers are indexes into state.colors.
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Line {
    pub pos: i32,
    pub color: Color,
    pub width: usize,
}
pub type Lines = Vec<Line>;
#[derive(Debug, Default)]
pub struct LineSet {
    pub horizontal: Lines,
    pub vertical: Lines,
}

impl Pixel {
    pub fn pos(&self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
}

/// Always inclusive. (0, 0, 1, 1) is a 2x2 square.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Rect {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
}
impl Rect {
    pub fn empty() -> Rect {
        Rect {
            top: std::i32::MAX,
            left: std::i32::MAX,
            bottom: std::i32::MIN,
            right: std::i32::MIN,
        }
    }
    pub fn bottom_right(&self) -> Vec2 {
        Vec2 {
            x: self.right - 1,
            y: self.bottom - 1,
        }
    }
    pub fn top_left(&self) -> Vec2 {
        Vec2 {
            x: self.left,
            y: self.top,
        }
    }
    pub fn width(&self) -> i32 {
        self.right - self.left + 1
    }
    pub fn height(&self) -> i32 {
        self.bottom - self.top + 1
    }

    pub fn is_horizontal_line(&self) -> bool {
        self.height() == 1 && self.width() > 1
    }
    pub fn is_vertical_line(&self) -> bool {
        self.width() == 1 && self.height() > 1
    }
    pub fn area(&self) -> i32 {
        self.width() * self.height()
    }
}

impl Shape {
    #[must_use]
    pub fn new(mut pixels: Vec<Pixel>) -> Shape {
        assert!(!pixels.is_empty());
        pixels.sort();
        let mut bb = Rect::empty();
        for &p in &pixels {
            bb.top = bb.top.min(p.y);
            bb.left = bb.left.min(p.x);
            bb.bottom = bb.bottom.max(p.y);
            bb.right = bb.right.max(p.x);
        }
        let mut previous = Vec2 { x: -1, y: -1 };
        for p in &mut pixels {
            p.x -= bb.left;
            p.y -= bb.top;
            assert!(p.x != previous.x || p.y != previous.y, "duplicate pixel");
            previous = p.pos();
        }
        Shape {
            pixels: pixels.into(),
            bb,
            has_relative_colors: false,
        }
    }

    pub fn cells(&self) -> impl Iterator<Item = Pixel> + '_ {
        self.pixels.iter().map(|p| Pixel {
            x: p.x + self.bb.left,
            y: p.y + self.bb.top,
            color: p.color,
        })
    }

    #[must_use]
    pub fn if_not_empty(cells: Vec<Pixel>) -> Res<Shape> {
        if cells.is_empty() {
            return Err("empty");
        }
        Ok(Shape::new(cells))
    }

    #[must_use]
    pub fn color_at(&self, x: i32, y: i32) -> Option<Color> {
        for Pixel {
            x: px,
            y: py,
            color,
        } in self.cells()
        {
            if px == x && py == y {
                return Some(color);
            }
        }
        None
    }
    #[must_use]
    pub fn does_overlap(&self, other: &Shape) -> bool {
        // Quick check by bounding box.
        let a_box = self.bb;
        let b_box = other.bb;
        if a_box.right < b_box.left || a_box.left > b_box.right {
            return false;
        }
        if a_box.bottom < b_box.top || a_box.top > b_box.bottom {
            return false;
        }
        // Slow check by pixel. TODO: The pixels are now sorted. Speed this up.
        for Pixel { x, y, color: _ } in self.cells() {
            if other.color_at(x, y).is_some() {
                return true;
            }
        }
        false
    }

    #[must_use]
    pub fn move_by(&self, vector: Vec2) -> Shape {
        let mut shape = self.clone();
        shape.move_by_mut(vector);
        shape
    }
    pub fn move_by_mut(&mut self, vector: Vec2) {
        self.bb.top += vector.y;
        self.bb.bottom += vector.y;
        self.bb.left += vector.x;
        self.bb.right += vector.x;
    }
    pub fn move_to(&self, vector: Vec2) -> Shape {
        let mut shape = self.clone();
        shape.move_to_mut(vector);
        shape
    }
    pub fn move_to_mut(&mut self, vector: Vec2) {
        self.move_by_mut(vector - self.bb.top_left());
    }
    /// Returns true if the shape matches the image at the given position.
    /// Returns false if the shape is entirely out of bounds.
    pub fn matches_image_when_moved_by(&self, image: &Image, vector: Vec2) -> bool {
        let mut matched_count = 0;
        for Pixel { x, y, color } in self.cells() {
            let nx = x + vector.x;
            let ny = y + vector.y;
            if nx < 0 || ny < 0 || nx >= image.width as i32 || ny >= image.height as i32 {
                continue;
            }
            let icolor = image[(nx as usize, ny as usize)];
            if icolor == 0 {
                continue;
            }
            if icolor != color {
                return false;
            }
            matched_count += 1;
        }
        matched_count > 0
        // TODO: Why isn't this better?
        // matched_count >= 2 && matched_count >= self.cells.len() / 2
    }

    #[must_use]
    pub fn recolor(&self, color: Color) -> Shape {
        Shape::new(
            self.cells()
                .map(|cell| Pixel {
                    x: cell.x,
                    y: cell.y,
                    color,
                })
                .collect(),
        )
    }
    #[must_use]
    pub fn color(&self) -> usize {
        self.pixels[0].color
    }
    #[must_use]
    pub fn cell0(&self) -> Pixel {
        self.cells().next().unwrap()
    }
    #[must_use]
    /// x_step is the tiling width (can be 0 for no tiling in x). width is the width of the tiling area.
    /// Same for y_step and height.
    pub fn tile(&self, x_step: i32, width: i32, y_step: i32, height: i32) -> Res<Shape> {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in self.cells() {
            let mut cx = if x_step == 0 { x } else { x % x_step };
            while cx < width {
                let mut cy = if y_step == 0 { y } else { y % y_step };
                while cy < height {
                    new_cells.push(Pixel {
                        x: cx,
                        y: cy,
                        color: color,
                    });
                    if y_step == 0 {
                        break;
                    } else {
                        cy += y_step;
                    }
                }
                if x_step == 0 {
                    break;
                } else {
                    cx += x_step;
                }
            }
        }
        if x_step != 0 && x_step < self.bb.width() || y_step != 0 && y_step < self.bb.height() {
            // The shifted bounding boxes overlap, so we may have duplicate pixels. Remove them.
            new_cells.sort();
            new_cells.dedup_by(|a, b| a.x == b.x && a.y == b.y);
        }
        Shape::if_not_empty(new_cells)
    }

    #[must_use]
    pub fn crop(&self, left: i32, top: i32, right: i32, bottom: i32) -> Res<Shape> {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in self.cells() {
            if x >= left && x <= right && y >= top && y <= bottom {
                new_cells.push(Pixel {
                    x: x - left,
                    y: y - top,
                    color: color,
                });
            }
        }
        Shape::if_not_empty(new_cells)
    }

    pub fn draw_where_non_empty(&self, image: &mut MutImage) {
        for Pixel { x, y, color } in self.cells() {
            if image.get_or(x, y, 0) != 0 {
                image[(x as usize, y as usize)] = color;
            }
        }
    }

    #[must_use]
    pub fn discard_color(&self, color: Color) -> Res<Shape> {
        let remaining: Vec<Pixel> = self.cells().filter(|cell| cell.color != color).collect();
        if remaining.is_empty() {
            return Err("nothing left after discarding color");
        }
        Ok(Shape::new(remaining))
    }

    #[must_use]
    pub fn from_image(image: &Image) -> Res<Shape> {
        let mut cells = vec![];
        for x in 0..image.width {
            for y in 0..image.height {
                let color = image[(x, y)];
                if color != 0 {
                    cells.push(Pixel {
                        x: x as i32,
                        y: y as i32,
                        color,
                    });
                }
            }
        }
        if cells.is_empty() {
            return Err("empty");
        }
        Ok(Shape::new(cells))
    }

    #[must_use]
    pub fn is_touching_border(&self, image: &Image) -> bool {
        for Pixel { x, y, color: _ } in self.cells() {
            if x == 0 || y == 0 || x == image.width as i32 - 1 || y == image.height as i32 - 1 {
                return true;
            }
        }
        false
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        println!("top left: {}, {}", self.bb.left, self.bb.top);
        println!("{}", self.as_image());
    }

    pub fn use_relative_colors(&mut self, reverse_colors: &[usize]) -> Shape {
        Shape::new(
            self.cells()
                .map(|cell| Pixel {
                    x: cell.x,
                    y: cell.y,
                    color: reverse_colors[cell.color],
                })
                .collect(),
        )
    }

    #[must_use]
    pub fn covers(&self, other: &Shape) -> bool {
        for Pixel { x, y, color: _ } in other.cells() {
            if self.color_at(x, y).is_none() {
                return false;
            }
        }
        true
    }

    /// Requires exact match.
    #[must_use]
    pub fn find_matching_shape_index(&self, shapes: &[Shape]) -> Option<usize> {
        for (i, shape) in shapes.iter().enumerate() {
            if self.pixels == shape.pixels {
                return Some(i);
            }
        }
        None
    }

    /// Fast bounding box-based distance.
    #[must_use]
    pub fn distance_to(&self, other: &Shape) -> i32 {
        let dx = (self.bb.left + self.bb.right - other.bb.left - other.bb.right).abs();
        let dy = (self.bb.top + self.bb.bottom - other.bb.top - other.bb.bottom).abs();
        dx + dy
    }

    #[must_use]
    pub fn find_nearest_shape_index(&self, shapes: &[Shape]) -> usize {
        shapes
            .iter()
            .enumerate()
            .min_by_key(|(_, shape)| self.distance_to(shape))
            .unwrap()
            .0
    }

    #[must_use]
    pub fn as_image(&self) -> Image {
        let mut image = MutImage::new(self.bb.width() as usize, self.bb.height() as usize);
        for Pixel { x, y, color } in self.cells() {
            image[((x - self.bb.left) as usize, (y - self.bb.top) as usize)] = color;
        }
        image.freeze()
    }

    #[must_use]
    pub fn rotate_90_cw(&self) -> Shape {
        let new_cells: Vec<Pixel> = self
            .cells()
            .map(|cell| Pixel {
                x: -cell.y,
                y: cell.x,
                color: cell.color,
            })
            .collect();
        Shape::new(new_cells)
    }

    #[must_use]
    pub fn flip_horizontal(&self) -> Shape {
        let new_cells = self
            .cells()
            .map(|cell| Pixel {
                x: -cell.x,
                y: cell.y,
                color: cell.color,
            })
            .collect();
        Shape::new(new_cells)
    }
}
