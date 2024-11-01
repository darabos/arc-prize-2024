use std::rc::Rc;

use crate::tools::{Color, Image, Res, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Pixel {
    pub x: i32,
    pub y: i32,
    pub color: Color,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub pixels: Rc<Vec<Pixel>>, // Always sorted. Relative to top left of bb.
    pub bb: Rect,               // Bounding box.
    pub has_relative_colors: bool, // Color numbers are indexes into state.colors.
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Line {
    pub pos: i32,
    pub color: i32,
    pub width: usize,
}
pub type Lines = Vec<Line>;
#[derive(Debug)]
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
}

impl Shape {
    #[must_use]
    pub fn new(mut pixels: Vec<Pixel>) -> Shape {
        assert!(!pixels.is_empty());
        pixels.sort();
        let mut top = std::i32::MAX;
        let mut left = std::i32::MAX;
        let mut bottom = std::i32::MIN;
        let mut right = std::i32::MIN;
        for Pixel { x, y, color: _ } in &pixels {
            top = top.min(*y);
            left = left.min(*x);
            bottom = bottom.max(*y);
            right = right.max(*x);
        }
        Shape {
            pixels: pixels
                .into_iter()
                .map(|p| Pixel {
                    x: p.x - left,
                    y: p.y - top,
                    color: p.color,
                })
                .collect::<Vec<_>>()
                .into(),
            bb: Rect {
                top,
                left,
                bottom,
                right,
            },
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
    pub fn color_at(&self, x: i32, y: i32) -> Option<i32> {
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
        // Slow check by pixel.
        for Pixel { x, y, color: _ } in self.cells() {
            if other.color_at(x, y).is_some() {
                return true;
            }
        }
        false
    }

    #[must_use]
    pub fn move_by(&self, vector: Vec2) -> Shape {
        let mut new_shape = self.clone();
        new_shape.move_by_mut(vector);
        new_shape
    }
    pub fn move_by_mut(&mut self, vector: Vec2) {
        self.bb.top += vector.y;
        self.bb.bottom += vector.y;
        self.bb.left += vector.x;
        self.bb.right += vector.x;
    }
    #[must_use]
    pub fn move_to(&self, vector: Vec2) -> Shape {
        let mut new_shape = self.clone();
        new_shape.move_to_mut(vector);
        new_shape
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

    pub fn recolor(&mut self, color: i32) {
        self.pixels = self
            .pixels
            .iter()
            .map(|p| Pixel {
                x: p.x,
                y: p.y,
                color,
            })
            .collect::<Vec<_>>()
            .into();
    }
    #[must_use]
    pub fn color(&self) -> i32 {
        self.pixels[0].color
    }
    #[must_use]
    pub fn tile(&self, x_step: i32, width: i32, y_step: i32, height: i32) -> Res<Shape> {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in self.cells() {
            for &tx in &[x_step, -x_step] {
                let mut cx = x;
                while cx >= 0 && cx < width {
                    for &ty in &[y_step, -y_step] {
                        let mut cy = y;
                        while cy >= 0 && cy < height {
                            new_cells.push(Pixel {
                                x: cx,
                                y: cy,
                                color: color,
                            });
                            if ty == 0 {
                                break;
                            }
                            cy += ty;
                        }
                    }
                    if tx == 0 {
                        break;
                    }
                    cx += tx;
                }
            }
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
                    color,
                });
            }
        }
        Shape::if_not_empty(new_cells)
    }

    pub fn draw_where_non_empty(&self, image: &mut Image) {
        for Pixel { x, y, color } in self.cells() {
            if image.get_or(x, y, 0) != 0 {
                image[(x as usize, y as usize)] = color;
            }
        }
    }

    pub fn discard_color(&mut self, color: i32) {
        self.pixels = self
            .pixels
            .iter()
            .filter(|cell| cell.color != color)
            .cloned()
            .collect::<Vec<_>>()
            .into();
    }

    #[must_use]
    pub fn from_image(image: &Image) -> Shape {
        let mut cells = vec![];
        for x in 0..image.width {
            for y in 0..image.height {
                let color = image[(x, y)];
                cells.push(Pixel {
                    x: x as i32,
                    y: y as i32,
                    color,
                });
            }
        }
        Shape::new(cells)
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

    pub fn use_relative_colors(&mut self, reverse_colors: &[i32]) {
        self.pixels = self
            .pixels
            .iter()
            .map(|cell| Pixel {
                x: cell.x,
                y: cell.y,
                color: reverse_colors[cell.color as usize],
            })
            .collect::<Vec<_>>()
            .into();
        self.has_relative_colors = true;
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
    pub fn find_matching_shape_index(&self, shapes: &[Rc<Shape>]) -> Option<usize> {
        for (i, shape) in shapes.iter().enumerate() {
            if self.pixels == shape.pixels {
                return Some(i);
            }
        }
        None
    }

    #[must_use]
    pub fn as_image(&self) -> Image {
        let mut image = Image::new(self.bb.width() as usize, self.bb.height() as usize);
        for &Pixel { x, y, color } in self.pixels.iter() {
            image[(x as usize, y as usize)] = color;
        }
        image
    }

    #[must_use]
    pub fn rotate_90_cw(&self) -> Shape {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in self.cells() {
            new_cells.push(Pixel {
                x: -y,
                y: x,
                color: color,
            });
        }
        Shape::new(new_cells)
    }

    #[must_use]
    pub fn flip_horizontal(&self) -> Shape {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in self.cells() {
            new_cells.push(Pixel {
                x: -x,
                y: y,
                color: color,
            });
        }
        Shape::new(new_cells)
    }
}
