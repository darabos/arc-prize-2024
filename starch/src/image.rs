use crate::tools::{write_color, Color, Pixel, Res, Shape, Vec2};

impl std::ops::Index<(usize, usize)> for Image {
    type Output = Color;
    fn index(&self, (x, y): (usize, usize)) -> &Color {
        &self.pixels[self.full_width * (self.top + y) + self.left + x]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Image {
    fn index_mut<'a>(&mut self, (x, y): (usize, usize)) -> &mut Color {
        &mut self.pixels[self.full_width * (self.top + y) + self.left + x]
    }
}
impl std::fmt::Display for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                write_color(f, self[(x, y)]);
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    pub top: usize,
    pub left: usize,
    pub width: usize,
    pub height: usize,
    pub full_width: usize,
    pub full_height: usize,
    pixels: Vec<Color>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Image {
        Image {
            top: 0,
            left: 0,
            width,
            height,
            full_width: width,
            full_height: height,
            pixels: vec![0; width * height],
        }
    }
    pub fn subimage(&self, left: usize, top: usize, width: usize, height: usize) -> Image {
        assert!(left + width <= self.width);
        assert!(top + height <= self.height);
        Image {
            top: self.top + top,
            left: self.left + left,
            width,
            height,
            full_width: self.full_width,
            full_height: self.full_height,
            pixels: self.pixels.clone(),
        }
    }
    pub fn is_zoomed(&self) -> bool {
        self.top > 0
            || self.left > 0
            || self.width != self.full_width
            || self.height != self.full_height
    }
    pub fn full(&self) -> Image {
        Image {
            top: 0,
            left: 0,
            width: self.full_width,
            height: self.full_height,
            full_width: self.full_width,
            full_height: self.full_height,
            pixels: self.pixels.clone(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.width == 0 && self.height == 0
    }
    pub fn update(&mut self, f: impl Fn(usize, usize, Color) -> Color) {
        for y in 0..self.height {
            for x in 0..self.width {
                self[(x, y)] = f(x, y, self[(x, y)]);
            }
        }
    }
    #[must_use]
    pub fn try_update(&mut self, f: impl Fn(usize, usize, Color) -> Res<Color>) -> Res<()> {
        for y in 0..self.height {
            for x in 0..self.width {
                self[(x, y)] = f(x, y, self[(x, y)])?;
            }
        }
        Ok(())
    }
    /// Returns an iterator over all the color values.
    pub fn colors_iter(&self) -> impl Iterator<Item = Color> + '_ {
        (0..self.width * self.height).map(|i| self[(i % self.width, i / self.width)])
    }
    pub fn get(&self, x: i32, y: i32) -> Res<Color> {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return Err("out of bounds");
        }
        Ok(self[(x as usize, y as usize)])
    }
    pub fn get_or(&self, x: i32, y: i32, default: Color) -> Color {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return default;
        }
        self[(x as usize, y as usize)]
    }
    #[must_use]
    pub fn set(&mut self, x: i32, y: i32, color: Color) -> Res<()> {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return Err("out of bounds");
        }
        self[(x as usize, y as usize)] = color;
        Ok(())
    }
    pub fn print(&self) {
        println!("{}", self);
    }
    pub fn from_vecvec(vecvec: Vec<Vec<Color>>) -> Image {
        let height = vecvec.len();
        let width = if height == 0 { 0 } else { vecvec[0].len() };
        let mut image = Image::new(width, height);
        for y in 0..height {
            for x in 0..width {
                image[(x, y)] = vecvec[y][x];
            }
        }
        image
    }

    pub fn crop(&self, left: i32, top: i32, width: i32, height: i32) -> Image {
        assert!(left >= 0);
        assert!(top >= 0);
        assert!(left + width <= self.width as i32);
        assert!(top + height <= self.height as i32);
        assert!(width >= 0);
        assert!(height >= 0);
        let mut new_image = Image::new(width as usize, height as usize);
        for y in 0..height {
            for x in 0..width {
                let nx = left + x;
                let ny = top + y;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }
                new_image[(x as usize, y as usize)] = self[(nx as usize, ny as usize)];
            }
        }
        new_image
    }

    pub fn clear(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                self[(x, y)] = 0;
            }
        }
    }

    /// Draws the image in the given color.
    pub fn paint_shape(&mut self, shape: &Shape, color: i32) {
        for Pixel { x, y, color: _ } in shape.cells() {
            if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
                continue;
            }
            self[(x as usize, y as usize)] = color;
        }
    }

    pub fn erase_shape(&mut self, shape: &Shape) {
        self.paint_shape(&shape, 0)
    }
    /// Draws the shape in its original color.
    pub fn draw_shape(&mut self, shape: &Shape) {
        for Pixel { x, y, color } in shape.cells() {
            if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
                continue;
            }
            self[(x as usize, y as usize)] = color;
        }
    }
    pub fn draw_shape_with_colors(&mut self, shape: &Shape, colors: &[i32]) {
        for Pixel { x, y, color } in shape.cells() {
            if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
                continue;
            }
            self[(x as usize, y as usize)] = colors[color as usize];
        }
    }
    pub fn draw_shape_at(&mut self, shape: &Shape, pos: Vec2) {
        for Pixel { x, y, color } in shape.cells() {
            let nx = pos.x + x;
            let ny = pos.y + y;
            if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                continue;
            }
            self[(nx as usize, ny as usize)] = color;
        }
    }

    /// With transparency.
    pub fn draw_image_at(&mut self, other: &Image, pos: Vec2) {
        for y in 0..other.height {
            for x in 0..other.width {
                let nx = pos.x + x as i32;
                let ny = pos.y + y as i32;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }
                if other[(x, y)] != 0 {
                    self[(nx as usize, ny as usize)] = other[(x, y)];
                }
            }
        }
    }
}
