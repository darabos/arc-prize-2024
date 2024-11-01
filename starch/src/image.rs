use crate::tools::{write_color, Color, Pixel, Res, Shape, Vec2};

impl std::ops::Index<(usize, usize)> for Image {
    type Output = Color;
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        match &self.sub_image {
            Some(SubImageSpec {
                top,
                left,
                full_image,
            }) => {
                let (x, y) = (x + left, y + top);
                full_image.index((x, y))
            }
            None => &self.pixels[y * self.width + x],
        }
    }
}

impl std::ops::IndexMut<(usize, usize)> for Image {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Color {
        match &mut self.sub_image {
            Some(SubImageSpec {
                top,
                left,
                full_image,
            }) => {
                let (x, y) = (x + *left, y + *top);
                full_image.index_mut((x, y))
            }
            None => &mut self.pixels[y * self.width + x],
        }
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
pub struct SubImageSpec {
    pub top: usize,
    pub left: usize,
    pub full_image: Box<Image>,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Color>,
    pub sub_image: Option<SubImageSpec>,
}

impl Image {
    pub fn new(width: usize, height: usize) -> Image {
        Image {
            width,
            height,
            pixels: vec![0; width * height],
            sub_image: None,
        }
    }
    pub fn sub_image(&self, left: usize, top: usize, width: usize, height: usize) -> Image {
        assert!(left + width <= self.width);
        assert!(top + height <= self.height);
        match &self.sub_image {
            Some(SubImageSpec {
                top,
                left,
                full_image,
            }) => full_image.sub_image(left + left, top + top, width, height),
            None => Image {
                width,
                height,
                pixels: vec![],
                sub_image: Some(SubImageSpec {
                    top,
                    left,
                    full_image: Box::new(self.clone()),
                }),
            },
        }
    }
    pub fn is_empty(&self) -> bool {
        self.width == 0 && self.height == 0
    }
    pub fn full(&self) -> &Image {
        match &self.sub_image {
            Some(SubImageSpec {
                top: _,
                left: _,
                full_image,
            }) => full_image.full(),
            None => self,
        }
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

    pub fn draw_image_at(&mut self, other: &Image, pos: Vec2) {
        for y in 0..other.height {
            for x in 0..other.width {
                let nx = pos.x + x as i32;
                let ny = pos.y + y as i32;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }
                self[(nx as usize, ny as usize)] = other[(x as usize, y as usize)];
            }
        }
    }
}
