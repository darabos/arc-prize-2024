use colored;

type Color = i32;

pub const COLORS: [colored::Color; 12] = [
    colored::Color::BrightWhite,
    colored::Color::Black,
    colored::Color::Blue,
    colored::Color::Red,
    colored::Color::Green,
    colored::Color::Yellow,
    colored::Color::TrueColor {
        r: 128,
        g: 0,
        b: 128,
    },
    colored::Color::TrueColor {
        r: 255,
        g: 165,
        b: 0,
    },
    colored::Color::TrueColor {
        r: 165,
        g: 42,
        b: 42,
    },
    colored::Color::Magenta,
    colored::Color::White,
    colored::Color::Cyan,
];

pub type Image = Vec<Vec<Color>>;
#[derive(Clone, Default)]
pub struct Task {
    pub train: Vec<Example>,
    pub test: Vec<Example>,
}

#[derive(Clone)]
pub struct Example {
    pub input: Image,
    pub output: Image,
}

pub type Res<T> = Result<T, &'static str>;

#[derive(Clone, PartialEq)]
pub struct Vec2 {
    pub x: i32,
    pub y: i32,
}
#[derive(Clone, PartialEq)]
pub struct Pixel {
    pub x: i32,
    pub y: i32,
    pub color: Color,
}

#[derive(Clone, Default)]
pub struct Shape {
    pub cells: Vec<Pixel>,
}

pub const UP: Vec2 = Vec2 { x: 0, y: -1 };
pub const DOWN: Vec2 = Vec2 { x: 0, y: 1 };
pub const LEFT: Vec2 = Vec2 { x: -1, y: 0 };
pub const RIGHT: Vec2 = Vec2 { x: 1, y: 0 };
pub const DIRECTIONS: [Vec2; 4] = [UP, DOWN, LEFT, RIGHT];

pub fn find_shapes_in_image(image: &Image) -> Vec<Shape> {
    let mut shapes = vec![];
    let mut visited = vec![vec![false; image[0].len()]; image.len()];
    for y in 0..image.len() {
        for x in 0..image[0].len() {
            if visited[y][x] {
                continue;
            }
            let color = image[y][x];
            if color == 0 {
                continue;
            }
            let mut cells = vec![Pixel {
                x: x as i32,
                y: y as i32,
                color,
            }];
            visited[y][x] = true;
            let mut i = 0;
            while i < cells.len() {
                let Pixel { x, y, color: _ } = cells[i];
                for dir in &DIRECTIONS {
                    let nx = x + dir.x;
                    let ny = y + dir.y;
                    if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
                        continue;
                    }
                    if visited[ny as usize][nx as usize] {
                        continue;
                    }
                    if image[ny as usize][nx as usize] != color {
                        continue;
                    }
                    visited[ny as usize][nx as usize] = true;
                    cells.push(Pixel {
                        x: nx,
                        y: ny,
                        color,
                    });
                }
                i += 1;
            }
            shapes.push(Shape { cells });
        }
    }
    shapes
}

/// Finds "colorsets" in the image. A colorset is a set of all pixels with the same color.
pub fn find_colorsets_in_image(image: &Image) -> Vec<Shape> {
    // Create blank colorset for each color.
    let mut colorsets = vec![Shape::default(); COLORS.len()];
    for y in 0..image.len() {
        for x in 0..image[0].len() {
            let color = image[y][x];
            if color == 0 {
                continue;
            }
            colorsets[color as usize].cells.push(Pixel {
                x: x as i32,
                y: y as i32,
                color,
            });
        }
    }
    // Filter non-empty colorsets.
    colorsets = colorsets
        .into_iter()
        .filter(|colorset| !colorset.cells.is_empty())
        .collect();
    colorsets
}

pub fn shape_by_color(shapes: &[Shape], color: i32) -> Option<&Shape> {
    for shape in shapes {
        if shape.color() == color {
            return Some(shape);
        }
    }
    None
}

impl Pixel {
    pub fn pos(&self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
}

pub struct Rect {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
}

impl Shape {
    pub fn bounding_box(&self) -> Rect {
        let mut top = std::i32::MAX;
        let mut left = std::i32::MAX;
        let mut bottom = std::i32::MIN;
        let mut right = std::i32::MIN;
        for Pixel { x, y, color: _ } in &self.cells {
            top = top.min(*y);
            left = left.min(*x);
            bottom = bottom.max(*y);
            right = right.max(*x);
        }
        Rect {
            top,
            left,
            bottom,
            right,
        }
    }

    pub fn color_at(&self, x: i32, y: i32) -> Option<i32> {
        for Pixel {
            x: px,
            y: py,
            color,
        } in &self.cells
        {
            if *px == x && *py == y {
                return Some(*color);
            }
        }
        None
    }
    pub fn does_overlap(&self, other: &Shape) -> bool {
        // Quick check by bounding box.
        let a_box = self.bounding_box();
        let b_box = other.bounding_box();
        if a_box.right < b_box.left || a_box.left > b_box.right {
            return false;
        }
        if a_box.bottom < b_box.top || a_box.top > b_box.bottom {
            return false;
        }
        // Slow check by pixel.
        for Pixel { x, y, color: _ } in &self.cells {
            if other.color_at(*x, *y).is_some() {
                return true;
            }
        }
        false
    }

    pub fn move_by(&self, vector: Vec2) -> Shape {
        let cells = self
            .cells
            .iter()
            .map(|Pixel { x, y, color }| Pixel {
                x: *x + vector.x,
                y: *y + vector.y,
                color: *color,
            })
            .collect();
        Shape { cells }
    }

    pub fn recolor(&mut self, color: i32) {
        for cell in &mut self.cells {
            cell.color = color;
        }
    }
    pub fn color(&self) -> i32 {
        self.cells[0].color
    }
}

/// Draws the image in the given color.
pub fn paint_shape(image: &Image, shape: &Shape, color: i32) -> Image {
    let mut new_image = image.clone();
    for Pixel { x, y, color: _ } in &shape.cells {
        if *x < 0 || *y < 0 || *x >= image[0].len() as i32 || *y >= image.len() as i32 {
            continue;
        }
        new_image[*y as usize][*x as usize] = color;
    }
    new_image
}

pub fn remove_shape(image: &Image, shape: &Shape) -> Image {
    paint_shape(image, &shape, 0)
}
/// Draws the shape in its original color.
pub fn draw_shape(image: &Image, shape: &Shape) -> Image {
    let mut new_image = image.clone();
    for Pixel { x, y, color } in &shape.cells {
        if *x < 0 || *y < 0 || *x >= image[0].len() as i32 || *y >= image.len() as i32 {
            continue;
        }
        new_image[*y as usize][*x as usize] = *color;
    }
    new_image
}

// Moves the first shape pixel by pixel. (Not using bounding boxes.)
pub fn move_shape_to_shape_in_direction(
    image: &Image,
    to_move: &Shape,
    move_to: &Shape,
    dir: Vec2,
) -> Res<Image> {
    // Figure out moving distance.
    let mut distance = 1;
    loop {
        let moved = to_move.move_by(Vec2 {
            x: dir.x * distance,
            y: dir.y * distance,
        });
        if moved.does_overlap(move_to) {
            distance -= 1;
            break;
        }
        distance += 1;
        if (dir == UP || dir == DOWN) && distance >= image.len() as i32 {
            return Err("never touched");
        }
        if (dir == LEFT || dir == RIGHT) && distance >= image[0].len() as i32 {
            return Err("never touched");
        }
    }
    let mut new_image = image.clone();
    let moved = to_move.move_by(Vec2 {
        x: dir.x * distance,
        y: dir.y * distance,
    });
    new_image = remove_shape(&new_image, to_move);
    new_image = draw_shape(&new_image, &moved);
    Ok(new_image)
}

// Moves the first shape in a cardinal direction until it touches the second shape.
pub fn move_shape_to_shape_in_image(image: &Image, to_move: &Shape, move_to: &Shape) -> Res<Image> {
    // Find the moving direction.
    let to_move_box = to_move.bounding_box();
    let move_to_box = move_to.bounding_box();
    if to_move_box.right < move_to_box.left {
        return move_shape_to_shape_in_direction(image, to_move, move_to, RIGHT);
    }
    if to_move_box.left > move_to_box.right {
        return move_shape_to_shape_in_direction(image, to_move, move_to, LEFT);
    }
    if to_move_box.bottom < move_to_box.top {
        return move_shape_to_shape_in_direction(image, to_move, move_to, DOWN);
    }
    return move_shape_to_shape_in_direction(image, to_move, move_to, UP);
}
pub fn smallest(shapes: &[Shape]) -> &Shape {
    shapes
        .iter()
        .min_by_key(|shape| shape.cells.len())
        .expect("Should have been a shape")
}

pub fn sort_shapes_by_size(shapes: &mut Vec<Shape>) {
    shapes.sort_by_key(|shape| shape.cells.len());
}

pub fn remap_colors_in_image(image: &mut Image, mapping: &[i32]) {
    for row in image {
        for cell in row {
            let c = mapping[*cell as usize];
            if c != -1 {
                *cell = c;
            }
        }
    }
}

/// Returns the total number of non-zero pixels in the boxes of the given radius
/// around the dots.
pub fn measure_boxes_with_radius(image: &Image, dots: &Shape, radius: i32) -> usize {
    let mut count = 0;
    for Pixel { x, y, color: _ } in &dots.cells {
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
                    continue;
                }
                if image[ny as usize][nx as usize] != 0 {
                    count += 1;
                }
            }
        }
    }
    count
}

pub fn get_pattern_around(image: &Image, dot: &Vec2, radius: i32) -> Shape {
    let mut cells = vec![];
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            let nx = dot.x + dx;
            let ny = dot.y + dy;
            if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
                continue;
            }
            cells.push(Pixel {
                x: dx,
                y: dy,
                color: image[ny as usize][nx as usize],
            });
        }
    }
    Shape { cells }
}

pub fn find_pattern_around(images: &[Image], dots: &[&Shape]) -> Shape {
    let mut radius = 0;
    let mut last_measure = 0;
    loop {
        let mut measure = 0;
        for i in 0..images.len() {
            measure += measure_boxes_with_radius(&images[i], &dots[i], radius);
        }
        if measure == last_measure {
            break;
        }
        last_measure = measure;
        radius += 1;
    }
    // TODO: Instead of just looking at the measure, we should look at the pattern.
    get_pattern_around(&images[0], &dots[0].cells[0].pos(), radius)
}

pub fn draw_shape_at(image: &mut Image, dot: &Vec2, shape: &Shape) {
    for Pixel { x, y, color } in &shape.cells {
        let nx = dot.x + x;
        let ny = dot.y + y;
        if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
            continue;
        }
        image[ny as usize][nx as usize] = *color;
    }
}

pub fn get_used_colors(images: &[Image]) -> Vec<i32> {
    let mut is_used = vec![false; COLORS.len()];
    for image in images {
        for row in image {
            for cell in row {
                is_used[*cell as usize] = true;
            }
        }
    }
    let mut used_colors = vec![];
    for (i, &used) in is_used.iter().enumerate() {
        if used && i != 0 {
            used_colors.push(i as i32);
        }
    }
    used_colors
}

pub fn resize_image(image: &Image, ratio: usize) -> Image {
    let height = image.len() * ratio;
    let width = image[0].len() * ratio;
    let mut new_image = vec![vec![0; width]; height];
    for y in 0..image.len() {
        for x in 0..image[0].len() {
            let color = image[y][x];
            for dy in 0..ratio {
                for dx in 0..ratio {
                    new_image[y * ratio + dy][x * ratio + dx] = color;
                }
            }
        }
    }
    new_image
}

pub fn compare_images(a: &Image, b: &Image) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (row_a, row_b) in a.iter().zip(b) {
        if row_a.len() != row_b.len() {
            return false;
        }
        for (cell_a, cell_b) in row_a.iter().zip(row_b) {
            if cell_a != cell_b {
                return false;
            }
        }
    }
    true
}
