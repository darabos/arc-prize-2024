use colored::Color;

pub const COLORS: [Color; 12] = [
    Color::BrightWhite,
    Color::Black,
    Color::Blue,
    Color::Red,
    Color::Green,
    Color::Yellow,
    Color::TrueColor {
        r: 128,
        g: 0,
        b: 128,
    },
    Color::TrueColor {
        r: 255,
        g: 165,
        b: 0,
    },
    Color::TrueColor {
        r: 165,
        g: 42,
        b: 42,
    },
    Color::Magenta,
    Color::White,
    Color::Cyan,
];

pub type Image = Vec<Vec<i32>>;
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

#[derive(Clone, Default)]
pub struct Shape {
    pub color: i32,
    pub cells: Vec<Vec2>,
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
            let mut cells = vec![Vec2 {
                x: x as i32,
                y: y as i32,
            }];
            visited[y][x] = true;
            let mut i = 0;
            while i < cells.len() {
                let Vec2 { x, y } = cells[i];
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
                    cells.push(Vec2 { x: nx, y: ny });
                }
                i += 1;
            }
            shapes.push(Shape { color, cells });
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
            colorsets[color as usize].cells.push(Vec2 {
                x: x as i32,
                y: y as i32,
            });
        }
    }
    // Set color attribute.
    for (color, colorset) in colorsets.iter_mut().enumerate() {
        colorset.color = color as i32;
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
        if shape.color == color {
            return Some(shape);
        }
    }
    None
}

pub struct Rect {
    pub top: i32,
    pub left: i32,
    pub bottom: i32,
    pub right: i32,
}

pub fn bounding_box(shape: &Shape) -> Rect {
    let mut top = std::i32::MAX;
    let mut left = std::i32::MAX;
    let mut bottom = std::i32::MIN;
    let mut right = std::i32::MIN;
    for Vec2 { x, y } in &shape.cells {
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

pub fn do_shapes_overlap(a: &Shape, b: &Shape) -> bool {
    // Quick check by bounding box.
    let a_box = bounding_box(a);
    let b_box = bounding_box(b);
    if a_box.right < b_box.left || a_box.left > b_box.right {
        return false;
    }
    if a_box.bottom < b_box.top || a_box.top > b_box.bottom {
        return false;
    }
    // Slow check by pixel.
    for Vec2 { x, y } in &a.cells {
        if b.cells.contains(&Vec2 { x: *x, y: *y }) {
            return true;
        }
    }
    false
}

pub fn move_shape(shape: &Shape, vector: Vec2) -> Shape {
    let cells = shape
        .cells
        .iter()
        .map(|Vec2 { x, y }| Vec2 {
            x: *x + vector.x,
            y: *y + vector.y,
        })
        .collect();
    Shape {
        color: shape.color,
        cells,
    }
}

pub fn paint_shape(image: &Image, shape: &Shape, color: i32) -> Image {
    let mut new_image = image.clone();
    for Vec2 { x, y } in &shape.cells {
        if *x < 0 || *y < 0 || *x >= image[0].len() as i32 || *y >= image.len() as i32 {
            continue;
        }
        new_image[*y as usize][*x as usize] = color;
    }
    new_image
}

pub fn remove_shape(image: &Image, shape: &Shape) -> Image {
    paint_shape(image, shape, 0)
}
pub fn draw_shape(image: &Image, shape: &Shape) -> Image {
    paint_shape(image, shape, shape.color)
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
        let moved = move_shape(
            to_move,
            Vec2 {
                x: dir.x * distance,
                y: dir.y * distance,
            },
        );
        if do_shapes_overlap(&moved, move_to) {
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
    let moved = move_shape(
        to_move,
        Vec2 {
            x: dir.x * distance,
            y: dir.y * distance,
        },
    );
    new_image = remove_shape(&new_image, to_move);
    new_image = draw_shape(&new_image, &moved);
    Ok(new_image)
}

// Moves the first shape in a cardinal direction until it touches the second shape.
pub fn move_shape_to_shape_in_image(image: &Image, to_move: &Shape, move_to: &Shape) -> Res<Image> {
    // Find the moving direction.
    let to_move_box = bounding_box(to_move);
    let move_to_box = bounding_box(move_to);
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

pub fn smallest(shapes: Vec<Shape>) -> Shape {
    shapes
        .iter()
        .min_by_key(|shape| shape.cells.len())
        .expect("Should have been a shape")
        .clone()
}

pub fn sort_shapes_by_size(shapes: Vec<Shape>) -> Vec<Shape> {
    let mut shapes = shapes;
    shapes.sort_by_key(|shape| shape.cells.len());
    shapes
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
    for Vec2 { x, y } in &dots.cells {
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

/// Pixels with their relative coordinates and color.
type Pattern = Vec<(Vec2, i32)>;

pub fn get_pattern_around(image: &Image, dot: &Vec2, radius: i32) -> Pattern {
    let mut pattern = vec![];
    for dx in -radius..=radius {
        for dy in -radius..=radius {
            let nx = dot.x + dx;
            let ny = dot.y + dy;
            if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
                continue;
            }
            pattern.push((Vec2 { x: dx, y: dy }, image[ny as usize][nx as usize]));
        }
    }
    pattern
}

pub fn find_pattern_around(images: &[Image], dots: &[Shape]) -> Pattern {
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
    get_pattern_around(&images[0], &dots[0].cells[0], radius)
}

pub fn draw_pattern_at(image: &mut Image, dot: &Vec2, pattern: &Pattern) {
    for (Vec2 { x, y }, color) in pattern {
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
