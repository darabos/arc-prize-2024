pub use crate::image::{Image, SubImageSpec};
pub use crate::shape::{Line, LineSet, Lines, Pixel, Rect, Shape};
use colored;
use colored::Colorize;
use std::rc::Rc;

pub type Color = i32;

pub const COLORS: [colored::Color; 12] = [
    colored::Color::Black,
    colored::Color::BrightWhite,
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

#[derive(Clone, Default)]
pub struct Task {
    pub id: String,
    pub train: Vec<Example>,
    pub test: Vec<Example>,
}

#[derive(Clone)]
pub struct Example {
    pub input: Image,
    pub output: Image,
}

pub type Res<T> = Result<T, &'static str>;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct Vec2 {
    pub x: i32,
    pub y: i32,
}
impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0, y: 0 };
    pub const UP: Vec2 = Vec2 { x: 0, y: -1 };
    pub const DOWN: Vec2 = Vec2 { x: 0, y: 1 };
    pub const LEFT: Vec2 = Vec2 { x: -1, y: 0 };
    pub const RIGHT: Vec2 = Vec2 { x: 1, y: 0 };
    pub const UP_LEFT: Vec2 = Vec2 { x: -1, y: -1 };
    pub const UP_RIGHT: Vec2 = Vec2 { x: 1, y: -1 };
    pub const DOWN_LEFT: Vec2 = Vec2 { x: -1, y: 1 };
    pub const DOWN_RIGHT: Vec2 = Vec2 { x: 1, y: 1 };

    pub const DIRECTIONS4: [Vec2; 4] = [Vec2::UP, Vec2::DOWN, Vec2::LEFT, Vec2::RIGHT];
    pub const DIRECTIONS8: [Vec2; 8] = [
        Vec2::UP,
        Vec2::DOWN,
        Vec2::LEFT,
        Vec2::RIGHT,
        Vec2::UP_LEFT,
        Vec2::UP_RIGHT,
        Vec2::DOWN_LEFT,
        Vec2::DOWN_RIGHT,
    ];
}
pub fn write_color<W: std::fmt::Write>(f: &mut W, color: i32) {
    if color < 0 {
        write!(f, " ").unwrap();
    } else if color == 0 {
        write!(f, "{}", "·".color(colored::Color::Black)).unwrap();
    } else {
        write!(f, "{}", "█".color(COLORS[color as usize])).unwrap();
    }
}

pub fn print_colors(colors: &[i32]) {
    let mut buffer = String::new();
    for &color in colors {
        write_color(&mut buffer, color);
    }
    println!("{}", buffer);
}

#[allow(dead_code)]
pub fn print_example(example: &Example) {
    println!("Input:");
    example.input.print();
    if !example.output.is_empty() {
        println!("Output:");
        example.output.print();
    }
}

#[allow(dead_code)]
pub fn print_task(task: &Task) {
    println!("Train of {}:", task.id);
    for example in &task.train {
        print_example(example);
    }
    println!("Test of {}:", task.id);
    for example in &task.test {
        print_example(example);
    }
}

/// Each shape is a single color. Includes color 0.
#[must_use]
pub fn find_shapes_in_image(image: &Image, directions: &[Vec2]) -> Vec<Rc<Shape>> {
    let mut shapes = vec![];
    let mut visited = vec![vec![false; image.width]; image.height];
    for y in 0..image.height {
        for x in 0..image.width {
            if visited[y][x] {
                continue;
            }
            let color = image[(x, y)];
            let mut cells = vec![Pixel {
                x: x as i32,
                y: y as i32,
                color,
            }];
            visited[y][x] = true;
            let mut i = 0;
            while i < cells.len() {
                let Pixel { x, y, color: _ } = cells[i];
                for dir in directions {
                    let nx = x + dir.x;
                    let ny = y + dir.y;
                    if let Ok(nc) = image.get(nx, ny) {
                        if nc != color {
                            continue;
                        }
                        if visited[ny as usize][nx as usize] {
                            continue;
                        }
                        visited[ny as usize][nx as usize] = true;
                        cells.push(Pixel {
                            x: nx,
                            y: ny,
                            color,
                        });
                    }
                }
                i += 1;
            }
            shapes.push(Rc::new(Shape::new(cells)));
        }
    }
    shapes
}

/// Shapes can include different colors. Color 0 is the separator.
#[must_use]
pub fn find_multicolor_shapes_in_image(image: &Image, directions: &[Vec2]) -> Vec<Rc<Shape>> {
    let mut shapes = vec![];
    let mut visited = vec![vec![false; image.width]; image.height];
    for y in 0..image.height {
        for x in 0..image.width {
            if visited[y][x] {
                continue;
            }
            let color = image[(x, y)];
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
                for dir in directions {
                    let nx = x + dir.x;
                    let ny = y + dir.y;
                    if let Ok(nc) = image.get(nx, ny) {
                        if nc == 0 {
                            continue;
                        }
                        if visited[ny as usize][nx as usize] {
                            continue;
                        }
                        visited[ny as usize][nx as usize] = true;
                        cells.push(Pixel {
                            x: nx,
                            y: ny,
                            color: nc,
                        });
                    }
                }
                i += 1;
            }
            shapes.push(Rc::new(Shape::new(cells)));
        }
    }
    shapes
}

#[must_use]
pub fn discard_background_shapes_touching_border(
    image: &Image,
    shapes: Vec<Rc<Shape>>,
) -> Vec<Rc<Shape>> {
    shapes
        .into_iter()
        .filter(|shape| shape.color() != 0 || !shape.is_touching_border(&image))
        .collect()
}

/// Finds "colorsets" in the image. A colorset is a set of all pixels with the same color.
pub fn find_colorsets_in_image(image: &Image) -> Vec<Rc<Shape>> {
    // Create blank colorset for each color.
    let mut colorsets = vec![vec![]; COLORS.len()];
    for y in 0..image.height {
        for x in 0..image.width {
            let color = image[(x, y)];
            if color == 0 {
                continue;
            }
            colorsets[color as usize].push(Pixel {
                x: x as i32,
                y: y as i32,
                color,
            });
        }
    }
    // Put non-empty colorsets into Rc.
    colorsets
        .into_iter()
        .filter(|colorset| !colorset.is_empty())
        .map(|colorset| Shape::new(colorset).into())
        .collect()
}

pub fn find_horizontal_lines_in_image(image: &Image) -> Lines {
    let mut lines: Lines = vec![];
    'outer: for y in 0..image.height {
        let color = image[(0, y)];
        for x in 0..image.width {
            if image[(x, y)] != color {
                continue 'outer;
            }
        }
        match lines.last_mut() {
            Some(last_line)
                if last_line.pos == y as i32 - last_line.width as i32
                    && last_line.color == color =>
            {
                last_line.width += 1
            }
            _ => lines.push(Line {
                pos: y as i32,
                width: 1,
                color,
            }),
        }
    }
    lines
}
pub fn find_vertical_lines_in_image(image: &Image) -> Lines {
    let mut lines: Lines = vec![];
    'outer: for x in 0..image.width {
        let color = image[(x, 0)];
        for y in 0..image.height {
            if image[(x, y)] != color {
                continue 'outer;
            }
        }
        match lines.last_mut() {
            Some(last_line)
                if last_line.pos == x as i32 - last_line.width as i32
                    && last_line.color == color =>
            {
                last_line.width += 1
            }
            _ => lines.push(Line {
                pos: x as i32,
                width: 1,
                color,
            }),
        }
    }
    lines
}
pub fn find_lines_in_image(image: &Image) -> LineSet {
    LineSet {
        horizontal: find_horizontal_lines_in_image(image),
        vertical: find_vertical_lines_in_image(image),
    }
}

pub fn shape_by_color(shapes: &[Rc<Shape>], color: i32) -> Option<Rc<Shape>> {
    for shape in shapes {
        if shape.color() == color {
            return Some(shape.clone());
        }
    }
    None
}

impl std::ops::Sub for Pixel {
    type Output = Vec2;
    fn sub(self, other: Pixel) -> Vec2 {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl std::ops::Mul<Vec2> for i32 {
    type Output = Vec2;
    fn mul(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self * other.x,
            y: self * other.y,
        }
    }
}
impl std::ops::Add<Vec2> for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
impl std::ops::Sub<Vec2> for Vec2 {
    type Output = Vec2;
    fn sub(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl std::ops::Add<Vec2> for Pixel {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

pub fn reverse_colors(colors: &[i32]) -> Vec<i32> {
    let mut reverse_colors = vec![-1; COLORS.len()];
    for (i, &color) in colors.iter().enumerate() {
        reverse_colors[color as usize] = i as i32;
    }
    reverse_colors
}

pub fn map_colors_in_image(image: &Image, colors_before: &[i32], colors_after: &[i32]) -> Image {
    let reversed_before = reverse_colors(colors_before);
    let mut new_image = image.clone();
    new_image.update(|_x, _y, color| colors_after[reversed_before[color as usize] as usize]);
    new_image
}

pub fn a_matches_b_where_a_is_not_transparent(a: &Image, b: &Image) -> bool {
    if a.width != b.width || a.height != b.height {
        return false;
    }
    for y in 0..a.height {
        for x in 0..a.width {
            if a[(x, y)] != 0 && a[(x, y)] != b[(x, y)] {
                return false;
            }
        }
    }
    true
}

// Moves the first shape pixel by pixel. (Not using bounding boxes.)
#[must_use]
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
        if (dir == Vec2::UP || dir == Vec2::DOWN) && distance >= image.height as i32 {
            return Err("never touched");
        }
        if (dir == Vec2::LEFT || dir == Vec2::RIGHT) && distance >= image.width as i32 {
            return Err("never touched");
        }
    }
    let mut new_image = image.clone();
    let moved = to_move.move_by(Vec2 {
        x: dir.x * distance,
        y: dir.y * distance,
    });
    new_image.erase_shape(to_move);
    new_image.draw_shape(&moved);
    Ok(new_image)
}

// Moves the first shape in a cardinal direction until it touches the second shape.
#[must_use]
pub fn move_shape_to_shape_in_image(image: &Image, to_move: &Shape, move_to: &Shape) -> Res<Image> {
    // Find the moving direction.
    if to_move.bb.right < move_to.bb.left {
        return move_shape_to_shape_in_direction(image, to_move, move_to, Vec2::RIGHT);
    }
    if to_move.bb.left > move_to.bb.right {
        return move_shape_to_shape_in_direction(image, to_move, move_to, Vec2::LEFT);
    }
    if to_move.bb.bottom < move_to.bb.top {
        return move_shape_to_shape_in_direction(image, to_move, move_to, Vec2::DOWN);
    }
    return move_shape_to_shape_in_direction(image, to_move, move_to, Vec2::UP);
}

#[derive(Debug, Clone)]
pub struct ShapePlacement {
    pub pos: Vec2,
    pub match_count: usize,
}
/// Finds the best placement with just translation.
#[must_use]
pub fn place_shape(image: &Image, shape: &Shape) -> Res<ShapePlacement> {
    let mut best_placement = ShapePlacement {
        pos: Vec2::ZERO,
        match_count: 0,
    };
    for y in (-shape.bb.height() + 1)..image.height as i32 {
        for x in (-shape.bb.width() + 1)..image.width as i32 {
            let pos = Vec2 { x, y };
            let mut match_count = 0;
            for Pixel { x, y, color } in shape.cells() {
                let ix = pos.x + x - shape.bb.left;
                let iy = pos.y + y - shape.bb.top;
                if let Ok(ic) = image.get(ix, iy) {
                    if ic == color {
                        match_count += 1;
                    }
                }
            }
            if match_count > best_placement.match_count {
                best_placement = ShapePlacement { pos, match_count };
            }
        }
    }
    if best_placement.match_count == 0 {
        return Err("no match in place_shape");
    }
    Ok(best_placement)
}

pub fn smallest(shapes: &[Shape]) -> &Shape {
    shapes
        .iter()
        .min_by_key(|shape| shape.pixels.len())
        .expect("Should have been a shape")
}

pub fn get_pattern_with_radius(
    images: &[Rc<Image>],
    dots: &[&Rc<Shape>],
    radius: i32,
) -> Res<Shape> {
    get_pattern_in_rect(images, dots, -radius, radius, -radius, radius)
}

pub fn get_pattern_in_rect(
    images: &[Rc<Image>],
    dots: &[&Rc<Shape>],
    min_dx: i32,
    max_dx: i32,
    min_dy: i32,
    max_dy: i32,
) -> Res<Shape> {
    let mut cells = vec![];
    for dx in min_dx..=max_dx {
        for dy in min_dy..=max_dy {
            let mut agreement = -1;
            'images: for i in 0..images.len() {
                let image = &images[i];
                for dot in dots[i].cells() {
                    let nx = dot.x + dx;
                    let ny = dot.y + dy;
                    // Ignore the dots themselves.
                    if dots[i]
                        .cells()
                        .any(|Pixel { x, y, color: _ }| x == nx && y == ny)
                    {
                        continue;
                    }
                    if let Ok(color) = image.get(nx, ny) {
                        if agreement == -1 {
                            agreement = color;
                        } else if agreement != color {
                            agreement = -1;
                            break 'images;
                        }
                    }
                }
            }
            if agreement > 0 {
                cells.push(Pixel {
                    x: dx,
                    y: dy,
                    color: agreement,
                });
            }
        }
    }
    Shape::if_not_empty(cells)
}

pub fn find_pattern_in_square(images: &[Rc<Image>], dots: &[&Rc<Shape>]) -> Res<Shape> {
    let mut last_pattern: Option<Shape> = None;
    for radius in 1..images[0].height as i32 {
        let p = get_pattern_with_radius(&images, &dots, radius)?;
        if let Some(last_pattern) = last_pattern {
            // No improvement. We're done.
            if p.pixels.len() <= last_pattern.pixels.len() {
                return Ok(last_pattern);
            }
        }
        last_pattern = Some(p);
    }
    last_pattern.ok_or("image too small")
}

pub fn find_pattern_horizontally(images: &[Rc<Image>], dots: &[&Rc<Shape>]) -> Res<Shape> {
    let mut last_pattern: Option<Shape> = None;
    for radius in 1..images[0].width as i32 {
        let p = get_pattern_in_rect(&images, &dots, -radius, radius, 0, 0)?;
        if let Some(last_pattern) = last_pattern {
            // No improvement. We're done.
            if p.pixels.len() <= last_pattern.pixels.len() {
                return Ok(last_pattern);
            }
        }
        last_pattern = Some(p);
    }
    last_pattern.ok_or("image too small")
}

pub fn find_pattern_vertically(images: &[Rc<Image>], dots: &[&Rc<Shape>]) -> Res<Shape> {
    let mut last_pattern: Option<Shape> = None;
    for radius in 1..images[0].height as i32 {
        let p = get_pattern_in_rect(&images, &dots, 0, 0, -radius, radius)?;
        if let Some(last_pattern) = last_pattern {
            // No improvement. We're done.
            if p.pixels.len() <= last_pattern.pixels.len() {
                return Ok(last_pattern);
            }
        }
        last_pattern = Some(p);
    }
    last_pattern.ok_or("image too small")
}

pub fn draw_shape_with_relative_colors_at(
    image: &mut Image,
    shape: &Shape,
    colors: &[i32],
    pos: &Vec2,
) {
    for Pixel { x, y, color } in shape.cells() {
        let nx = pos.x + x;
        let ny = pos.y + y;
        if nx < 0 || ny < 0 || nx >= image.width as i32 || ny >= image.height as i32 {
            continue;
        }
        image[(nx as usize, ny as usize)] = colors[color as usize];
    }
}

pub fn count_colors_in_image(image: &Image, counts: &mut Vec<usize>) {
    for c in image.colors_iter() {
        counts[c as usize] += 1;
    }
}

pub fn get_used_colors(images: &[Rc<Image>]) -> Vec<i32> {
    let mut counts = vec![0; COLORS.len()];
    for image in images {
        count_colors_in_image(image, &mut counts);
    }
    let mut used_colors = vec![];
    for (i, &count) in counts.iter().enumerate() {
        if count > 0 && i != 0 {
            used_colors.push(i as i32);
        }
    }
    used_colors
}

pub fn add_remaining_colors(colors: &[i32]) -> Vec<i32> {
    let mut is_used = vec![false; COLORS.len()];
    for &color in colors {
        is_used[color as usize] = true;
    }
    let mut all_colors = Vec::with_capacity(COLORS.len());
    all_colors.extend_from_slice(colors);
    for color in 0..COLORS.len() {
        if !is_used[color] {
            all_colors.push(color as i32);
        }
    }
    all_colors
}

pub fn scale_up_image(image: &Image, ratio: Vec2) -> Image {
    let (rx, ry) = (ratio.x as usize, ratio.y as usize);
    let height = image.height * ry;
    let width = image.width * rx;
    let mut new_image = Image::new(width, height);
    for y in 0..image.height {
        for x in 0..image.width {
            let color = image[(x, y)];
            for dy in 0..ry {
                for dx in 0..rx {
                    new_image[(x * rx + dx, y * ry + dy)] = color;
                }
            }
        }
    }
    new_image
}

pub fn tile_image(image: &Image, repeat_x: usize, repeat_y: usize) -> Image {
    let height = image.height;
    let width = image.width;
    let mut new_image = Image::new(width * repeat_x, height * repeat_y);
    for y in 0..image.height {
        for x in 0..image.width {
            let color = image[(x, y)];
            for dy in 0..repeat_y {
                for dx in 0..repeat_x {
                    new_image[(x + dx * width, y + dy * height)] = color;
                }
            }
        }
    }
    new_image
}

/// Given a grid of lines, returns the images that are separated by the lines.
pub fn grid_cut_image(image: &Image, lines: &LineSet) -> Vec<Image> {
    let mut images = vec![];
    for y in 0..=lines.horizontal.len() {
        let start_y = if y == 0 {
            0
        } else {
            lines.horizontal[y - 1].pos + lines.horizontal[y - 1].width as i32
        };
        let end_y = if y == lines.horizontal.len() {
            image.height as i32
        } else {
            lines.horizontal[y].pos
        };
        if end_y <= start_y {
            continue;
        }
        for x in 0..=lines.vertical.len() {
            let start_x = if x == 0 {
                0
            } else {
                lines.vertical[x - 1].pos + lines.vertical[x - 1].width as i32
            };
            let end_x = if x == lines.vertical.len() {
                image.width as i32
            } else {
                lines.vertical[x].pos
            };
            if end_x <= start_x {
                continue;
            }
            let new_image = image.crop(start_x, start_y, end_x - start_x, end_y - start_y);
            images.push(new_image);
        }
    }
    images
}

/// Returns the number of non-zero pixels in the image.
pub fn count_non_zero_pixels(image: &Image) -> usize {
    image.colors_iter().filter(|&cell| cell != 0).count()
}

pub fn width_and_height(image: &Image) -> (i32, i32) {
    let height = image.height as i32;
    if height == 0 {
        return (0, 0);
    }
    let width = image.width as i32;
    (width, height)
}

#[derive(Debug, Clone, Copy)]
pub enum Rotation {
    CW,
    CCW,
}

impl Rotation {
    pub fn opposite(&self) -> Rotation {
        match self {
            Rotation::CW => Rotation::CCW,
            Rotation::CCW => Rotation::CW,
        }
    }
}

pub fn rotate_image(image: &Image, direction: Rotation) -> Image {
    match direction {
        Rotation::CW => rotate_image_cw(image),
        Rotation::CCW => rotate_image_ccw(image),
    }
}

/// Rotates the image 90 degrees clockwise.
fn rotate_image_cw(image: &Image) -> Image {
    let (width, height) = width_and_height(image);
    let mut new_image = Image::new(height as usize, width as usize);
    for y in 0..height {
        for x in 0..width {
            new_image[((height - y - 1) as usize, x as usize)] = image[(x as usize, y as usize)];
        }
    }
    new_image
}

/// Rotates the image 90 degrees counterclockwise.
fn rotate_image_ccw(image: &Image) -> Image {
    let (width, height) = width_and_height(image);
    let mut new_image = Image::new(height as usize, width as usize);
    for y in 0..height {
        for x in 0..width {
            new_image[(y as usize, (width - x - 1) as usize)] = image[(x as usize, y as usize)];
        }
    }
    new_image
}

#[must_use]
pub fn blend_if_same_color(a: i32, b: i32) -> Res<i32> {
    if a == 0 {
        Ok(b)
    } else if b == 0 {
        Ok(a)
    } else if a == b {
        Ok(a)
    } else {
        Err("different colors")
    }
}

pub fn possible_orders<T: Clone + std::fmt::Debug>(items: &[T]) -> Vec<Vec<T>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut orders = vec![];
    for (i, item) in items.iter().enumerate() {
        let mut rest = items.to_vec();
        rest.remove(i);
        for mut order in possible_orders(&rest) {
            order.push(item.clone());
            orders.push(order);
        }
    }
    orders
}
