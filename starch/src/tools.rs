use colored;
use colored::Colorize;
use std::rc::Rc;

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
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Pixel {
    pub x: i32,
    pub y: i32,
    pub color: Color,
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub cells: Vec<Pixel>,            // Always sorted.
    pub bb: Rect,                     // Bounding box.
    pub has_relative_colors: bool,    // Color numbers are indexes into state.colors.
    pub has_relative_positions: bool, // x/y are relative to the top-left corner of the shape.
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

pub const UP: Vec2 = Vec2 { x: 0, y: -1 };
pub const DOWN: Vec2 = Vec2 { x: 0, y: 1 };
pub const LEFT: Vec2 = Vec2 { x: -1, y: 0 };
pub const RIGHT: Vec2 = Vec2 { x: 1, y: 0 };
pub const UP_LEFT: Vec2 = Vec2 { x: -1, y: -1 };
pub const UP_RIGHT: Vec2 = Vec2 { x: 1, y: -1 };
pub const DOWN_LEFT: Vec2 = Vec2 { x: -1, y: 1 };
pub const DOWN_RIGHT: Vec2 = Vec2 { x: 1, y: 1 };

pub const DIRECTIONS4: [Vec2; 4] = [UP, DOWN, LEFT, RIGHT];
pub const DIRECTIONS8: [Vec2; 8] = [
    UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT,
];

pub fn print_color(color: i32) {
    if color < 0 {
        print!(" ");
    } else if color == 0 {
        print!("{}", "·".color(colored::Color::Black));
    } else {
        print!("{}", "█".color(COLORS[color as usize]));
    }
}

pub fn print_image(image: &Image) {
    for row in image.iter() {
        for cell in row {
            print_color(*cell);
        }
        println!();
    }
}

#[allow(dead_code)]
pub fn print_example(example: &Example) {
    println!("Input:");
    print_image(&example.input);
    if !example.output.is_empty() {
        println!("Output:");
        print_image(&example.output);
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

pub fn resize_canvas(image: &Image, width: usize, height: usize) -> Image {
    let mut new_image = vec![vec![0; width]; height];
    for y in 0..image.len().min(height) {
        for x in 0..image[0].len().min(width) {
            new_image[y][x] = image[y][x];
        }
    }
    new_image
}

/// Each shape is a single color. Includes color 0.
#[must_use]
pub fn find_shapes_in_image(image: &Image, directions: &[Vec2]) -> Vec<Rc<Shape>> {
    let mut shapes = vec![];
    let mut visited = vec![vec![false; image[0].len()]; image.len()];
    for y in 0..image.len() {
        for x in 0..image[0].len() {
            if visited[y][x] {
                continue;
            }
            let color = image[y][x];
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
                    if let Ok(nc) = lookup_in_image(image, nx, ny) {
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
                for dir in directions {
                    let nx = x + dir.x;
                    let ny = y + dir.y;
                    if let Ok(nc) = lookup_in_image(image, nx, ny) {
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
    for y in 0..image.len() {
        for x in 0..image[0].len() {
            let color = image[y][x];
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
    'outer: for y in 0..image.len() {
        let color = image[y][0];
        for x in 0..image[y].len() {
            if image[y][x] != color {
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
    'outer: for x in 0..image[0].len() {
        let color = image[0][x];
        for y in 0..image.len() {
            if image[y][x] != color {
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

impl Pixel {
    pub fn pos(&self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
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
    pub fn new(mut cells: Vec<Pixel>) -> Shape {
        assert!(!cells.is_empty());
        cells.sort();
        let mut top = std::i32::MAX;
        let mut left = std::i32::MAX;
        let mut bottom = std::i32::MIN;
        let mut right = std::i32::MIN;
        for Pixel { x, y, color: _ } in &cells {
            top = top.min(*y);
            left = left.min(*x);
            bottom = bottom.max(*y);
            right = right.max(*x);
        }
        Shape {
            cells,
            bb: Rect {
                top,
                left,
                bottom,
                right,
            },
            has_relative_colors: false,
            has_relative_positions: false,
        }
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
        } in &self.cells
        {
            if *px == x && *py == y {
                return Some(*color);
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
        for Pixel { x, y, color: _ } in &self.cells {
            if other.color_at(*x, *y).is_some() {
                return true;
            }
        }
        false
    }

    #[must_use]
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
        Shape::new(cells)
    }
    pub fn move_by_mut(&mut self, vector: Vec2) {
        for Pixel { x, y, color: _ } in &mut self.cells {
            *x += vector.x;
            *y += vector.y;
        }
        self.bb.top += vector.y;
        self.bb.bottom += vector.y;
        self.bb.left += vector.x;
        self.bb.right += vector.x;
    }
    pub fn restore_from(&mut self, other: &Shape) {
        for (a, b) in self.cells.iter_mut().zip(&other.cells) {
            a.x = b.x;
            a.y = b.y;
            a.color = b.color;
        }
        self.bb = other.bb;
    }
    /// Returns true if the shape matches the image at the given position.
    /// Returns false if the shape is entirely out of bounds.
    pub fn matches_image_when_moved_by(&self, image: &Image, vector: Vec2) -> bool {
        let mut matched_count = 0;
        for Pixel { x, y, color } in &self.cells {
            let nx = x + vector.x;
            let ny = y + vector.y;
            if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
                continue;
            }
            let icolor = image[ny as usize][nx as usize];
            if icolor == 0 {
                continue;
            }
            if icolor != *color {
                return false;
            }
            matched_count += 1;
        }
        matched_count > 0
        // TODO: Why isn't this better?
        // matched_count >= 2 && matched_count >= self.cells.len() / 2
    }

    pub fn recolor(&mut self, color: i32) {
        for cell in &mut self.cells {
            cell.color = color;
        }
    }
    #[must_use]
    pub fn color(&self) -> i32 {
        self.cells[0].color
    }
    #[must_use]
    pub fn tile(&self, x_step: i32, width: i32, y_step: i32, height: i32) -> Res<Shape> {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in &self.cells {
            for &tx in &[x_step, -x_step] {
                let mut cx = *x;
                while cx >= 0 && cx < width {
                    for &ty in &[y_step, -y_step] {
                        let mut cy = *y;
                        while cy >= 0 && cy < height {
                            new_cells.push(Pixel {
                                x: cx,
                                y: cy,
                                color: *color,
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
        for Pixel { x, y, color } in &self.cells {
            if *x >= left && *x <= right && *y >= top && *y <= bottom {
                new_cells.push(Pixel {
                    x: *x - left,
                    y: *y - top,
                    color: *color,
                });
            }
        }
        Shape::if_not_empty(new_cells)
    }

    pub fn draw_where_non_empty(&self, image: &mut Image) {
        for Pixel { x, y, color } in &self.cells {
            if lookup_in_image(image, *x, *y).unwrap_or(0) != 0 {
                image[*y as usize][*x as usize] = *color;
            }
        }
    }

    pub fn discard_color(&mut self, color: i32) {
        self.cells = std::mem::take(&mut self.cells)
            .into_iter()
            .filter(|cell| cell.color != color)
            .collect();
    }

    #[must_use]
    pub fn from_image(image: &Image) -> Shape {
        let mut cells = vec![];
        for x in 0..image[0].len() {
            for y in 0..image.len() {
                let color = image[y][x];
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
        for Pixel { x, y, color: _ } in &self.cells {
            if *x == 0 || *y == 0 || *x == image[0].len() as i32 - 1 || *y == image.len() as i32 - 1
            {
                return true;
            }
        }
        false
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        println!("top left: {}, {}", self.bb.left, self.bb.top);
        for y in self.bb.top..=self.bb.bottom {
            for x in self.bb.left..=self.bb.right {
                print_color(self.color_at(x, y).unwrap_or(-1));
            }
            println!();
        }
    }

    pub fn use_relative_colors(&mut self, reverse_colors: &[i32]) {
        for cell in &mut self.cells {
            cell.color = reverse_colors[cell.color as usize];
        }
        self.has_relative_colors = true;
    }

    #[must_use]
    pub fn covers(&self, other: &Shape) -> bool {
        for Pixel { x, y, color: _ } in &other.cells {
            if self.color_at(*x, *y).is_none() {
                return false;
            }
        }
        true
    }

    #[must_use]
    pub fn to_relative_pos(&self) -> Shape {
        let min_x = self.cells.iter().map(|cell| cell.x).min().unwrap();
        let min_y = self.cells.iter().map(|cell| cell.y).min().unwrap();
        let cells = self
            .cells
            .iter()
            .map(|Pixel { x, y, color }| Pixel {
                x: x - min_x,
                y: y - min_y,
                color: *color,
            })
            .collect();
        let mut shp = Shape::new(cells);
        shp.has_relative_positions = true;
        shp
    }

    /// Requires exact match.
    #[must_use]
    pub fn find_matching_shape_index(&self, shapes: &[Rc<Shape>]) -> Option<usize> {
        for (i, shape) in shapes.iter().enumerate() {
            if self == shape.as_ref() {
                return Some(i);
            }
        }
        None
    }

    #[must_use]
    pub fn as_image(&self) -> Image {
        let mut image = vec![vec![0; self.bb.width() as usize]; self.bb.height() as usize];
        for Pixel { x, y, color } in &self.cells {
            image[(y - self.bb.top) as usize][(x - self.bb.left) as usize] = *color;
        }
        image
    }

    #[must_use]
    pub fn rotate_90_cw(&self) -> Shape {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in &self.cells {
            new_cells.push(Pixel {
                x: -y,
                y: *x,
                color: *color,
            });
        }
        Shape::new(new_cells)
    }

    #[must_use]
    pub fn flip_horizontal(&self) -> Shape {
        let mut new_cells = vec![];
        for Pixel { x, y, color } in &self.cells {
            new_cells.push(Pixel {
                x: -x,
                y: *y,
                color: *color,
            });
        }
        Shape::new(new_cells)
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        if self.has_relative_colors != other.has_relative_colors {
            return false;
        }
        if self.cells.len() != other.cells.len() {
            return false;
        }
        for (a, b) in self.cells.iter().zip(other.cells.iter()) {
            if a.x != b.x || a.y != b.y || a.color != b.color {
                return false;
            }
        }
        true
    }
}

impl Eq for Shape {}

pub fn reverse_colors(colors: &[i32]) -> Vec<i32> {
    let mut reverse_colors = vec![-1; COLORS.len()];
    for (i, &color) in colors.iter().enumerate() {
        reverse_colors[color as usize] = i as i32;
    }
    reverse_colors
}

pub fn map_colors_in_image(image: &Image, colors_before: &[i32], colors_after: &[i32]) -> Image {
    let reversed_before = reverse_colors(colors_before);
    image
        .iter()
        .map(|row| {
            row.iter()
                .map(|&color| colors_after[reversed_before[color as usize] as usize])
                .collect()
        })
        .collect()
}

/// Draws the image in the given color.
pub fn paint_shape(image: &mut Image, shape: &Shape, color: i32) {
    for Pixel { x, y, color: _ } in &shape.cells {
        if *x < 0 || *y < 0 || *x >= image[0].len() as i32 || *y >= image.len() as i32 {
            continue;
        }
        image[*y as usize][*x as usize] = color;
    }
}

pub fn crop_image(image: &Image, left: i32, top: i32, width: i32, height: i32) -> Image {
    let mut new_image = vec![vec![0; width as usize]; height as usize];
    for y in 0..height {
        for x in 0..width {
            let nx = left + x;
            let ny = top + y;
            if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
                continue;
            }
            new_image[y as usize][x as usize] = image[ny as usize][nx as usize];
        }
    }
    new_image
}

pub fn erase_shape(image: &mut Image, shape: &Shape) {
    paint_shape(image, &shape, 0)
}
/// Draws the shape in its original color.
pub fn draw_shape(image: &mut Image, shape: &Shape) {
    for Pixel { x, y, color } in &shape.cells {
        if *x < 0 || *y < 0 || *x >= image[0].len() as i32 || *y >= image.len() as i32 {
            continue;
        }
        image[*y as usize][*x as usize] = *color;
    }
}

pub fn draw_shape_with_colors(image: &mut Image, shape: &Shape, colors: &[i32]) {
    for Pixel { x, y, color } in &shape.cells {
        if *x < 0 || *y < 0 || *x >= image[0].len() as i32 || *y >= image.len() as i32 {
            continue;
        }
        image[*y as usize][*x as usize] = colors[*color as usize];
    }
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
    erase_shape(&mut new_image, to_move);
    draw_shape(&mut new_image, &moved);
    Ok(new_image)
}

// Moves the first shape in a cardinal direction until it touches the second shape.
pub fn move_shape_to_shape_in_image(image: &Image, to_move: &Shape, move_to: &Shape) -> Res<Image> {
    // Find the moving direction.
    if to_move.bb.right < move_to.bb.left {
        return move_shape_to_shape_in_direction(image, to_move, move_to, RIGHT);
    }
    if to_move.bb.left > move_to.bb.right {
        return move_shape_to_shape_in_direction(image, to_move, move_to, LEFT);
    }
    if to_move.bb.bottom < move_to.bb.top {
        return move_shape_to_shape_in_direction(image, to_move, move_to, DOWN);
    }
    return move_shape_to_shape_in_direction(image, to_move, move_to, UP);
}

#[derive(Debug, Clone)]
pub struct ShapePlacement {
    pub pos: Vec2,
    pub match_count: usize,
}
/// Finds the best placement with just translation.
pub fn place_shape(image: &Image, shape: &Shape) -> Res<ShapePlacement> {
    let mut best_placement = ShapePlacement {
        pos: Vec2::ZERO,
        match_count: 0,
    };
    for y in (-shape.bb.height() + 1)..image.len() as i32 {
        for x in (-shape.bb.width() + 1)..image[0].len() as i32 {
            let pos = Vec2 { x, y };
            let mut match_count = 0;
            for Pixel { x, y, color } in &shape.cells {
                let ix = pos.x + x - shape.bb.left;
                let iy = pos.y + y - shape.bb.top;
                if let Ok(ic) = lookup_in_image(image, ix, iy) {
                    if ic == *color {
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
        return Err("no match");
    }
    Ok(best_placement)
}

pub fn smallest(shapes: &[Shape]) -> &Shape {
    shapes
        .iter()
        .min_by_key(|shape| shape.cells.len())
        .expect("Should have been a shape")
}

pub fn lookup_in_image(image: &Image, x: i32, y: i32) -> Res<i32> {
    if x < 0 || y < 0 || x >= image[0].len() as i32 || y >= image.len() as i32 {
        return Err("out of bounds");
    }
    Ok(image[y as usize][x as usize])
}
pub fn set_in_image(image: &mut Image, x: i32, y: i32, color: i32) {
    if x < 0 || y < 0 || x >= image[0].len() as i32 || y >= image.len() as i32 {
        return;
    }
    image[y as usize][x as usize] = color;
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
                for dot in &dots[i].cells {
                    let nx = dot.x + dx;
                    let ny = dot.y + dy;
                    // Ignore the dots themselves.
                    if dots[i]
                        .cells
                        .iter()
                        .any(|Pixel { x, y, color: _ }| *x == nx && *y == ny)
                    {
                        continue;
                    }
                    if let Ok(color) = lookup_in_image(image, nx, ny) {
                        if agreement == -1 {
                            agreement = color;
                        } else if agreement != color {
                            // println!(
                            //     "disagreement in {} {}!={} at {} {} ({} {})",
                            //     i, agreement, color, nx, ny, dx, dy
                            // );
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
    for radius in 1..images[0].len() as i32 {
        let p = get_pattern_with_radius(&images, &dots, radius)?;
        if let Some(last_pattern) = last_pattern {
            // No improvement. We're done.
            if p.cells.len() <= last_pattern.cells.len() {
                return Ok(last_pattern);
            }
        }
        last_pattern = Some(p);
    }
    last_pattern.ok_or("image too small")
}

pub fn find_pattern_horizontally(images: &[Rc<Image>], dots: &[&Rc<Shape>]) -> Res<Shape> {
    let mut last_pattern: Option<Shape> = None;
    for radius in 1..images[0][0].len() as i32 {
        let p = get_pattern_in_rect(&images, &dots, -radius, radius, 0, 0)?;
        if let Some(last_pattern) = last_pattern {
            // No improvement. We're done.
            if p.cells.len() <= last_pattern.cells.len() {
                return Ok(last_pattern);
            }
        }
        last_pattern = Some(p);
    }
    last_pattern.ok_or("image too small")
}

pub fn find_pattern_vertically(images: &[Rc<Image>], dots: &[&Rc<Shape>]) -> Res<Shape> {
    let mut last_pattern: Option<Shape> = None;
    for radius in 1..images[0].len() as i32 {
        let p = get_pattern_in_rect(&images, &dots, 0, 0, -radius, radius)?;
        if let Some(last_pattern) = last_pattern {
            // No improvement. We're done.
            if p.cells.len() <= last_pattern.cells.len() {
                return Ok(last_pattern);
            }
        }
        last_pattern = Some(p);
    }
    last_pattern.ok_or("image too small")
}

pub fn draw_shape_at(image: &mut Image, shape: &Shape, pos: Vec2) {
    for Pixel { x, y, color } in &shape.cells {
        let nx = pos.x + x;
        let ny = pos.y + y;
        if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
            continue;
        }
        image[ny as usize][nx as usize] = *color;
    }
}

pub fn draw_shape_with_relative_colors_at(
    image: &mut Image,
    shape: &Shape,
    colors: &[i32],
    pos: &Vec2,
) {
    for Pixel { x, y, color } in &shape.cells {
        let nx = pos.x + x;
        let ny = pos.y + y;
        if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
            continue;
        }
        image[ny as usize][nx as usize] = colors[*color as usize];
    }
}

pub fn set_used_colors_in_image(image: &Image, is_used: &mut Vec<bool>) {
    for row in image.iter() {
        for cell in row {
            is_used[*cell as usize] = true;
        }
    }
}

pub fn get_used_colors(images: &[Rc<Image>]) -> Vec<i32> {
    let mut is_used = vec![false; COLORS.len()];
    for image in images {
        set_used_colors_in_image(image, &mut is_used);
    }
    let mut used_colors = vec![];
    for (i, &used) in is_used.iter().enumerate() {
        if used && i != 0 {
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
    let height = image.len() * ry;
    let width = image[0].len() * rx;
    let mut new_image = vec![vec![0; width]; height];
    for y in 0..image.len() {
        for x in 0..image[0].len() {
            let color = image[y][x];
            for dy in 0..ry {
                for dx in 0..rx {
                    new_image[y * ry + dy][x * rx + dx] = color;
                }
            }
        }
    }
    new_image
}

pub fn tile_image(image: &Image, repeat_x: usize, repeat_y: usize) -> Image {
    let height = image.len();
    let width = image[0].len();
    let mut new_image = vec![vec![0; width * repeat_x]; height * repeat_y];
    for y in 0..image.len() {
        for x in 0..image[0].len() {
            let color = image[y][x];
            for dy in 0..repeat_y {
                for dx in 0..repeat_x {
                    new_image[y + dy * height][x + dx * width] = color;
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
            image.len() as i32
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
                image[0].len() as i32
            } else {
                lines.vertical[x].pos
            };
            if end_x <= start_x {
                continue;
            }
            let new_image = crop_image(image, start_x, start_y, end_x - start_x, end_y - start_y);
            images.push(new_image);
        }
    }
    images
}

/// Returns the number of non-zero pixels in the image.
pub fn count_non_zero_pixels(image: &Image) -> usize {
    image.iter().flatten().filter(|&&cell| cell != 0).count()
}

pub fn width_and_height(image: &Image) -> (i32, i32) {
    let height = image.len() as i32;
    if height == 0 {
        return (0, 0);
    }
    let width = image[0].len() as i32;
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
    let mut new_image = vec![vec![0; height as usize]; width as usize];
    for y in 0..height {
        for x in 0..width {
            new_image[x as usize][(height - y - 1) as usize] = image[y as usize][x as usize];
        }
    }
    new_image
}

/// Rotates the image 90 degrees counterclockwise.
fn rotate_image_ccw(image: &Image) -> Image {
    let (width, height) = width_and_height(image);
    let mut new_image = vec![vec![0; height as usize]; width as usize];
    for y in 0..height {
        for x in 0..width {
            new_image[(width - x - 1) as usize][y as usize] = image[y as usize][x as usize];
        }
    }
    new_image
}
