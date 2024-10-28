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
#[derive(Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub struct Pixel {
    pub x: i32,
    pub y: i32,
    pub color: Color,
}

#[derive(Clone, Default)]
pub struct Shape {
    pub cells: Vec<Pixel>,            // Always sorted.
    pub has_relative_colors: bool,    // Color numbers are indexes into state.colors.
    pub has_relative_positions: bool, // x/y are relative to the top-left corner of the shape.
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub struct Line {
    pub pos: i32,
    pub color: i32,
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
    // Put non-empty colorsets into Rc.
    colorsets
        .into_iter()
        .filter(|colorset| !colorset.cells.is_empty())
        .map(|colorset| Rc::new(colorset))
        .collect()
}

pub fn find_horizontal_lines_in_image(image: &Image) -> Lines {
    let mut lines = vec![];
    let mut last_line = -10;
    'outer: for y in 0..image.len() {
        let color = image[y][0];
        if color == 0 {
            continue;
        }
        for x in 0..image[0].len() {
            if image[y][x] != color {
                continue 'outer;
            }
        }
        if y as i32 == last_line + 1 {
            lines.pop();
        } else {
            lines.push(Line {
                pos: y as i32,
                color,
            });
        }
        last_line = y as i32;
    }
    lines
}
pub fn find_vertical_lines_in_image(image: &Image) -> Lines {
    let mut lines = vec![];
    let mut last_line = -10;
    'outer: for x in 0..image[0].len() {
        let color = image[0][x];
        if color == 0 {
            continue;
        }
        for y in 0..image.len() {
            if image[y][x] != color {
                continue 'outer;
            }
        }
        if x as i32 == last_line + 1 {
            lines.pop();
        } else {
            lines.push(Line {
                pos: x as i32,
                color,
            });
        }
        last_line = x as i32;
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
        cells.sort();
        Shape {
            cells,
            ..Default::default()
        }
    }
    #[must_use]
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
        Shape { cells, ..*self }
    }
    pub fn move_by_mut(&mut self, vector: Vec2) {
        for Pixel { x, y, color: _ } in &mut self.cells {
            *x += vector.x;
            *y += vector.y;
        }
    }
    pub fn restore_from(&mut self, other: &Shape) {
        for (a, b) in self.cells.iter_mut().zip(&other.cells) {
            a.x = b.x;
            a.y = b.y;
            a.color = b.color;
        }
    }
    /// Returns true if the shape matches the image at the given position.
    /// Returns false if the shape is entirely out of bounds.
    pub fn matches_image_when_moved_by(&self, image: &Image, vector: Vec2) -> bool {
        let mut something_matched = false;
        for Pixel { x, y, color } in &self.cells {
            let nx = x + vector.x;
            let ny = y + vector.y;
            if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
                continue;
            }
            if image[ny as usize][nx as usize] != *color {
                return false;
            }
            something_matched = true;
        }
        something_matched
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
    pub fn tile(&self, x_step: i32, width: i32, y_step: i32, height: i32) -> Shape {
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
        Shape {
            cells: new_cells,
            ..*self
        }
    }

    #[must_use]
    pub fn crop(&self, left: i32, top: i32, right: i32, bottom: i32) -> Shape {
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
        Shape {
            cells: new_cells,
            ..*self
        }
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
        Shape {
            cells,
            ..Default::default()
        }
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
        let box_ = self.bounding_box();
        println!("top left: {}, {}", box_.left, box_.top);
        for y in box_.top..=box_.bottom {
            for x in box_.left..=box_.right {
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
        Shape {
            cells,
            has_relative_positions: true,
            ..*self
        }
    }

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
        let box_ = self.bounding_box();
        let mut image = vec![vec![0; box_.width() as usize]; box_.height() as usize];
        for Pixel { x, y, color } in &self.cells {
            image[(y - box_.top) as usize][(x - box_.left) as usize] = *color;
        }
        image
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
    let mut reverse_colors = vec![-1; colors.len()];
    for (i, &color) in colors.iter().enumerate() {
        reverse_colors[color as usize] = i as i32;
    }
    reverse_colors
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

pub fn remove_shape(image: &mut Image, shape: &Shape) {
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
    remove_shape(&mut new_image, to_move);
    draw_shape(&mut new_image, &moved);
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

/// Returns the total number of non-zero pixels in the boxes of the given radius
/// around the dots.
pub fn measure_boxes_with_radius(image: &Image, dots: &Shape, radius: i32) -> usize {
    let mut count = 0;
    for Pixel { x, y, color: _ } in &dots.cells {
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let nx = x + dx;
                let ny = y + dy;
                if lookup_in_image(image, nx, ny).unwrap_or(0) != 0 {
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
            let color = lookup_in_image(image, nx, ny).unwrap_or(0);
            if color != 0 {
                cells.push(Pixel {
                    x: dx,
                    y: dy,
                    color,
                });
            }
        }
    }
    Shape {
        cells,
        ..Default::default()
    }
}

pub fn find_pattern_around(images: &[Rc<Image>], dots: &[&Rc<Shape>]) -> Shape {
    // TODO: This is slow. Limit radius? Build pattern incrementally?
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

pub fn get_used_colors(images: &[Rc<Image>]) -> Vec<i32> {
    let mut is_used = vec![false; COLORS.len()];
    for image in images {
        for row in image.iter() {
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
            lines.horizontal[y - 1].pos + 1
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
                lines.vertical[x - 1].pos + 1
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

/// Rotates the image 90 degrees clockwise.
pub fn rotate_image_cw(image: &Image) -> Image {
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
pub fn rotate_image_ccw(image: &Image) -> Image {
    let (width, height) = width_and_height(image);
    let mut new_image = vec![vec![0; height as usize]; width as usize];
    for y in 0..height {
        for x in 0..width {
            new_image[(width - x - 1) as usize][y as usize] = image[y as usize][x as usize];
        }
    }
    new_image
}
