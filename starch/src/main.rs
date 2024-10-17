use colored::Colorize;
use serde_json;
use std::fs;

#[derive(Clone, Default)]
pub struct Task {
    pub train: Vec<Example>,
    pub test: Vec<Example>,
}

pub type Image = Vec<Vec<i32>>;
#[derive(Clone)]
pub struct Example {
    pub input: Image,
    pub output: Image,
}

type Res<T> = Result<T, &'static str>;
use std::collections::HashMap as Map;

pub fn parse_image(image: &serde_json::Value) -> Image {
    image
        .as_array()
        .expect("Should have been an array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("Should have been an array")
                .iter()
                .map(|cell| cell.as_i64().expect("Should have been an integer") as i32)
                .collect()
        })
        .collect()
}

pub fn parse_example(example: &serde_json::Value) -> Example {
    let input = parse_image(&example["input"]);
    if example["output"].is_null() {
        return Example {
            input,
            output: vec![],
        };
    }
    let output = parse_image(&example["output"]);
    Example { input, output }
}
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

pub fn print_color(color: i32) {
    if color == 0 {
        print!(" ");
    } else {
        print!("{}", "█".color(COLORS[color as usize]));
    }
}

pub fn print_image(image: &Image) {
    for row in image {
        for cell in row {
            print_color(*cell);
        }
        println!();
    }
}

pub fn print_example(example: &Example) {
    println!("Input:");
    print_image(&example.input);
    if !example.output.is_empty() {
        println!("Output:");
        print_image(&example.output);
    }
}

pub fn parse_task(task: &serde_json::Value) -> Task {
    let train = task["train"].as_array().expect("Should have been an array");
    let train = train.iter().map(|example| parse_example(example)).collect();
    let test = task["test"].as_array().expect("Should have been an array");
    let test = test.iter().map(|example| parse_example(example)).collect();
    Task { train, test }
}

pub fn print_task(task: &Task) {
    println!("Train:");
    for example in &task.train {
        print_example(example);
    }
    println!("Test:");
    for example in &task.test {
        print_example(example);
    }
}

pub fn read_arc_file(file_path: &str) -> Map<String, Task> {
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    let data: serde_json::Value =
        serde_json::from_str(&contents).expect("Should have been able to parse the json");
    let data = data.as_object().expect("Should have been an object");
    let mut tasks = Map::new();
    for (key, task) in data {
        tasks.insert(key.clone(), parse_task(task));
    }
    tasks
}

pub fn read_arc_solutions_file(file_path: &str) -> Map<String, Vec<Image>> {
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    let data: serde_json::Value =
        serde_json::from_str(&contents).expect("Should have been able to parse the json");
    let data = data.as_object().expect("Should have been an object");
    let mut tasks = Map::new();
    for (key, task) in data {
        let images = task
            .as_array()
            .expect("Should have been an array")
            .iter()
            .map(|image| parse_image(image))
            .collect();
        tasks.insert(key.clone(), images);
    }
    tasks
}
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

pub fn sort_these_shapes_by_size(shapes: Vec<Shape>) -> Vec<Shape> {
    let mut shapes = shapes;
    shapes.sort_by_key(|shape| shape.cells.len());
    shapes
}

pub fn find_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    if s.shapes.is_none() {
        s.shapes = Some(vec![vec![]; s.images.len()]);
    }
    s.shapes.as_mut().unwrap()[i] = find_shapes_in_image(&s.images[i]);
    Ok(())
}

pub fn find_colorsets(s: &mut SolverState, i: usize) -> Res<()> {
    if s.colorsets.is_none() {
        s.colorsets = Some(vec![vec![]; s.images.len()]);
    }
    s.colorsets.as_mut().unwrap()[i] = find_colorsets_in_image(&s.images[i]);
    Ok(())
}

pub fn use_colorsets_as_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.colorsets.clone();
    Ok(())
}

pub fn sort_shapes_by_size(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    *shapes = sort_these_shapes_by_size(shapes.clone());
    Ok(())
}

fn remap_colors_in_image(image: &mut Image, mapping: &[i32]) {
    for row in image {
        for cell in row {
            let c = mapping[*cell as usize];
            if c != -1 {
                *cell = c;
            }
        }
    }
}

fn remap_colors(s: &mut SolverState, i: usize, mapping: &[i32]) {
    remap_colors_in_image(&mut s.images[i], mapping);
    if i < s.output_images.len() {
        remap_colors_in_image(&mut s.output_images[i], mapping);
    }
    if let Some(colorsets) = &mut s.colorsets {
        for shape in colorsets[i].iter_mut() {
            shape.color = mapping[shape.color as usize];
        }
    }
    if let Some(shapes) = &mut s.shapes {
        for shape in shapes[i].iter_mut() {
            shape.color = mapping[shape.color as usize];
        }
    }
    if let Some(saved_shapes) = &mut s.saved_shapes {
        for shape in saved_shapes[i].iter_mut() {
            shape.color = mapping[shape.color as usize];
        }
    }
    if let Some(dots) = &mut s.dots {
        for shape in dots[i].iter_mut() {
            shape.color = mapping[shape.color as usize];
        }
    }
    if i == s.images.len() - 1 {
        s.used_colors = get_used_colors(&s.images);
    }
}

/// Renumbers the colors of the image to match the order of the shapes.
/// Modifies the image and the shapes. Returns the mapping.
fn remap_colors_by_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    let mut mapping = vec![-1; COLORS.len()];
    for (i, shape) in shapes.iter_mut().enumerate() {
        mapping[shape.color as usize] = i as i32 + 1;
    }
    remap_colors(s, i, &mapping);
    if s.color_mapping.is_none() {
        s.color_mapping = Some(vec![vec![]; s.images.len()]);
    }
    s.color_mapping.as_mut().unwrap()[i] = mapping;
    Ok(())
}

fn unmap_colors(s: &mut SolverState, i: usize) -> Res<()> {
    let mapping = &s.color_mapping.as_ref().expect("must have color mapping")[i];
    let mut reverse_mapping = vec![0; COLORS.len()];
    for (i, &c) in mapping.iter().enumerate() {
        if c != -1 {
            reverse_mapping[c as usize] = i as i32;
        }
    }
    remap_colors(s, i, &reverse_mapping);
    if i == s.images.len() - 1 {
        s.color_mapping = None;
    }
    Ok(())
}

/// Returns the total number of non-zero pixels in the boxes of the given radius
/// around the dots.
fn measure_boxes_with_radius(image: &Image, dots: &Shape, radius: i32) -> usize {
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

fn get_pattern_around(image: &Image, dot: &Vec2, radius: i32) -> Pattern {
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

fn find_pattern_around(images: &[Image], dots: &[Shape]) -> Pattern {
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

fn draw_pattern_at(image: &mut Image, dot: &Vec2, pattern: &Pattern) {
    for (Vec2 { x, y }, color) in pattern {
        let nx = dot.x + x;
        let ny = dot.y + y;
        if nx < 0 || ny < 0 || nx >= image[0].len() as i32 || ny >= image.len() as i32 {
            continue;
        }
        image[ny as usize][nx as usize] = *color;
    }
}

fn get_firsts<T : Clone>(vec: &Vec<Vec<T>>) -> Res<Vec<T>> {
    let mut firsts = vec![];
    for e in vec {
        if e.is_empty() {
            return Err("empty list");
        }
        firsts.push(e[0].clone());
    }
    Ok(firsts)
}

fn grow_flowers(s: &mut SolverState) -> Res<()> {
    let shapes = &s.shapes.as_ref().ok_or("must have shapes")?;
    let dots = get_firsts(&shapes)?;
    // let input_pattern = find_pattern_around(&s.images[..s.task.train.len()], &dots);
    let output_pattern = find_pattern_around(&s.output_images, &dots);
    // TODO: Instead of growing each dot, we should filter by the input_pattern.
    s.apply(|s: &mut SolverState, i: usize| {
        let shapes = &s.shapes.as_ref().expect("must have shapes");
        let dots = &shapes[i][0];
        for dot in dots.cells.iter() {
            draw_pattern_at(&mut s.images[i], dot, &output_pattern);
        }
        Ok(())
    })
}

fn save_picked_shapes(s: &mut SolverState) -> Res<()> {
    s.saved_shapes = s.picked_shapes.clone();
    Ok(())
}

fn save_whole_image(s: &mut SolverState) -> Res<()> {
    s.saved_images = s.images.clone();
    Ok(())
}

fn get_used_colors(images: &[Image]) -> Vec<i32> {
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

fn resize_image(image: &Image, ratio: usize) -> Image {
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

/// Tracks information while applying operations on all examples at once.
/// Most fields are vectors storing information for each example.
#[derive(Default)]
pub struct SolverState {
    pub task: Task,
    pub images: Vec<Image>,
    pub saved_images: Vec<Image>,
    pub output_images: Vec<Image>,
    pub used_colors: Vec<i32>,
    pub shapes: Option<Vec<Vec<Shape>>>,
    pub picked_shapes: Option<Vec<Vec<Shape>>>,
    pub saved_shapes: Option<Vec<Vec<Shape>>>,
    pub colorsets: Option<Vec<Vec<Shape>>>,
    pub dots: Option<Vec<Vec<Shape>>>,
    pub color_mapping: Option<Vec<Vec<i32>>>,
}

impl SolverState {
    fn new(task: &Task) -> Self {
        let images: Vec<Image> = task
            .train
            .iter()
            .chain(task.test.iter())
            .map(|example| example.input.clone())
            .collect();
        let output_images = task
            .train
            .iter()
            .map(|example| example.output.clone())
            .collect();
        let used_colors = get_used_colors(&images);
        SolverState {
            task: task.clone(),
            images,
            output_images,
            used_colors,
            ..Default::default()
        }
    }

    fn apply<F>(&mut self, f: F) -> Res<()>
    where
        F: Fn(&mut SolverState, usize) -> Res<()>,
    {
        for i in 0..self.images.len() {
            f(self, i)?;
        }
        Ok(())
    }

    fn get_results(&self) -> Vec<Example> {
        self.images
            .iter()
            .zip(self.task.train.iter().chain(self.task.test.iter()))
            .map(|(image, example)| Example {
                input: example.input.clone(),
                output: image.clone(),
            })
            .collect()
    }
}

fn rotate_used_colors(s: &mut SolverState) -> Res<()> {
    if s.used_colors.is_empty() {
        return Err("no used colors");
    }
    let first_color = s.used_colors[0];
    let n = s.used_colors.len();
    for i in 0..n - 1 {
        s.used_colors[i] = s.used_colors[i + 1];
    }
    s.used_colors[n - 1] = first_color;
    Ok(())
}

fn pick_shape_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes.as_ref().expect("must have shapes")[i];
    let shape = shape_by_color(&shapes, s.used_colors[0]).ok_or("should have been a shape")?;
    if s.picked_shapes.is_none() {
        s.picked_shapes = Some(vec![vec![]; s.images.len()]);
    }
    s.picked_shapes.as_mut().unwrap()[i] = vec![shape.clone()];
    Ok(())
}

fn move_picked_shape_to_saved_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let picked_shapes = &s.picked_shapes.as_ref().expect("must have picked shapes")[i];
    let saved_shapes = &s.saved_shapes.as_ref().expect("must have saved shapes")[i];
    s.images[i] = move_shape_to_shape_in_image(
        &s.images[i], &picked_shapes[0], &saved_shapes[0])?;
    Ok(())
}

fn solve_example_7(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_colorsets)?;
    use_colorsets_as_shapes(&mut state)?;
    state.apply(remap_colors_by_shapes)?;
    state.apply(find_shapes)?;
    rotate_used_colors(&mut state)?;
    state.apply(pick_shape_by_color)?;
    save_picked_shapes(&mut state)?;
    rotate_used_colors(&mut state)?;
    // TODO: We shouldn't need to find shapes again!
    state.apply(pick_shape_by_color)?;
    state.apply(move_picked_shape_to_saved_shape)?;
    state.apply(unmap_colors)?;
    Ok(state.get_results())
}

fn solve_example_11(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    state.apply(find_colorsets)?;
    use_colorsets_as_shapes(&mut state)?;
    state.apply(sort_shapes_by_size)?;
    state.apply(remap_colors_by_shapes)?;
    grow_flowers(&mut state)?;
    state.apply(unmap_colors)?;
    Ok(state.get_results())
}

fn solve_example_0(task: &Task) -> Res<Vec<Example>> {
    let mut state = SolverState::new(task);
    // Working on this...
    state.apply(find_colorsets)?;
    use_colorsets_as_shapes(&mut state)?;
    state.apply(sort_shapes_by_size)?;
    state.apply(remap_colors_by_shapes)?;
    grow_flowers(&mut state)?;
    state.apply(unmap_colors)?;
    Ok(state.get_results())
}

type Solver = fn(&Task) -> Res<Vec<Example>>;
const SOLVERS: &[Solver] = &[solve_example_0, solve_example_7, solve_example_11];

fn compare_images(a: &Image, b: &Image) -> bool {
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

fn test_on_all_tasks() {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let ref_solutions = read_arc_solutions_file("../arc-agi_training_solutions.json");
    let mut correct = 0;
    for (name, task) in &tasks {
        for solver in SOLVERS {
            let solutions = solver(task);
            if solutions.is_err() {
                continue;
            }
            let solutions = solutions.unwrap()[task.train.len()..].to_vec();
            let ref_images = ref_solutions.get(name).expect("Should have been a solution");
            let mut all_correct = true;
            for i in 0..ref_images.len() {
                let ref_image = &ref_images[i];
                let image = &solutions[i].output;
                if !compare_images(ref_image, image) {
                    all_correct = false;
                    break;
                }
                println!("{}: {}", name, "Correct".green());
            }
            if all_correct {
                correct += 1;
                break;
            }
        }
    }
    println!("Correct: {}/{}", correct, tasks.len());
}

#[allow(dead_code)]
fn test_on_one_task() {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let ref_solutions = read_arc_solutions_file("../arc-agi_training_solutions.json");
    // let name = "05f2a901"; // 7
    let name = "0962bcdd"; // 11
    let task = tasks.get(name).expect("Should have been a task");
    let solutions = solve_example_11(task).expect("No solution found");
    let ref_solution = ref_solutions
        .get(name)
        .expect("Should have been a solution");
    let ref_images: Vec<Image> = task
        .train
        .iter()
        .map(|example| example.output.clone())
        .chain(ref_solution.iter().cloned())
        .collect();
    let mut correct = 0;
    for i in 0..ref_images.len() {
        let ref_image = &ref_images[i];
        let image = &solutions[i].output;
        if compare_images(ref_image, image) {
            correct += 1;
        } else {
            println!("Expected:");
            print_image(ref_image);
            println!("Actual:");
            print_image(image);
            println!();
        }
    }
    println!("Correct: {}/{}", correct, ref_images.len());
}

fn main() {
    test_on_all_tasks();
    // test_on_one_task();
}
