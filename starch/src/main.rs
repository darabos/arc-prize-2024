use colored::Colorize;
use serde_json;
use std::fs;

pub struct Task {
    train: Vec<Example>,
    test: Vec<Example>,
}

pub type Image = Vec<Vec<i32>>;

pub struct Example {
    input: Image,
    output: Image,
}

type Map<K, V> = std::collections::HashMap<K, V>;

fn parse_example(example: &serde_json::Value) -> Example {
    let input = example["input"].as_array().expect("Should have been an array");
    let input = input.iter().map(|row| {
        row.as_array().expect("Should have been an array")
            .iter().map(|cell| {
                cell.as_i64().expect("Should have been an integer") as i32
            }).collect()
    }).collect();
    if example["output"].is_null() {
        return Example { input, output: vec![] };
    }
    let output = example["output"].as_array().expect("Should have been an array");
    let output = output.iter().map(|row| {
        row.as_array().expect("Should have been an array")
            .iter().map(|cell| {
                cell.as_i64().expect("Should have been an integer") as i32
            }).collect()
    }).collect();
    Example { input, output }
}
use colored::Color;
const COLORS: [Color; 12] = [
    Color::BrightWhite,
    Color::Black,
    Color::Blue,
    Color::Red,
    Color::Green,
    Color::Yellow,
    Color::TrueColor { r: 128, g: 0, b: 128 },
    Color::TrueColor { r: 255, g: 165, b: 0 },
    Color::TrueColor { r: 165, g: 42, b: 42 },
    Color::Magenta,
    Color::White,
    Color::Cyan,
];

fn print_color(color: i32) {
    if color == 0 {
        print!(" ");
    } else {
        print!("{}", "█".color(COLORS[color as usize]));
    }
}

fn print_image(image: &Image) {
    for row in image {
        for cell in row {
            print_color(*cell);
        }
        println!();
    }
}

fn print_example(example: &Example) {
    println!("Input:");
    print_image(&example.input);
    if !example.output.is_empty() {
        println!("Output:");
        print_image(&example.output);
    }
}

fn parse_task(task: &serde_json::Value) -> Task {
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

fn read_arc_file(file_path: &str) -> Map<String, Task> {
    let contents = fs::read_to_string(file_path)
        .expect("Should have been able to read the file");
    let data: serde_json::Value = serde_json::from_str(&contents)
        .expect("Should have been able to parse the json");
    let data = data.as_object().expect("Should have been an object");
    let mut tasks = Map::new();
    for (key, task) in data {
        tasks.insert(key.clone(), parse_task(task));
    }
    tasks
}

struct Shape {
    color: i32,
    cells: Vec<(i32, i32)>,
}

struct Vec2 {
    x: i32,
    y: i32,
}

const UP: Vec2 = Vec2 { x: 0, y: -1 };
const DOWN: Vec2 = Vec2 { x: 0, y: 1 };
const LEFT: Vec2 = Vec2 { x: -1, y: 0 };
const RIGHT: Vec2 = Vec2 { x: 1, y: 0 };
const DIRECTIONS: [Vec2; 4] = [UP, DOWN, LEFT, RIGHT];

fn find_shapes(image: &Image) -> Vec<Shape> {
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
            let mut cells = vec![(x as i32, y as i32)];
            visited[y][x] = true;
            let mut i = 0;
            while i < cells.len() {
                let (x, y) = cells[i];
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
                    cells.push((nx, ny));
                }
                i += 1;
            }
            shapes.push(Shape { color, cells });
        }
    }
    shapes
}

fn shape_by_color(shapes: &Vec<Shape>, color: i32) -> Option<&Shape> {
    for shape in shapes {
        if shape.color == color {
            return Some(shape);
        }
    }
    None
}

struct Box {
    top: i32,
    left: i32,
    bottom: i32,
    right: i32,
}

fn bounding_box(shape: &Shape) -> Box {
    let mut top = std::i32::MAX;
    let mut left = std::i32::MAX;
    let mut bottom = std::i32::MIN;
    let mut right = std::i32::MIN;
    for (x, y) in &shape.cells {
        top = top.min(*y);
        left = left.min(*x);
        bottom = bottom.max(*y);
        right = right.max(*x);
    }
    Box { top, left, bottom, right }
}

fn do_shapes_overlap(a: &Shape, b: &Shape) -> bool {
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
    for (x, y) in &a.cells {
        if b.cells.contains(&(*x, *y)) {
            return true;
        }
    }
    false
}

fn move_shape(shape: &Shape, vector: Vec2) -> Shape {
    let cells = shape.cells.iter().map(|(x, y)| {
        (*x + vector.x, *y + vector.y)
    }).collect();
    Shape { color: shape.color, cells }
}

fn paint_shape(image: &Image, shape: &Shape, color: i32) -> Image {
    let mut new_image = image.clone();
    for (x, y) in &shape.cells {
        new_image[*y as usize][*x as usize] = color;
    }
    new_image
}

fn remove_shape(image: &Image, shape: &Shape) -> Image {
    paint_shape(image, shape, 0)
}
fn draw_shape(image: &Image, shape: &Shape) -> Image {
    paint_shape(image, shape, shape.color)
}

// Moves the first shape pixel by pixel. (Not using bounding boxes.)
fn move_shape_to_shape_in_direction(image: &Image, to_move: &Shape, move_to: &Shape, dir: Vec2) -> Image {
    // Figure out moving distance.
    let mut distance = 1;
    loop {
        let moved = move_shape(to_move, Vec2 { x: dir.x * distance, y: dir.y * distance });
        if do_shapes_overlap(&moved, move_to) {
            distance -= 1;
            break;
        }
        distance += 1;
    }
    let mut new_image = image.clone();
    let moved = move_shape(to_move, Vec2 { x: dir.x * distance, y: dir.y * distance });
    new_image = remove_shape(&new_image, to_move);
    new_image = draw_shape(&new_image, &moved);
    new_image
}

// Moves the first shape in a cardinal direction until it touches the second shape.
fn move_shape_to_shape(image: &Image, to_move: &Shape, move_to: &Shape) -> Image {
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

fn solve_example(example: &Example) -> Image {
    let shapes = find_shapes(&example.input);
    let red = shape_by_color(&shapes, 8).expect("Should have been a shape");
    let blue = shape_by_color(&shapes, 2).expect("Should have been a shape");
    move_shape_to_shape(&example.input, &blue, &red)
}

fn main() {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let name = "05f2a901";
    let task = tasks.get(name).expect("Should have been a task");
    for example in &task.train {
        let solution = solve_example(example);
        print_example(example);
        print_image(&solution);
    }
    for example in &task.test {
        let solution = solve_example(example);
        print_example(example);
        print_image(&solution);
    }
}
