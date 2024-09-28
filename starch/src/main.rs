use colored::Colorize;
use serde_json;
use std::fs;

struct Task {
    train: Vec<Example>,
    test: Vec<Example>,
}

struct Example {
    input: Vec<Vec<i32>>,
    output: Vec<Vec<i32>>,
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

fn print_example(example: &Example) {
    println!("Input:");
    for row in &example.input {
        for cell in row {
            print_color(*cell);
        }
        println!();
    }
    println!("Output:");
    for row in &example.output {
        for cell in row {
            print_color(*cell);
        }
        println!();
    }
}

fn parse_task(task: &serde_json::Value) -> Task {
    let train = task["train"].as_array().expect("Should have been an array");
    let train = train.iter().map(|example| parse_example(example)).collect();
    let test = task["test"].as_array().expect("Should have been an array");
    let test = test.iter().map(|example| parse_example(example)).collect();
    Task { train, test }
}

fn print_task(task: &Task) {
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

fn main() {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let name = "05f2a901";
    let task = tasks.get(name).expect("Should have been a task");
    print_task(task);
}
