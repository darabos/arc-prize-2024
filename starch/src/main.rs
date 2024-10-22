use colored::Colorize;
use serde_json;
use std::collections::HashMap as Map;
use std::fs;
mod solvers;
mod tools;
use tools::{Example, Image, Task};

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

pub fn parse_task(task: &serde_json::Value) -> Task {
    let train = task["train"].as_array().expect("Should have been an array");
    let train = train.iter().map(|example| parse_example(example)).collect();
    let test = task["test"].as_array().expect("Should have been an array");
    let test = test.iter().map(|example| parse_example(example)).collect();
    Task { train, test }
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

fn main() {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let ref_solutions = read_arc_solutions_file("../arc-agi_training_solutions.json");
    let mut correct = 0;
    // let debug = (5, "0520fde7");
    let debug = (-1, "");
    for (name, task) in tasks.iter().filter(|(k, _)| debug.0 < 0 || *k == debug.1) {
        // println!("Task: {}", name);
        // tools::print_task(task);
        let state = solvers::SolverState::new(task);
        let active_solvers = if debug.0 < 0 {
            &solvers::SOLVERS
        } else {
            &solvers::SOLVERS[debug.0 as usize..debug.0 as usize + 1]
        };
        for solver in active_solvers {
            let mut s = state.clone();
            if let Err(error) = s.run_steps(solver) {
                if debug.0 >= 0 {
                    println!("{}: {}", name, error.red());
                }
                continue;
            }
            let solutions = s.get_results()[task.train.len()..].to_vec();
            let ref_images = ref_solutions
                .get(name)
                .expect("Should have been a solution");
            let mut all_correct = true;
            for i in 0..ref_images.len() {
                let ref_image = &ref_images[i];
                let image = &solutions[i].output;
                if debug.0 >= 0 {
                    tools::print_image(image);
                }
                if !tools::compare_images(ref_image, image) {
                    if debug.0 >= 0 {
                        println!("expected:");
                        tools::print_image(ref_image);
                    }
                    all_correct = false;
                    break;
                }
                if debug.0 < 0 {
                    tools::print_image(image);
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
