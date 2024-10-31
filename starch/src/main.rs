use colored::Colorize;
use indicatif::ProgressBar;
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

pub fn parse_task(id: &str, task: &serde_json::Value) -> Task {
    let train = task["train"].as_array().expect("Should have been an array");
    let train = train.iter().map(|example| parse_example(example)).collect();
    let test = task["test"].as_array().expect("Should have been an array");
    let test = test.iter().map(|example| parse_example(example)).collect();
    Task {
        id: id.into(),
        train,
        test,
    }
}

pub fn read_arc_file(file_path: &str) -> Vec<(String, Task)> {
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    let data: serde_json::Value =
        serde_json::from_str(&contents).expect("Should have been able to parse the json");
    let data = data.as_object().expect("Should have been an object");
    let mut tasks = Vec::new();
    for (key, task) in data {
        tasks.push((key.clone(), parse_task(key, task)));
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

fn set_bar_style(bar: &ProgressBar) {
    bar.set_style(
        indicatif::ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .expect("no template error"),
    );
}

fn automatic_solver(task: &Task) -> tools::Res<Vec<Example>> {
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(solvers::SolverState::new(task));
    let mut budget = 100;
    while let Some(state) = queue.pop_front() {
        if budget == 0 {
            break;
        }
        budget -= 1;
        for step in solvers::ALL_STEPS {
            let mut s = state.clone();
            if s.run_step_safe(step).is_ok() {
                if let Err(error) = s.validate() {
                    println!("{} after {}", error.red(), step);
                }
                if s.correct_on_train() {
                    return Ok(s.get_results()[task.train.len()..].to_vec());
                }
                queue.push_back(s);
            }
        }
    }
    Err("No solution found")
}

#[allow(dead_code)]
fn evaluate_automatic_solver() {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let ref_solutions = read_arc_solutions_file("../arc-agi_training_solutions.json");
    let mut correct = 0;
    // let debug = "0a938d79";
    let debug = "";
    let tasks: Vec<(String, Task)> = if debug == "" {
        tasks //.into_iter().skip(30).collect()
    } else {
        tasks.into_iter().filter(|(k, _)| *k == debug).collect()
    };
    let bar = ProgressBar::new(tasks.len() as u64);
    set_bar_style(&bar);
    for (name, task) in &tasks {
        bar.inc(1);
        // println!("Task: {}", name);
        // tools::print_task(task);
        if let Ok(solutions) = automatic_solver(task) {
            let ref_images = ref_solutions
                .get(name)
                .expect("Should have been a solution");
            let mut all_correct = true;
            for i in 0..ref_images.len() {
                let ref_image = &ref_images[i];
                let image = &solutions[i].output;
                if debug != "" {
                    tools::print_image(image);
                }
                if !tools::compare_images(ref_image, image) {
                    if debug != "" {
                        println!("expected:");
                        tools::print_image(ref_image);
                    }
                    all_correct = false;
                    break;
                }
                if debug != "" {
                    tools::print_image(image);
                }
            }
            if all_correct {
                println!("{}: {}", name, "Correct".green());
                correct += 1;
            }
        }
    }
    bar.finish();
    println!("Correct: {}/{}", correct, tasks.len());
}

#[allow(dead_code)]
fn evaluate_manual_solvers() {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let ref_solutions = read_arc_solutions_file("../arc-agi_training_solutions.json");
    let mut expected_correct: Vec<&str> = "007bbfb7 00d62c1b 025d127b 045e512c 0520fde7 05269061 05f2a901 06df4c85 08ed6ac7 09629e4f 0962bcdd 0a938d79 0b148d64 0ca9ddb6 0d3d703e 0dfd9992 0e206a2e 10fcaaa3 11852cab 1190e5a7 137eaa0f 150deff5 178fcbfb 1a07d186 1b2d62fb 1cf80156 22168020 22eb0ac0 29ec7d0e 2dc579da 3428a4f5 4258a5f9 6430c8c4 8403a5d5 90c28cc7 913fb3ed 99b1bc43 a5313dff b1948b0a ba97ae07 c3f564a4 c8f0f002 ce4f8723 d364b489 d511f180 d687bc17 dc1df850 ded97339 ea32f347 f2829549".split(" ").collect();
    let mut correct: Vec<colored::ColoredString> = vec![];
    // let debug = (25, "1b2d62fb");
    let debug = (-1, "");
    let tasks: Vec<(String, Task)> = if debug.0 < 0 {
        tasks
    } else {
        tasks.into_iter().filter(|(k, _)| *k == debug.1).collect()
    };
    let bar = ProgressBar::new(tasks.len() as u64);
    set_bar_style(&bar);
    for (name, task) in &tasks {
        bar.inc(1);
        // println!("Task: {}", name);
        // tools::print_task(task);
        let state = solvers::SolverState::new(task);
        let active_solvers = if debug.0 < 0 {
            &solvers::SOLVERS
        } else {
            &solvers::SOLVERS[debug.0 as usize..debug.0 as usize + 1]
        };
        for (solver_index, solver) in active_solvers.iter().enumerate() {
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
                println!("{}: {} (by {})", name, "Correct".green(), solver_index);
            }
            if all_correct {
                if let Some(e) = expected_correct.iter().position(|&x| x == name) {
                    correct.push(name.white());
                    expected_correct.remove(e);
                } else {
                    correct.push(name.green());
                }
                break;
            }
        }
    }
    bar.finish();
    println!("Correct: {}/{}", correct.len(), tasks.len());
    for c in correct {
        print!("{} ", c);
    }
    println!("");
    if !expected_correct.is_empty() {
        println!("Expected correct: {}", expected_correct.join(" ").red());
    }
}

fn main() {
    // evaluate_automatic_solver();
    evaluate_manual_solvers();
}
