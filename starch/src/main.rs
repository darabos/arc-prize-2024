use colored::Colorize;
use indicatif::ProgressBar;
use rayon::prelude::*;
use serde::{self, ser::SerializeMap};
use serde_json;
use std::fs;
use std::{collections::HashMap as Map, rc::Rc};
mod image;
mod shape;
mod solvers;
mod steps;
mod tools;
use tools::{Example, Image, MutImage, Task};

static SEARCH_BUDGET: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("SEARCH_BUDGET")
        .unwrap_or(1000.to_string())
        .parse()
        .expect("Should have been a number")
});

pub fn parse_image(image: &serde_json::Value) -> Image {
    let vecvec: Vec<Vec<tools::Color>> = image
        .as_array()
        .expect("Should have been an array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("Should have been an array")
                .iter()
                .map(|cell| cell.as_i64().expect("Should have been an integer") as usize)
                .collect()
        })
        .collect();
    Image::from_vecvec(vecvec)
}

pub fn parse_example(example: &serde_json::Value) -> Example {
    let input = parse_image(&example["input"]);
    if example["output"].is_null() {
        return Example {
            input,
            output: MutImage::new(0, 0).freeze(),
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
pub fn parse_ref_images(task: &serde_json::Value) -> Vec<Image> {
    task["solution"]
        .as_array()
        .expect("Should have been an array")
        .iter()
        .map(|image| parse_image(image))
        .collect()
}

pub fn read_arc_file(file_path: &str) -> serde_json::Map<String, serde_json::Value> {
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    let data: serde_json::Value =
        serde_json::from_str(&contents).expect("Should have been able to parse the json");
    data.as_object()
        .expect("Should have been an object")
        .clone() // Not sure how to avoid this.
}

pub fn read_arc_solutions_file(file_path: &str) -> serde_json::Map<String, serde_json::Value> {
    let contents = fs::read_to_string(file_path).expect("Should have been able to read the file");
    let data: serde_json::Value =
        serde_json::from_str(&contents).expect("Should have been able to parse the json");
    data.as_object()
        .expect("Should have been an object")
        .clone() // Not sure how to avoid this.
}

pub fn read_files() -> Vec<(String, serde_json::Value)> {
    let tasks = read_arc_file("../arc-agi_training_challenges.json");
    let mut solutions = read_arc_solutions_file("../arc-agi_training_solutions.json");
    tasks
        .into_iter()
        .map(|(k, mut v)| {
            v["solution"] = solutions.remove(&k).unwrap_or_default();
            (k, v)
        })
        .collect::<Vec<(String, serde_json::Value)>>()
}

pub fn save_solution(file_path: &str, state: &solvers::SolverState) {
    save_submission(state);
    let mut data: serde_json::Value =
        if fs::exists(file_path).expect("Should have been able to check if the file exists") {
            let contents =
                fs::read_to_string(file_path).expect("Should have been able to read the file");
            serde_json::from_str(&contents).expect("Should have been able to parse the json")
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
    let data = data.as_object_mut().expect("Should have been an object");
    let key = &state.task.id;
    let step_list: Vec<String> = state.steps.iter().map(|step| step.to_string()).collect();
    if data.contains_key(key) {
        let existing_steps = data[key]
            .as_array()
            .expect("Should have been an array")
            .iter()
            .map(|step| step.as_str().expect("Should have been a string"))
            .collect::<Vec<&str>>();
        if existing_steps.len() <= step_list.len() {
            return;
        }
    }
    data.insert(key.clone(), serde_json::json!(step_list));
    let new_contents =
        serde_json::to_string_pretty(&data).expect("Should have been able to serialize the json");
    fs::write(file_path, new_contents).expect("Should have been able to write the file");
}

pub fn save_submission(state: &solvers::SolverState) {
    let file_path = "../submission.json";
    let mut data: serde_json::Value =
        if fs::exists(file_path).expect("Should have been able to check if the file exists") {
            let contents =
                fs::read_to_string(file_path).expect("Should have been able to read the file");
            serde_json::from_str(&contents).expect("Should have been able to parse the json")
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
    let data = data.as_object_mut().expect("Should have been an object");
    let key = &state.task.id;
    let examples: Vec<Example> = state.get_results()[state.task.train.len()..].to_vec();
    let images: Vec<Image> = examples
        .into_iter()
        .map(|image| image.output)
        .collect::<Vec<Image>>();
    let solutions: Vec<Vec<Vec<tools::Color>>> =
        images.into_iter().map(|image| image.to_vecvec()).collect();
    let attempts: Vec<Map<&str, Vec<Vec<tools::Color>>>> = solutions
        .into_iter()
        .map(|solution| Map::from([("attempt_1", solution.clone()), ("attempt_2", solution)]))
        .collect();
    data.insert(key.clone(), serde_json::json!(attempts));
    let new_contents =
        serde_json::to_string_pretty(&data).expect("Should have been able to serialize the json");
    fs::write(file_path, new_contents).expect("Should have been able to write the file");
}

pub fn read_list_of_solved_tasks() -> std::collections::HashSet<String> {
    let file_path = "../submission.json";
    let data: serde_json::Value =
        if fs::exists(file_path).expect("Should have been able to check if the file exists") {
            let contents =
                fs::read_to_string(file_path).expect("Should have been able to read the file");
            serde_json::from_str(&contents).expect("Should have been able to parse the json")
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
    let data = data.as_object().expect("Should have been an object");
    data.keys().map(|key| key.clone()).collect()
}

fn set_bar_style(bar: &ProgressBar) {
    bar.set_style(
        indicatif::ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .expect("no template error"),
    );
}

type StepCounts = Map<String, usize>;
/// The step counts given some prefix, and pointers to StepTrees for longer prefixes.
struct StepTree {
    counts: StepCounts,
    children: Map<String, StepTree>,
}
impl serde::Serialize for StepTree {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("counts", &self.counts)?;
        map.serialize_entry("children", &self.children)?;
        map.end()
    }
}
struct SolutionsHeuristics {
    step_tree: StepTree,
    step_costs: Map<String, f32>,
}
impl StepTree {
    fn get_counts(&self, prefix: &[String]) -> StepCounts {
        let mut counts = self.counts.clone();
        if let Some(last) = prefix.last() {
            if let Some(child) = self.children.get(last) {
                for (k, v) in child.get_counts(&prefix[..prefix.len() - 1]) {
                    *counts.entry(k).or_insert(0) += v * 10;
                }
            }
        }
        counts
    }
}

type ScoredStep = (f32, &'static solvers::SolverStep);
type ScoredSteps = Vec<ScoredStep>;
impl SolutionsHeuristics {
    fn get_candidates(&self, state: &solvers::SolverState) -> ScoredSteps {
        let step_list: Vec<String> = vec!["START".to_string()]
            .into_iter()
            .chain(state.steps.iter().map(|step| step.to_string()))
            .collect();
        let counts = self.step_tree.get_counts(&step_list);
        let state_score = self.state_score(state);
        solvers::ALL_STEPS
            .iter()
            .map(|step| {
                let count = counts.get(step.name()).copied().unwrap_or(0) as f32;
                let cost = self.step_costs.get(step.name()).copied().unwrap_or(0.);
                let use_count = state
                    .steps
                    .iter()
                    .filter(|&&n| std::ptr::eq(n, step))
                    .count() as f32;
                (count + state_score - cost - use_count * 10., step)
            })
            .collect()
    }
    fn state_score(&self, state: &solvers::SolverState) -> f32 {
        let mut score = 0.;
        score -= state.steps.len() as f32;
        score
    }
    fn load() -> Self {
        let contents = fs::read_to_string("../solutions.json")
            .expect("Should have been able to read the file");
        let data: serde_json::Value =
            serde_json::from_str(&contents).expect("Should have been able to parse the json");
        let solutions = data.as_object().expect("Should have been an object");
        let mut step_tree = StepTree {
            counts: Map::new(),
            children: Map::new(),
        };
        for (_id, steps) in solutions {
            // Load strings.
            let steps = steps
                .as_array()
                .expect("Should have been an array")
                .iter()
                .map(|step| step.as_str().expect("Should have been a string"))
                .collect::<Vec<&str>>();
            // println!("{}: {:?}", key, steps);
            for step_index in 0..steps.len() {
                let current_step = &steps[step_index];
                let mut st = &mut step_tree;
                *st.counts.entry(current_step.to_string()).or_insert(0) += 1;
                for prefix_length in 1..=step_index {
                    let step = &steps[step_index - prefix_length];
                    st = st.children.entry(step.to_string()).or_insert(StepTree {
                        counts: Map::new(),
                        children: Map::new(),
                    });
                    *st.counts.entry(current_step.to_string()).or_insert(0) += 1;
                }
                st = st.children.entry("START".to_owned()).or_insert(StepTree {
                    counts: Map::new(),
                    children: Map::new(),
                });
                *st.counts.entry(current_step.to_string()).or_insert(0) += 1;
            }
        }
        // let step_tree_json = serde_json::to_string_pretty(&step_tree)
        //     .expect("Should have been able to serialize the json");
        // println!("{}", step_tree_json);
        let mut step_costs = Map::new();
        for (name, cost) in solvers::STEP_COSTS {
            step_costs.insert(name.to_string(), *cost as f32);
        }
        Self {
            step_tree,
            step_costs,
        }
    }
}

struct SearchNode {
    score: f32,
    state: Rc<solvers::SolverState>,
    next_step: Option<&'static solvers::SolverStep>,
}
impl Eq for SearchNode {}
impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.partial_cmp(&other.score).unwrap()
    }
}

fn automatic_solver(task: &Task) -> tools::Res<solvers::SolverState> {
    let heur = SolutionsHeuristics::load();
    let mut queue = std::collections::BinaryHeap::new();
    queue.push(SearchNode {
        score: 0.,
        state: solvers::SolverState::new(task).into(),
        next_step: None,
    });
    let mut budget = *SEARCH_BUDGET;
    while let Some(node) = queue.pop() {
        if budget == 0 {
            break;
        }
        budget -= 1;
        let mut state = (*node.state).clone();
        if let Some(step) = node.next_step {
            if state.run_step_safe(step).is_ok() {
                // if let Err(error) = s.validate() {
                //     println!("{} after {}", error.red(), step);
                // }
                if state.correct_on_train() {
                    return Ok(state);
                }
            }
            // } else {
            //     println!("{:?}", heur.step_tree.counts);
        }
        // state.print_steps();
        let steps = heur.get_candidates(&state);
        // if node.next_step.is_none() {
        //     println!("candidates: {:?}", steps);
        // }
        let state_rc = Rc::new(state);
        for (score, step) in steps {
            // if node.next_step.is_none() {
            //     println!("adding {} with score {}", step.name(), score);
            // }
            // if score > 0. {
            // println!("{}: {}", step.name(), score);
            // }
            queue.push(SearchNode {
                score,
                state: state_rc.clone(),
                next_step: Some(step),
            });
        }
        // println!("{} budget remains, {} items in queue", budget, queue.len());
    }
    Err("No solution found")
}

#[allow(dead_code)]
fn evaluate_automatic_solver() {
    let tasks = read_files();
    let bar = ProgressBar::new(tasks.len() as u64);
    set_bar_style(&bar);
    let mutex = std::sync::Arc::new(std::sync::Mutex::new(()));
    let correct: usize = tasks
        .par_iter()
        .map(|(name, value)| {
            let task = parse_task(name, value);
            let ref_images = parse_ref_images(value);
            bar.inc(1);
            if let Ok(state) = automatic_solver(&task) {
                state.print_steps();
                let solutions = state.get_results()[task.train.len()..].to_vec();
                assert!(!ref_images.is_empty());
                let mut all_correct = true;
                for i in 0..ref_images.len() {
                    let ref_image = &ref_images[i];
                    let image = &solutions[i].output;
                    if ref_image != image {
                        println!("Task: {}", name);
                        image.print();
                        println!("expected:");
                        ref_image.print();
                        all_correct = false;
                        break;
                    }
                }
                if all_correct {
                    let _lock = mutex.lock().unwrap();
                    save_solution("../solutions.json", &state);
                    println!("{}: {}", name, "Correct".green());
                    return 1;
                }
            }
            0
        })
        .sum();
    bar.finish();
    println!("Correct: {}/{}", correct, tasks.len());
}

#[allow(dead_code)]
fn submit_with_automatic_solver() {
    let tasks = read_arc_file("../arc-agi_test_challenges.json");
    let already_solved = read_list_of_solved_tasks();
    let tasks = tasks
        .into_iter()
        .filter(|(k, _)| !already_solved.contains(k))
        .collect::<Vec<(String, serde_json::Value)>>();
    let bar = ProgressBar::new(tasks.len() as u64);
    set_bar_style(&bar);
    let mutex = std::sync::Arc::new(std::sync::Mutex::new(()));
    let found: usize = tasks
        .par_iter()
        .map(|(name, value)| {
            let task = parse_task(name, value);
            bar.inc(1);
            if let Ok(state) = automatic_solver(&task) {
                state.print_steps();
                let _lock = mutex.lock().unwrap();
                save_solution("../solutions.json", &state);
                1
            } else {
                0
            }
        })
        .sum();
    bar.finish();
    println!("Found {} possible solutions.", found);
}

#[allow(dead_code)]
fn evaluate_manual_solvers() {
    let tasks = read_files();
    let mut expected_correct: Vec<&str> = "007bbfb7 00d62c1b 025d127b 045e512c 0520fde7 05269061 05f2a901 06df4c85 08ed6ac7 09629e4f 0962bcdd 0a938d79 0b148d64 0ca9ddb6 0d3d703e 0dfd9992 0e206a2e 10fcaaa3 11852cab 1190e5a7 137eaa0f 150deff5 178fcbfb 1a07d186 1b2d62fb 1b60fb0c 1bfc4729 1c786137 1caeab9d 1cf80156 1e0a9b12 1e32b0e9 1f0c79e5 1f642eb9 1f85a75f 1f876c06 2013d3e2 2204b7a8 22168020 22eb0ac0 25ff71a9 29ec7d0e 2dc579da 2dee498d 3428a4f5 39a8645d 4258a5f9 48d8fb45 4c4377d9 6430c8c4 6d0aefbc 8403a5d5 90c28cc7 913fb3ed 963e52fc 99b1bc43 a416b8f3 a5313dff ae4f1146 b1948b0a ba97ae07 be94b721 c3f564a4 c8f0f002 ce4f8723 d364b489 d511f180 d687bc17 dc1df850 ded97339 e9afcf9a ea32f347 f2829549".split(" ").collect();
    let mutex = std::sync::Arc::new(std::sync::Mutex::new(()));
    // let debug = (42, "");
    let debug = (-1, "");
    let tasks: Vec<(String, serde_json::Value)> = if debug.0 < 0 {
        tasks
    } else {
        tasks.into_iter().filter(|(k, _)| *k == debug.1).collect()
    };
    let bar = ProgressBar::new(tasks.len() as u64);
    set_bar_style(&bar);
    let correct: Vec<&String> = tasks
        .par_iter()
        .map(|(name, value)| {
            let task = parse_task(name, value);
            let ref_images = parse_ref_images(value);
            bar.inc(1);
            let state = solvers::SolverState::new(&task);
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
                let mut all_correct = true;
                for i in 0..ref_images.len() {
                    let ref_image = &ref_images[i];
                    let image = &solutions[i].output;
                    if debug.0 >= 0 {
                        image.print();
                    }
                    if ref_image != image {
                        if debug.0 >= 0 {
                            println!("expected:");
                            ref_image.print();
                        }
                        all_correct = false;
                        break;
                    }
                    if debug.0 < 0 {
                        image.print();
                    }
                    println!("{}: {} (by {})", name, "Correct".green(), solver_index);
                }
                if all_correct {
                    let _lock = mutex.lock().unwrap();
                    save_solution("../solutions.json", &s);
                    return Some(name);
                }
            }
            None
        })
        .flatten()
        .collect();
    bar.finish();
    println!("Correct: {}/{}", correct.len(), tasks.len());
    for c in correct {
        if let Some(e) = expected_correct.iter().position(|&x| x == c) {
            print!("{} ", c.white());
            expected_correct.remove(e);
        } else {
            print!("{} ", c.green());
        }
        print!("{} ", c);
    }
    println!("");
    if !expected_correct.is_empty() {
        println!("Expected correct: {}", expected_correct.join(" ").red());
    }
}

fn main() {
    // evaluate_automatic_solver();
    // evaluate_manual_solvers();
    submit_with_automatic_solver();
}
