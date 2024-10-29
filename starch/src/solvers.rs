use crate::tools;
use crate::tools::{Example, Image, Res, Shape, Task, COLORS};
use std::rc::Rc;

macro_rules! err {
    ($msg:expr) => {
        concat!($msg, " at ", file!(), ":", line!())
    };
}

/// Returns an error if the given Vec is empty.
macro_rules! must_not_be_empty {
    ($e:expr) => {
        if $e.is_empty() {
            return Err(err!("empty"));
        }
    };
}
/// Returns an error if any of the Vecs is empty.
macro_rules! must_all_be_non_empty {
    ($e:expr) => {
        if $e.iter().any(|e| e.is_empty()) {
            return Err(err!("empty"));
        }
    };
}

pub fn use_colorsets_as_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.colorsets.clone();
    Ok(())
}

fn order_colors_by_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    let mut already_used = vec![false; COLORS.len()];
    let mut next_color = 0;
    for shape in shapes.iter() {
        if !already_used[shape.color() as usize] {
            s.colors[i][next_color] = shape.color();
            already_used[shape.color() as usize] = true;
            next_color += 1;
        }
    }
    // Unused colors at the end.
    for color in 0..COLORS.len() {
        if !already_used[color] {
            s.colors[i][next_color] = color as i32;
            next_color += 1;
        }
    }
    Ok(())
}

/// Returns the first element of each list.
fn get_firsts<T>(vec: &Vec<Vec<T>>) -> Res<Vec<&T>> {
    let mut firsts = vec![];
    for e in vec {
        if e.is_empty() {
            return Err("empty list");
        }
        firsts.push(&e[0]);
    }
    Ok(firsts)
}

fn grow_flowers(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    // It's okay for some images to not have dots.
    let indexes: Vec<usize> = s
        .shapes
        .iter()
        .enumerate()
        .filter_map(|(i, shapes)| {
            if shapes.is_empty() || i >= s.output_images.len() {
                None
            } else {
                Some(i)
            }
        })
        .collect();
    if indexes.is_empty() {
        return Err(err!("no dots"));
    }
    let dots: Vec<&Rc<Shape>> = indexes.iter().map(|&i| &s.shapes[i][0]).collect();
    let output_images: Vec<Rc<Image>> = indexes
        .iter()
        .map(|&i| &s.output_images[i])
        .cloned()
        .collect();
    let mut output_pattern = tools::find_pattern_around(&output_images, &dots)?;
    output_pattern.use_relative_colors(&tools::reverse_colors(&s.colors[0]));
    // TODO: Instead of growing each dot, we should filter by the input_pattern.
    s.apply(|s: &mut SolverState, i: usize| {
        let mut new_image = (*s.images[i]).clone();
        for dots in &s.shapes[i] {
            for dot in dots.cells.iter() {
                tools::draw_shape_with_relative_colors_at(
                    &mut new_image,
                    &output_pattern,
                    &s.colors[i],
                    &dot.pos(),
                );
            }
        }
        s.images[i] = Rc::new(new_image);
        Ok(())
    })
}

fn save_shapes(s: &mut SolverState) -> Res<()> {
    s.saved_shapes.push(s.shapes.clone());
    Ok(())
}

fn load_earlier_shapes(s: &mut SolverState, offset: usize) -> Res<()> {
    if s.saved_shapes.len() < offset + 1 {
        return Err(err!("no saved shapes"));
    }
    s.shapes = s.saved_shapes[s.saved_shapes.len() - 1 - offset].clone();
    Ok(())
}

fn save_shapes_and_load_previous(s: &mut SolverState) -> Res<()> {
    save_shapes(s)?;
    load_earlier_shapes(s, 1)
}

fn save_first_shape_use_the_rest(s: &mut SolverState) -> Res<()> {
    if s.shapes.iter().any(|shapes| shapes.len() < 2) {
        return Err(err!("not enough shapes"));
    }
    let first_shapes: ShapesPerExample = s
        .shapes
        .iter()
        .map(|shapes| vec![shapes[0].clone()])
        .collect();
    s.saved_shapes.push(first_shapes);
    s.shapes = std::mem::take(&mut s.shapes)
        .into_iter()
        .map(|shapes| shapes[1..].to_vec())
        .collect();
    Ok(())
}

fn load_shapes(s: &mut SolverState) -> Res<()> {
    load_earlier_shapes(s, 0)
}

fn load_shapes_except_current_shapes(s: &mut SolverState) -> Res<()> {
    let excluded_shapes = std::mem::take(&mut s.shapes);
    load_shapes(s)?;
    s.shapes = std::mem::take(&mut s.shapes)
        .into_iter()
        .zip(excluded_shapes)
        .map(|(shapes, excluded)| {
            shapes
                .iter()
                .filter(|shape| !excluded.contains(shape))
                .cloned()
                .collect()
        })
        .collect();
    must_all_be_non_empty!(&s.shapes);
    Ok(())
}

fn draw_saved_shapes(s: &mut SolverState) -> Res<()> {
    load_shapes(s)?;
    s.apply(draw_shapes)
}

fn save_whole_image(s: &mut SolverState) -> Res<()> {
    s.saved_images.push(s.images.clone());
    Ok(())
}

type Shapes = Vec<Rc<Shape>>;
type ShapesPerExample = Vec<Shapes>;
type ImagePerExample = Vec<Rc<Image>>;
type ColorList = Vec<i32>;
type ColorListPerExample = Vec<ColorList>;
type LinesPerExample = Vec<Rc<tools::LineSet>>;
type NumberSequence = Vec<i32>;
type NumberSequencesPerExample = Vec<NumberSequence>;

/// A sub-state that contains only one image and its shapes.
/// It uses the image from the parent state and applies operations on it.
#[derive(Default, Clone)]
pub struct SubState {
    state: SolverState,
    image_indexes: Vec<usize>,
}

/// Tracks information while applying operations on all examples at once.
/// Most fields are vectors storing information for each example.
#[derive(Default, Clone)]
pub struct SolverState {
    pub task: Rc<Task>,
    pub images: ImagePerExample,
    pub saved_images: Vec<ImagePerExample>,
    pub output_images: ImagePerExample,
    pub colors: ColorListPerExample,
    pub shapes: ShapesPerExample,
    pub shapes_including_background: ShapesPerExample,
    pub output_shapes: ShapesPerExample,
    pub output_shapes_including_background: ShapesPerExample,
    pub saved_shapes: Vec<ShapesPerExample>,
    pub colorsets: ShapesPerExample,
    pub scale_up: tools::Vec2,
    pub last_move: Vec<tools::Vec2>,
    // Lines that go all the way through the image.
    pub lines: LinesPerExample,
    pub saved_lines: Vec<LinesPerExample>,
    pub number_sequences: NumberSequencesPerExample,
    pub output_number_sequences: NumberSequencesPerExample,
    // If this is set, we will apply steps to these states.
    pub substates: Option<Vec<SubState>>,
    // Steps made so far. For the record.
    pub steps: Vec<&'static SolverStep>,
    // Steps to be done at the end. (In get_results().)
    pub finishing_steps: Vec<Rc<Box<dyn Fn(&mut SolverState) -> Res<()>>>>,
    pub is_substate: bool,
}

impl SolverState {
    pub fn new(task: &Task) -> Self {
        let images: Vec<Rc<Image>> = task
            .train
            .iter()
            .chain(task.test.iter())
            .map(|example| Rc::new(example.input.clone()))
            .collect();
        let output_images = task
            .train
            .iter()
            .map(|example| Rc::new(example.output.clone()))
            .collect();
        let mut state = SolverState {
            task: Rc::new(task.clone()),
            output_images,
            last_move: vec![tools::Vec2::ZERO; images.len()],
            ..Default::default()
        };
        state.init_from_images(images);
        state
    }

    fn init_from_images(&mut self, images: ImagePerExample) {
        self.images = images;
        self.colorsets = self
            .images
            .iter()
            .map(|image| tools::find_colorsets_in_image(image))
            .collect();
        self.shapes_including_background = self
            .images
            .iter()
            .map(|image| tools::find_shapes_in_image(image, &tools::DIRECTIONS4))
            .collect();
        self.output_shapes_including_background = self
            .output_images
            .iter()
            .map(|image| tools::find_shapes_in_image(image, &tools::DIRECTIONS4))
            .collect();
        // Discard background shapes that touch the border.
        self.shapes = self
            .shapes_including_background
            .iter()
            .zip(&self.images)
            .map(|(shapes, image)| {
                tools::discard_background_shapes_touching_border(image, shapes.clone())
            })
            .collect();
        self.output_shapes = self
            .output_shapes_including_background
            .iter()
            .zip(&self.output_images)
            .map(|(shapes, image)| {
                tools::discard_background_shapes_touching_border(image, shapes.clone())
            })
            .collect();
        let all_colors: ColorList = (0..COLORS.len() as i32).collect();
        self.colors = self.images.iter().map(|_| all_colors.clone()).collect();
        for s in &mut self.shapes {
            s.sort_by_key(|shape| shape.color());
        }
        self.apply(order_colors_by_shapes).unwrap();
        self.saved_shapes = vec![self.shapes.clone()];
        self.lines = self
            .images
            .iter()
            .map(|image| Rc::new(tools::find_lines_in_image(image)))
            .collect();
    }

    pub fn validate(&self) -> Res<()> {
        if self.images.is_empty() {
            return Err("no images");
        }
        if self.images.iter().any(|image| image.is_empty()) {
            return Err("empty image");
        }
        if self.images.iter().any(|image| image[0].is_empty()) {
            return Err("empty image");
        }
        if !self.output_images.is_empty() {
            if self.output_images.iter().any(|image| image.is_empty()) {
                return Err("empty output image");
            }
            if self.output_images.iter().any(|image| image[0].is_empty()) {
                return Err("empty output image");
            }
        }
        if self.colors.is_empty() {
            return Err("no colors");
        }
        if self.colors.len() != self.images.len() {
            return Err("wrong number of color lists");
        }
        if self.shapes.is_empty() {
            return Err("no shapes");
        }
        if self.shapes.len() != self.images.len() {
            return Err("wrong number of shape lists");
        }
        if self.colorsets.is_empty() {
            return Err("no colorsets");
        }
        if self.colorsets.len() != self.images.len() {
            return Err("wrong number of colorset lists");
        }
        if self.lines.is_empty() {
            return Err("no lines");
        }
        if self.lines.len() != self.images.len() {
            return Err("wrong number of line sets");
        }
        if let Some(substates) = &self.substates {
            for substate in substates {
                substate.state.validate()?;
            }
        }
        Ok(())
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

    pub fn get_results(&self) -> Vec<Example> {
        fn images_to_examples(state: &SolverState) -> Vec<Example> {
            state
                .images
                .iter()
                .zip(state.task.train.iter().chain(state.task.test.iter()))
                .map(|(image, example)| Example {
                    input: example.input.clone(),
                    output: (**image).clone(),
                })
                .collect()
        }
        if self.finishing_steps.is_empty() {
            images_to_examples(self)
        } else {
            let mut finished = self.clone();
            for step in &self.finishing_steps {
                step(&mut finished).unwrap();
            }
            images_to_examples(&finished)
        }
    }
    fn width_and_height(&self, i: usize) -> (i32, i32) {
        tools::width_and_height(&self.images[i])
    }
    fn width_and_height_all(&self) -> Res<(i32, i32)> {
        let (w, h) = self.width_and_height(0);
        for i in 1..self.images.len() {
            let (w1, h1) = self.width_and_height(i);
            if w1 != w || h1 != h {
                return Err("images have different sizes");
            }
        }
        Ok((w, h))
    }
    fn output_width_and_height(&self, i: usize) -> (i32, i32) {
        tools::width_and_height(&self.output_images[i])
    }
    fn output_width_and_height_all(&self) -> Res<(i32, i32)> {
        let (w, h) = self.output_width_and_height(0);
        for i in 1..self.output_images.len() {
            let (w1, h1) = self.output_width_and_height(i);
            if w1 != w || h1 != h {
                return Err("output images have different sizes");
            }
        }
        Ok((w, h))
    }

    #[allow(dead_code)]
    fn print_shapes(&self) {
        print_shapes(&self.shapes);
    }
    #[allow(dead_code)]
    fn print_saved_shapes(&self) {
        for per_example in &self.saved_shapes {
            for shapes in per_example {
                for shape in shapes {
                    shape.print();
                    println!();
                }
            }
        }
    }
    #[allow(dead_code)]
    fn print_colorsets(&self) {
        print_shapes(&self.colorsets);
    }
    #[allow(dead_code)]
    fn print_images(&self) {
        for image in &self.images {
            tools::print_image(image);
            println!();
        }
    }

    #[allow(dead_code)]
    fn print_colors(&self) {
        for colors in &self.colors {
            for color in colors {
                tools::print_color(*color);
            }
            println!();
        }
    }

    #[allow(dead_code)]
    fn print_steps(&self) {
        for step in &self.steps {
            println!("- {}", step);
        }
    }

    /// An iterator for sub-states that contain only one image and its shapes.
    pub fn substate_per_image(&self) -> Vec<SubState> {
        (0..self.images.len())
            .map(|i| SubState {
                image_indexes: vec![i],
                state: SolverState {
                    task: self.task.clone(),
                    images: vec![self.images[i].clone()],
                    output_images: self.output_images.get(i).cloned().into_iter().collect(),
                    saved_images: self
                        .saved_images
                        .iter()
                        .map(|s| vec![s[i].clone()])
                        .collect(),
                    colors: vec![self.colors[i].clone()],
                    shapes: vec![self.shapes[i].clone()],
                    shapes_including_background: vec![self.shapes_including_background[i].clone()],
                    output_shapes: vec![],
                    output_shapes_including_background: vec![],
                    saved_shapes: self
                        .saved_shapes
                        .iter()
                        .map(|s| vec![s[i].clone()])
                        .collect(),
                    colorsets: vec![self.colorsets[i].clone()],
                    scale_up: self.scale_up,
                    last_move: self.last_move.clone(),
                    lines: vec![self.lines[i].clone()],
                    saved_lines: self
                        .saved_lines
                        .iter()
                        .map(|s| vec![s[i].clone()])
                        .collect(),
                    number_sequences: if self.number_sequences.is_empty() {
                        vec![]
                    } else {
                        vec![self.number_sequences[i].clone()]
                    },
                    output_number_sequences: vec![],
                    substates: None,
                    steps: vec![],
                    finishing_steps: vec![],
                    is_substate: true,
                },
            })
            .collect()
    }

    /// A substate that is the same as this state. It carries all images.
    pub fn substate(&self) -> SubState {
        let mut state = self.clone();
        state.is_substate = true;
        SubState {
            image_indexes: (0..self.images.len()).collect(),
            state,
        }
    }

    pub fn run_steps(&mut self, steps: &'static [SolverStep]) -> Res<()> {
        for step in steps {
            self.validate()?;
            self.run_step_safe(step)?;
        }
        Ok(())
    }

    pub fn run_step_safe(&mut self, step: &'static SolverStep) -> Res<()> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.run_step(step)));
        match result {
            Ok(res) => res,
            Err(_) => {
                tools::print_task(&self.task);
                self.print_steps();
                Err(err!("panic"))
            }
        }
    }

    pub fn run_step(&mut self, step: &'static SolverStep) -> Res<()> {
        self.steps.push(step);
        if let Some(substates) = &mut self.substates {
            for substate in substates {
                let state = &mut substate.state;
                state.images = substate
                    .image_indexes
                    .iter()
                    .map(|&i| self.images[i].clone())
                    .collect();
                state.run_step(&step)?;
                for (i, &image_index) in substate.image_indexes.iter().enumerate() {
                    self.images[image_index] = state.images[i].clone();
                }
            }
            return Ok(());
        }
        match step {
            SolverStep::Each(_name, f) => self.apply(f)?,
            SolverStep::All(_name, f) => f(self)?,
        }
        Ok(())
    }

    pub fn correct_on_train(&self) -> bool {
        self.images[..self.task.train.len()]
            .iter()
            .zip(self.output_images.iter())
            .all(|(image, output)| tools::compare_images(image, output))
    }

    pub fn add_finishing_step(&mut self, f: impl Fn(&mut SolverState) -> Res<()> + 'static) {
        self.finishing_steps.push(Rc::new(Box::new(f)));
    }
}

#[allow(dead_code)]
fn print_shapes(shapes: &ShapesPerExample) {
    for (i, shapes) in shapes.iter().enumerate() {
        println!("{} shapes for example {}", shapes.len(), i);
        for shape in shapes {
            shape.print();
            println!();
        }
    }
}

#[allow(dead_code)]
fn print_shapes_step(s: &mut SolverState) -> Res<()> {
    s.print_shapes();
    Ok(())
}

#[allow(dead_code)]
fn print_images_step(s: &mut SolverState) -> Res<()> {
    s.print_images();
    Ok(())
}

fn substates_for_each_image(s: &mut SolverState) -> Res<()> {
    if s.is_substate {
        return Err(err!("already split into substates"));
    }
    s.substates = Some(s.substate_per_image());
    Ok(())
}

fn substates_for_each_shape(s: &mut SolverState) -> Res<()> {
    if s.is_substate {
        return Err(err!("already split into substates"));
    }
    if s.shapes.iter().any(|shapes| shapes.len() > 10) {
        return Err(err!("too many shapes"));
    }
    s.substates = Some(
        s.substate_per_image()
            .into_iter()
            .map(|mut s| {
                let shapes = std::mem::take(&mut s.state.shapes[0]);
                shapes.into_iter().map(move |shape| {
                    let mut state = s.state.clone();
                    state.shapes = vec![vec![shape.clone()]];
                    SubState {
                        state,
                        image_indexes: s.image_indexes.clone(),
                    }
                })
            })
            .flatten()
            .collect(),
    );
    Ok(())
}

fn substates_for_each_color(s: &mut SolverState) -> Res<()> {
    if s.is_substate {
        return Err(err!("already split into substates"));
    }
    let used_colors = tools::get_used_colors(&s.images);
    if used_colors.is_empty() {
        return Err(err!("no colors"));
    }
    s.substates = Some(
        used_colors
            .into_iter()
            .map(|color| {
                let mut substate = s.substate();
                substate.state.colors =
                    vec![tools::add_remaining_colors(&vec![color]); s.images.len()];
                substate
            })
            .collect(),
    );
    Ok(())
}

fn use_next_color(s: &mut SolverState, i: usize) -> Res<()> {
    let first_color = s.colors[i][0];
    let n = COLORS.len();
    for j in 0..n - 1 {
        s.colors[i][j] = s.colors[i][j + 1];
    }
    s.colors[i][n - 1] = first_color;
    Ok(())
}

fn use_previous_color(s: &mut SolverState, i: usize) -> Res<()> {
    let last_color = s.colors[i][COLORS.len() - 1];
    let n = COLORS.len();
    for j in (1..n).rev() {
        s.colors[i][j] = s.colors[i][j - 1];
    }
    s.colors[i][0] = last_color;
    Ok(())
}

fn filter_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    let color = s.colors[i].get(0).ok_or("no used colors")?;
    s.shapes[i] = shapes
        .iter()
        .filter(|shape| shape.color() == *color)
        .cloned()
        .collect();
    Ok(())
}

fn move_shapes_to_touch_saved_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
    let saved_shapes = &s.saved_shapes.last().ok_or("must have saved shapes")?[i];
    let saved_shape = saved_shapes.get(0).ok_or("saved shapes empty")?;
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        new_image = tools::move_shape_to_shape_in_image(&new_image, &shape, &saved_shape)?;
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

/// Scales up the image to match the output image.
fn scale_up_image(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    // Find ratio from looking at example outputs.
    let output_size = s.output_images[0].len() as i32;
    let input_size = s.images[0].len() as i32;
    if output_size % input_size != 0 {
        return Err(err!("output size must be a multiple of input size"));
    }
    let scale = output_size / input_size;
    s.scale_up = tools::Vec2 { x: scale, y: scale };
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::scale_up_image(&s.images[i], s.scale_up));
        Ok(())
    })
}

/// Scales up the image to match the output image after adding a grid stored in horizontal_lines and
/// vertical_lines.
fn scale_up_image_add_grid(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    let lines = &s.saved_lines;
    must_not_be_empty!(lines);
    let lines = &lines[lines.len() - 1];
    let num_h = lines[0].horizontal.len() as i32;
    let num_v = lines[0].vertical.len() as i32;
    if num_h + num_v == 0 {
        return Err(err!("no grid"));
    }
    if lines.iter().any(|lines| {
        lines.horizontal.len() != num_h as usize || lines.vertical.len() != num_v as usize
    }) {
        return Err(err!("lines have different lengths"));
    }
    // Find ratio from looking at example outputs.
    let (output_width, output_height) = s.output_width_and_height_all()?;
    let (width, height) = s.width_and_height_all()?;
    if (output_width - num_v) % width != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    if (output_height - num_h) % height != 0 {
        return Err(err!("output width must be a multiple of input width"));
    }
    s.scale_up = tools::Vec2 {
        x: (output_width - num_v) / width,
        y: (output_height - num_h) / height,
    };
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::scale_up_image(&s.images[i], s.scale_up));
        Ok(())
    })?;
    restore_grid(s)
}

fn use_image_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i] = vec![Rc::new(Shape::from_image(&s.images[i]))];
    Ok(())
}

fn use_image_without_background_as_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let mut shape = Shape::from_image(&s.images[i]);
    shape.discard_color(0);
    if shape.cells.is_empty() {
        return Err(err!("no non-background cells"));
    }
    s.shapes[i] = vec![shape.into()];
    Ok(())
}

fn tile_shapes_after_scale_up(s: &mut SolverState, i: usize) -> Res<()> {
    if s.scale_up.x <= 1 && s.scale_up.y <= 1 {
        return Err(err!("not scaled up"));
    }
    let (current_width, current_height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    let old_width = current_width / s.scale_up.x;
    let old_height = current_height / s.scale_up.y;
    for shape in shapes.iter_mut() {
        *shape = shape
            .tile(old_width, current_width, old_height, current_height)?
            .into();
    }
    Ok(())
}

fn draw_shape_where_non_empty(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        shape.draw_where_non_empty(&mut new_image);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn delete_shapes_touching_border(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    *shapes = shapes
        .iter()
        .filter(|shape| !shape.is_touching_border(&s.images[i]))
        .cloned()
        .collect();
    must_not_be_empty!(shapes);
    Ok(())
}

fn recolor_shapes_per_output(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    // Get colors from first output_image.
    let shapes = &s.shapes[0];
    let colors: Res<Vec<i32>> = shapes
        .iter()
        .map(|shape| {
            tools::lookup_in_image(&s.output_images[0], shape.cells[0].x, shape.cells[0].y)
        })
        .collect();
    let mut colors = colors?;
    if colors.is_empty() {
        return Err(err!("no colors"));
    }
    let all_same_color = colors.iter().all(|&c| c == colors[0]);
    // Fail the operation if any shape in any output has a different color.
    for (i, image) in s.output_images.iter_mut().enumerate() {
        for (j, shape) in &mut s.shapes[i].iter().enumerate() {
            let cell = &shape.cells[0];
            if let Ok(c) = tools::lookup_in_image(image, cell.x, cell.y) {
                if all_same_color {
                    if c != colors[0] {
                        return Err(err!("output shapes have different colors"));
                    }
                } else {
                    while j >= colors.len() {
                        colors.push(c);
                    }
                    if c != colors[j] {
                        return Err(err!("output shapes have different colors"));
                    }
                }
            }
        }
    }
    // Recolor shapes.
    for shapes in &mut s.shapes {
        for (j, shape) in shapes.iter_mut().enumerate() {
            let mut new_shape = (**shape).clone();
            let color = if all_same_color {
                colors[0]
            } else {
                colors[j % colors.len()]
            };
            new_shape.recolor(color);
            *shape = new_shape.into();
        }
    }
    Ok(())
}

fn draw_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &s.shapes[i];
    let mut new_image = (*s.images[i]).clone();
    for shape in shapes {
        tools::draw_shape(&mut new_image, shape);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn order_shapes_by_color(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.color());
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.color());
    }
    Ok(())
}

fn order_shapes_by_size_decreasing(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| -(shape.cells.len() as i32));
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| -(shape.cells.len() as i32));
    }
    Ok(())
}

fn order_shapes_by_size_increasing(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.cells.len());
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.cells.len());
    }
    Ok(())
}

fn order_shapes_from_left_to_right(s: &mut SolverState, i: usize) -> Res<()> {
    s.shapes[i].sort_by_key(|shape| shape.bb.left);
    if i < s.output_shapes.len() {
        s.output_shapes[i].sort_by_key(|shape| shape.bb.left);
    }
    Ok(())
}

fn find_repeating_pattern_in_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    let shape = shapes.get(0).ok_or(err!("no shapes"))?;
    if shape.bb.width() < 3 && shape.bb.height() < 3 {
        return Err(err!("shape too small"));
    }
    let mut w = 1;
    let mut h = 1;
    for _ in 0..10 {
        let pattern = tile_shape(shape, w, h, width, height);
        if let Ok(pattern) = pattern {
            s.shapes[i] = vec![pattern.into()];
            return Ok(());
        }
        if w < h && w < width {
            w += 1;
        } else {
            h += 1;
            if height < h {
                break;
            }
        }
    }
    Err(err!("no repeating pattern found"))
}

fn tile_shape(shape: &Shape, shape_w: i32, shape_h: i32, image_w: i32, image_h: i32) -> Res<Shape> {
    let p = shape.crop(0, 0, shape_w, shape_h)?;
    let p1 = p.tile(shape_w, shape.bb.right, shape_h, shape.bb.bottom)?;
    if p1 != *shape {
        return Err(err!("not matching"));
    }
    p.tile(shape_w, image_w, shape_h, image_h)
}

fn use_output_size(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    let (current_width, current_height) = s.width_and_height(0);
    let (output_width, output_height) = s.output_width_and_height(0);
    if current_width == output_width && current_height == output_height {
        return Err(err!("already correct size"));
    }
    // Make sure all outputs have the same size.
    for i in 1..s.output_images.len() {
        let (w, h) = s.output_width_and_height(i);
        if w != output_width || h != output_height {
            return Err(err!("output images have different sizes"));
        }
    }
    s.apply(|s: &mut SolverState, i: usize| {
        s.images[i] = Rc::new(tools::resize_canvas(
            &s.images[i],
            output_width as usize,
            output_height as usize,
        ));
        Ok(())
    })
}

fn pick_bottom_right_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    let shape = shapes
        .iter()
        .max_by_key(|shape| shape.cells[0])
        .ok_or(err!("no shapes"))?;
    *shapes = vec![shape.clone()];
    Ok(())
}

fn pick_bottom_right_shape_per_color(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    let mut new_shapes = vec![];
    for color in &s.colors[i] {
        let shape = shapes
            .iter()
            .filter(|shape| shape.color() == *color)
            .max_by_key(|shape| shape.bb.bottom_right());
        if let Some(shape) = shape {
            new_shapes.push(shape.clone());
        }
    }
    if new_shapes.is_empty() {
        return Err(err!("no shapes"));
    }
    *shapes = new_shapes;
    Ok(())
}

fn allow_diagonals_in_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut shapes = tools::find_shapes_in_image(&s.images[i], &tools::DIRECTIONS8);
    shapes = tools::discard_background_shapes_touching_border(&s.images[i], shapes);
    s.shapes[i] = shapes;
    Ok(())
}

fn delete_background_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes[i];
    *shapes = shapes
        .iter()
        .filter(|shape| shape.color() != 0)
        .cloned()
        .collect();
    must_not_be_empty!(shapes);
    Ok(())
}

fn move_shapes_per_output_shapes(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_shapes);
    let shapes0 = &s.shapes[0];
    let output_shapes0 = &s.output_shapes[0];
    let relative_output_shapes0: Shapes = output_shapes0
        .iter()
        .map(|shape| shape.to_relative_pos().into())
        .collect();
    // Figure out the offset.
    let in00 = shapes0.get(0).ok_or(err!("no shape"))?;
    let out0_index = in00
        .to_relative_pos()
        .find_matching_shape_index(&relative_output_shapes0)
        .ok_or(err!("no match"))?;
    let out0 = &output_shapes0[out0_index];
    let offset = in00.cells[0] - out0.cells[0];
    // Confirm that this offset is correct for all shapes in all examples.
    for i in 0..s.output_images.len() {
        let shapes = &s.shapes[i];
        let output_shapes = &s.output_shapes[i];
        let relative_output_shapes: Shapes = output_shapes
            .iter()
            .map(|shape| shape.to_relative_pos().into())
            .collect();
        for shape in shapes {
            let out_index = shape
                .to_relative_pos()
                .find_matching_shape_index(&relative_output_shapes)
                .ok_or(err!("no match"))?;
            let out = &output_shapes[out_index];
            if shape.cells[0] - out.cells[0] != offset {
                return Err(err!("offsets don't match"));
            }
        }
    }
    // Move shapes.
    for i in 0..s.images.len() {
        let shapes = &mut s.shapes[i];
        *shapes = shapes
            .iter()
            .map(|shape| shape.move_by(offset).into())
            .collect();
    }
    Ok(())
}

fn move_shapes_per_output(s: &mut SolverState) -> Res<()> {
    let shapes = &s.shapes;
    let outputs = &s.output_images;
    for distance in 1..5 {
        for direction in tools::DIRECTIONS8 {
            let mut correct = true;
            let offset = distance * direction;
            'images: for i in 0..outputs.len() {
                for shape in &shapes[i] {
                    if !shape.matches_image_when_moved_by(&outputs[i], offset) {
                        correct = false;
                        break 'images;
                    }
                }
            }
            if correct {
                for i in 0..s.images.len() {
                    let mut new_image = (*s.images[i]).clone();
                    for shape in &shapes[i] {
                        tools::remove_shape(&mut new_image, shape);
                        tools::draw_shape_at(&mut new_image, shape, offset);
                    }
                    s.images[i] = Rc::new(new_image);
                }
                return Ok(());
            }
        }
    }
    Err(err!("no match found"))
}

/// Moves the saved shape in one of the 8 directions as far as possible while still covering the
/// current shape. Works with a single saved shape and current shape.
fn move_saved_shape_to_cover_current_shape_max(s: &mut SolverState, i: usize) -> Res<()> {
    let saved_shapes = &s.saved_shapes.last().ok_or(err!("no saved shapes"))?[i];
    let current_shape = &s.shapes[i].get(0).ok_or(err!("no current shape"))?;
    let saved_shape = saved_shapes.get(0).ok_or(err!("no saved shape"))?;
    let mut moved: Shape = (**saved_shape).clone();
    for distance in (1..10).rev() {
        for direction in tools::DIRECTIONS8 {
            moved.move_by_mut(distance * direction);
            if moved.covers(current_shape) {
                s.shapes[i] = vec![moved.into()];
                s.last_move[i] = distance * direction;
                return Ok(());
            }
            moved.restore_from(&saved_shape);
        }
    }
    Err(err!("no move found"))
}

/// Draws the shape in its current location, then moves it again and draws it again,
/// until it leaves the image.
fn repeat_last_move_and_draw(s: &mut SolverState, i: usize) -> Res<()> {
    if s.last_move[i] == tools::Vec2::ZERO {
        return Err(err!("no last move"));
    }
    let mut shapes: Vec<Shape> = s.shapes[i].iter().map(|shape| (**shape).clone()).collect();
    let mut new_image = (*s.images[i]).clone();
    for _ in 0..10 {
        for shape in &mut shapes {
            tools::draw_shape(&mut new_image, &shape);
            shape.move_by_mut(s.last_move[i]);
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn recolor_saved_shapes_to_current_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let saved_shapes = &s.saved_shapes.last().ok_or(err!("no saved shapes"))?[i];
    let current_shape = &s.shapes[i].get(0).ok_or(err!("no current shape"))?;
    let color = current_shape.color();
    let mut new_saved_shapes = vec![];
    for saved_shape in saved_shapes {
        let mut new_shape = (**saved_shape).clone();
        new_shape.recolor(color);
        new_saved_shapes.push(new_shape.into());
    }
    let len = s.saved_shapes.len();
    s.saved_shapes[len - 1][i] = new_saved_shapes;
    Ok(())
}

fn split_into_two_images(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.output_images);
    let (width, height) = s.width_and_height_all()?;
    let (output_width, output_height) = s.output_width_and_height_all()?;
    if width == output_width * 2 + 1 {
        let mut to_save = vec![];
        for image in &mut s.images {
            let left_image = Rc::new(tools::crop_image(image, 0, 0, width / 2, height));
            let right_image = Rc::new(tools::crop_image(
                image,
                width / 2 + 1,
                0,
                width / 2,
                height,
            ));
            *image = left_image;
            to_save.push(right_image);
        }
        s.saved_images.push(to_save);
        return Ok(());
    } else if height == output_height * 2 + 1 {
        let mut to_save = vec![];
        for image in &mut s.images {
            let top_image = Rc::new(tools::crop_image(image, 0, 0, width, height / 2));
            let bottom_image = Rc::new(tools::crop_image(
                image,
                0,
                height / 2 + 1,
                width,
                height / 2,
            ));
            *image = top_image;
            to_save.push(bottom_image);
        }
        s.saved_images.push(to_save);
        return Ok(());
    }
    Err(err!("no split found"))
}

fn boolean_with_saved_image_and(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if a == 0 { 0 } else { b })
}
fn boolean_with_saved_image_or(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| if a == 0 { b } else { a })
}
fn boolean_with_saved_image_xor(s: &mut SolverState, i: usize) -> Res<()> {
    boolean_with_saved_image_function(s, i, |a, b| {
        if a == 0 {
            b
        } else if b == 0 {
            a
        } else {
            0
        }
    })
}

fn boolean_with_saved_image_function(
    s: &mut SolverState,
    i: usize,
    func: fn(i32, i32) -> i32,
) -> Res<()> {
    let saved_image = &s.saved_images.last().ok_or(err!("no saved images"))?[i];
    let (width, height) = s.width_and_height(i);
    let (saved_width, saved_height) = tools::width_and_height(&saved_image);
    if width != saved_width || height != saved_height {
        return Err(err!("images have different sizes"));
    }
    let mut new_image: Image = (*s.images[i]).clone();
    for i in 0..new_image.len() {
        for j in 0..new_image[0].len() {
            let a = new_image[i][j];
            let b = saved_image[i][j];
            new_image[i][j] = func(a, b);
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn recolor_image_per_output(s: &mut SolverState) -> Res<()> {
    let used_colors = tools::get_used_colors(&s.output_images);
    if used_colors.len() != 1 {
        return Err(err!("output images have different colors"));
    }
    let color = used_colors[0];
    for image in &mut s.images {
        let new_image = image
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&c| if c == 0 { 0 } else { color })
                    .collect()
            })
            .collect();
        *image = Rc::new(new_image);
    }
    Ok(())
}

fn find_matching_offset(
    shapes: &ShapesPerExample,
    images: &ImagePerExample,
    direction: tools::Vec2,
    start_pos: tools::Vec2,
    min_offset: i32,
    max_offset: i32,
) -> Res<i32> {
    for distance in min_offset..=max_offset {
        if are_shapes_present_at(shapes, images, start_pos + distance * direction) {
            return Ok(distance);
        }
    }
    Err(err!("no match found"))
}

fn are_shapes_present_at(
    shapes: &ShapesPerExample,
    images: &ImagePerExample,
    pos: tools::Vec2,
) -> bool {
    for i in 0..images.len() {
        let image = &images[i];
        for shape in &shapes[i] {
            if !shape.matches_image_when_moved_by(image, pos) {
                return false;
            }
        }
    }
    true
}

fn repeat_shapes_on_lattice_per_output(s: &mut SolverState) -> Res<()> {
    repeat_shapes_on_lattice_per_reference(s, &s.output_images.clone())
}

fn repeat_shapes_on_lattice_per_image(s: &mut SolverState) -> Res<()> {
    repeat_shapes_on_lattice_per_reference(s, &s.images.clone())
}

fn repeat_shapes_on_lattice_per_reference(
    s: &mut SolverState,
    references: &ImagePerExample,
) -> Res<()> {
    // Make sure the shapes are not tiny.
    for shapes_per in &s.shapes {
        let total_cells: usize = shapes_per.iter().map(|s| s.cells.len()).sum();
        if total_cells < 4 {
            return Err(err!("shapes too small"));
        }
    }
    // Find a lattice. This is a periodic pattern described by two parameters,
    // the horizontal period (a single number) and the vertical offset (a Vec2).
    let horizontal_period = find_matching_offset(
        &s.shapes,
        references,
        tools::RIGHT,
        tools::Vec2::ZERO,
        1,
        10,
    )?;
    for vertical_y in 1..10 {
        for m in [1, -1].iter() {
            let vertical_y = vertical_y * *m;
            if let Ok(vertical_x) = find_matching_offset(
                &s.shapes,
                references,
                tools::RIGHT,
                vertical_y * tools::DOWN,
                -2,
                2,
            ) {
                if are_shapes_present_at(
                    &s.shapes,
                    references,
                    vertical_y * tools::DOWN + (vertical_x + horizontal_period) * tools::RIGHT,
                ) {
                    return repeat_shapes_on_lattice(s, horizontal_period, vertical_x, vertical_y);
                }
                if are_shapes_present_at(
                    &s.shapes,
                    references,
                    vertical_y * tools::DOWN + (vertical_x - horizontal_period) * tools::RIGHT,
                ) {
                    return repeat_shapes_on_lattice(s, horizontal_period, vertical_x, vertical_y);
                }
            }
        }
    }
    Err(err!("no match found"))
}

fn repeat_shapes_on_lattice(
    s: &mut SolverState,
    horizontal_period: i32,
    vertical_x: i32,
    vertical_y: i32,
) -> Res<()> {
    for i in 0..s.images.len() {
        let image = &s.images[i];
        let shapes = &s.shapes[i];
        let mut new_image = (**image).clone();
        for shape in shapes {
            for rep_x in -5..=5 {
                for rep_y in -5..=5 {
                    let pos = tools::Vec2 {
                        x: rep_x * horizontal_period + rep_y * vertical_x,
                        y: rep_y * vertical_y,
                    };
                    tools::draw_shape_at(&mut new_image, shape, pos);
                }
            }
        }
        s.images[i] = Rc::new(new_image);
    }
    Ok(())
}

fn repeat_shapes_vertically(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    for shape in shapes {
        let bb = shape.bb;
        *shape = shape.tile(0, width, bb.height(), height)?.into();
    }
    Ok(())
}

fn repeat_shapes_horizontally(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    let shapes = &mut s.shapes[i];
    for shape in shapes {
        let bb = shape.bb;
        *shape = shape.tile(bb.width(), width, 0, height)?.into();
    }
    Ok(())
}

/// Deletes the lines. Instead of erasing them, we completely remove them from the image.
fn remove_horizontal_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.lines[i].horizontal;
    let mut line_i = 0;
    let (_width, height) = s.width_and_height(i);
    for y in 0..height {
        if line_i < lines.len() && y == lines[line_i].pos {
            line_i += 1;
        } else {
            new_image.push(image[y as usize].clone());
        }
    }
    if new_image.len() == 0 {
        return Err(err!("nothing left"));
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn remove_vertical_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.lines[i].vertical;
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        let mut row = vec![];
        let mut line_i = 0;
        for x in 0..width {
            if line_i < lines.len() && x == lines[line_i].pos {
                line_i += 1;
            } else {
                row.push(image[y as usize][x as usize]);
            }
        }
        if row.len() == 0 {
            return Err(err!("nothing left"));
        }
        new_image.push(row);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn remove_grid(s: &mut SolverState, i: usize) -> Res<()> {
    let (width, height) = s.width_and_height(i);
    if width < 3 && height < 3 {
        return Err(err!("image too small"));
    }
    if s.lines[i].horizontal.is_empty() && s.lines[i].vertical.is_empty() {
        return Err(err!("no grid"));
    }
    remove_horizontal_lines(s, i)?;
    remove_vertical_lines(s, i)?;
    // Keep the horizontal and vertical lines so we can restore the grid later.
    let lines = std::mem::take(&mut s.lines);
    let images = std::mem::take(&mut s.images);
    s.init_from_images(images);
    s.saved_lines.push(lines);
    Ok(())
}

fn insert_horizontal_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.saved_lines;
    let lines = &lines[lines.len() - 1][i].horizontal;
    let mut line_i = 0;
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        while line_i < lines.len() && new_image.len() == lines[line_i].pos as usize {
            new_image.push(vec![lines[line_i].color; width as usize]);
            line_i += 1;
        }
        new_image.push(image[y as usize].clone());
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn insert_vertical_lines(s: &mut SolverState, i: usize) -> Res<()> {
    let image = &*s.images[i];
    let mut new_image = vec![];
    let lines = &s.saved_lines;
    let lines = &lines[lines.len() - 1][i].vertical;
    let (width, height) = s.width_and_height(i);
    for y in 0..height {
        let mut row = vec![];
        let mut line_i = 0;
        for x in 0..width {
            while line_i < lines.len() && row.len() == lines[line_i].pos as usize {
                row.push(lines[line_i].color);
                line_i += 1;
            }
            row.push(image[y as usize][x as usize]);
        }
        new_image.push(row);
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn restore_grid(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(&s.saved_lines);
    s.apply(|s: &mut SolverState, i: usize| {
        insert_horizontal_lines(s, i)?;
        insert_vertical_lines(s, i)?;
        Ok(())
    })?;
    s.saved_lines.pop();
    Ok(())
}

fn connect_aligned_pixels_in_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    let mut new_image = (*s.images[i]).clone();
    for shape in &s.shapes[i] {
        let bb = shape.bb;
        let shape_as_image = shape.as_image();
        for cell in &shape.cells {
            for dir in tools::DIRECTIONS4 {
                for distance in 1..10 {
                    let image_pos = *cell + distance * dir;
                    let shape_pos = image_pos - bb.top_left();
                    if tools::lookup_in_image(&shape_as_image, shape_pos.x, shape_pos.y)
                        .unwrap_or(0)
                        != 0
                    {
                        for d in 1..distance {
                            let pos = *cell + d * dir;
                            tools::set_in_image(&mut new_image, pos.x, pos.y, cell.color);
                        }
                    }
                }
            }
        }
    }
    s.images[i] = Rc::new(new_image);
    Ok(())
}

fn select_grid_cell_most_filled_in(s: &mut SolverState) -> Res<()> {
    select_grid_cell_max_by(s, |image| tools::count_non_zero_pixels(image) as i32)
}

fn select_grid_cell_least_filled_in(s: &mut SolverState) -> Res<()> {
    select_grid_cell_max_by(s, |image| -(tools::count_non_zero_pixels(image) as i32))
}

/// Adds up the width of all lines.
fn total_width(lines: &tools::Lines) -> usize {
    lines.iter().map(|l| l.width).sum()
}

fn select_grid_cell_max_by<F>(s: &mut SolverState, score_func: F) -> Res<()>
where
    F: Fn(&Image) -> i32,
{
    select_grid_cell(s, |grid_cells: &[Image]| {
        let mut best_index = 0;
        let mut best_score = std::i32::MIN;
        for (index, c) in grid_cells.iter().enumerate() {
            let score = score_func(c);
            if score > best_score {
                best_score = score;
                best_index = index;
            }
        }
        Ok(best_index)
    })
}

fn select_grid_cell_outlier_by_color(s: &mut SolverState) -> Res<()> {
    select_grid_cell(s, |grid_cells: &[Image]| {
        let mut users = vec![vec![]; COLORS.len()];
        for (index, c) in grid_cells.iter().enumerate() {
            let mut is_used = vec![false; COLORS.len()];
            tools::set_used_colors_in_image(c, &mut is_used);
            for (color, &used) in is_used.iter().enumerate() {
                if used {
                    users[color].push(index);
                }
            }
        }
        for users in users {
            if users.len() == 1 {
                return Ok(users[0]);
            }
        }
        Err(err!("no outlier found"))
    })
}

fn select_grid_cell<F>(s: &mut SolverState, select_func: F) -> Res<()>
where
    F: Fn(&[Image]) -> Res<usize>,
{
    s.apply(|s: &mut SolverState, i: usize| {
        let lines = &s.lines[i];
        if lines.horizontal.len() + lines.vertical.len() == 0 {
            return Err(err!("no grid"));
        }
        let (width, height) = s.width_and_height(i);
        if (width as usize) < (total_width(&lines.horizontal) + lines.horizontal.len() + 1)
            || (height as usize) < (total_width(&lines.vertical) + lines.vertical.len() + 1)
        {
            return Err(err!("image too small"));
        }
        let image = &s.images[i];
        let mut grid_cells = tools::grid_cut_image(image, &lines);
        assert!(!grid_cells.is_empty());
        let selected = select_func(&grid_cells)?;
        s.images[i] = grid_cells.swap_remove(selected).into();
        Ok(())
    })?;
    // Keep the horizontal and vertical lines so we can restore the grid later.
    let lines = std::mem::take(&mut s.lines);
    let images = std::mem::take(&mut s.images);
    s.init_from_images(images);
    s.saved_lines.push(lines);
    Ok(())
}

fn rotate_to_landscape_cw(s: &mut SolverState) -> Res<()> {
    rotate_to_landscape(s, tools::Rotation::CW)
}

fn rotate_to_landscape_ccw(s: &mut SolverState) -> Res<()> {
    rotate_to_landscape(s, tools::Rotation::CCW)
}

fn rotate_to_landscape(s: &mut SolverState, direction: tools::Rotation) -> Res<()> {
    let needs_rotating: Vec<bool> = (0..s.images.len())
        .map(|i| {
            let (w, h) = s.width_and_height(i);
            h > w
        })
        .collect();
    if needs_rotating.iter().all(|&rotate| !rotate) {
        return Err(err!("already landscape"));
    }
    let new_images = rotate_some_images(&s.images, &needs_rotating, direction);
    let new_output_images = rotate_some_images(&s.output_images, &needs_rotating, direction);
    s.output_images = new_output_images;
    s.init_from_images(new_images);
    s.add_finishing_step(move |s: &mut SolverState| {
        for (image, rotate) in s.images.iter_mut().zip(&needs_rotating) {
            if *rotate {
                *image = tools::rotate_image(image, direction.opposite()).into();
            }
        }
        Ok(())
    });
    Ok(())
}

fn rotate_some_images(
    images: &ImagePerExample,
    needs_rotating: &Vec<bool>,
    direction: tools::Rotation,
) -> ImagePerExample {
    images
        .iter()
        .zip(needs_rotating)
        .map(|(image, rotate)| {
            if *rotate {
                tools::rotate_image(image, direction).into()
            } else {
                (*image).clone()
            }
        })
        .collect()
}

fn refresh_from_image(s: &mut SolverState) -> Res<()> {
    let images = s.images.clone();
    s.init_from_images(images);
    Ok(())
}

fn allow_background_color_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.shapes_including_background.clone();
    s.output_shapes = s.output_shapes_including_background.clone();
    Ok(())
}

fn shapes_to_number_sequence(s: &mut SolverState, i: usize) -> Res<()> {
    if i == 0 {
        s.number_sequences = vec![vec![]; s.images.len()];
        s.output_number_sequences = vec![vec![]; s.output_images.len()];
    }
    // The input is easy. Just number the shapes.
    s.number_sequences[i] = (0..s.shapes[i].len() as i32).collect();
    if i < s.output_shapes.len() {
        // The output is harder. We need to find the shapes in the output image.
        let output_shapes = &s.output_shapes[i];
        // Then identify which of the input shapes they correspond to.
        let relative_input_shapes: Shapes = s.shapes[i]
            .iter()
            .map(|shape| shape.to_relative_pos().into())
            .collect();
        s.output_number_sequences[i] = output_shapes
            .iter()
            .map(|output_shape| {
                output_shape
                    .to_relative_pos()
                    .find_matching_shape_index(&relative_input_shapes)
                    .map(|index| index as i32)
                    .unwrap_or(-1)
            })
            .collect();
    }
    Ok(())
}

fn solve_number_sequence(s: &mut SolverState) -> Res<()> {
    must_not_be_empty!(s.number_sequences);
    must_not_be_empty!(s.output_number_sequences);
    // Simple placeholder for now.
    let longest_sequence = s
        .output_number_sequences
        .iter()
        .enumerate()
        .max_by_key(|(_i, seq)| seq.len())
        .map(|(i, _seq)| i)
        .ok_or(err!("no sequences"))?;
    for seq in &mut s.number_sequences {
        *seq = s.output_number_sequences[longest_sequence].clone();
    }
    Ok(())
}

/// Put the shapes after each other from left to right.
fn number_sequence_to_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    must_not_be_empty!(s.number_sequences);
    let relative_shapes: Vec<Shape> = s.shapes[i]
        .iter()
        .map(|shape| shape.to_relative_pos())
        .collect();
    let mut shapes: Shapes = vec![];
    let mut next_x = 0;
    for &index in &s.number_sequences[i] {
        if index < 0 {
            return Err(err!("invalid index"));
        }
        let shape = relative_shapes
            .get(index as usize)
            .ok_or(err!("no shape for index"))?
            .move_by(tools::Vec2 { x: next_x, y: 0 });
        next_x += shape.bb.width();
        shapes.push(shape.into());
    }
    s.shapes[i] = shapes;
    Ok(())
}

pub fn remap_colors_per_output(s: &mut SolverState) -> Res<()> {
    let mut color_map: Vec<i32> = vec![-1; COLORS.len()];
    for i in 0..s.output_images.len() {
        let input = &s.images[i];
        let output = &s.output_images[i];
        let (w, h) = tools::width_and_height(input);
        let (ow, oh) = tools::width_and_height(output);
        if w != ow || h != oh {
            return Err(err!("images have different sizes"));
        }
        for y in 0..h {
            for x in 0..w {
                let input_color = input[y as usize][x as usize];
                let output_color = output[y as usize][x as usize];
                if color_map[input_color as usize] == -1 {
                    color_map[input_color as usize] = output_color;
                } else if color_map[input_color as usize] != output_color {
                    return Err(err!("no consistent mapping"));
                }
            }
        }
    }
    let mut new_images: ImagePerExample = vec![];
    for i in 0..s.images.len() {
        let mut new_image = vec![];
        for row in &*s.images[i] {
            let mut new_row = vec![];
            for &cell in row {
                let c = color_map[cell as usize];
                new_row.push(if c == -1 { cell } else { c });
            }
            new_image.push(new_row);
        }
        new_images.push(new_image.into());
    }
    s.images = new_images;
    Ok(())
}

pub enum SolverStep {
    Each(&'static str, fn(&mut SolverState, usize) -> Res<()>),
    All(&'static str, fn(&mut SolverState) -> Res<()>),
}
macro_rules! step_all {
    ($func:ident) => {
        All(stringify!($func), $func)
    };
}
macro_rules! step_each {
    ($func:ident) => {
        Each(stringify!($func), $func)
    };
}
use SolverStep::*;
impl std::fmt::Display for SolverStep {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Each(name, _func) => write!(f, "{}", name),
            All(name, _func) => write!(f, "{}", name),
        }
    }
}
pub const ALL_STEPS: &[SolverStep] = &[
    step_all!(allow_background_color_shapes),
    step_all!(grow_flowers),
    step_all!(load_shapes_except_current_shapes),
    step_all!(move_shapes_per_output_shapes),
    step_all!(move_shapes_per_output),
    step_all!(recolor_image_per_output),
    step_all!(recolor_shapes_per_output),
    step_all!(refresh_from_image),
    step_all!(remap_colors_per_output),
    step_all!(repeat_shapes_on_lattice_per_image),
    step_all!(repeat_shapes_on_lattice_per_output),
    step_all!(restore_grid),
    step_all!(rotate_to_landscape_ccw),
    step_all!(rotate_to_landscape_cw),
    step_all!(save_first_shape_use_the_rest),
    step_all!(save_shapes_and_load_previous),
    step_all!(scale_up_image_add_grid),
    step_all!(scale_up_image),
    step_all!(select_grid_cell_least_filled_in),
    step_all!(select_grid_cell_most_filled_in),
    step_all!(select_grid_cell_outlier_by_color),
    step_all!(solve_number_sequence),
    step_all!(split_into_two_images),
    step_all!(substates_for_each_color),
    step_all!(substates_for_each_image),
    step_all!(substates_for_each_shape),
    step_all!(use_colorsets_as_shapes),
    step_all!(use_output_size),
    step_each!(allow_diagonals_in_shapes),
    step_each!(boolean_with_saved_image_and),
    step_each!(boolean_with_saved_image_or),
    step_each!(boolean_with_saved_image_xor),
    step_each!(connect_aligned_pixels_in_shapes),
    step_each!(delete_background_shapes),
    step_each!(delete_shapes_touching_border),
    step_each!(draw_shape_where_non_empty),
    step_each!(draw_shapes),
    step_each!(filter_shapes_by_color),
    step_each!(find_repeating_pattern_in_shape),
    step_each!(move_saved_shape_to_cover_current_shape_max),
    step_each!(move_shapes_to_touch_saved_shape),
    step_each!(number_sequence_to_shapes),
    step_each!(order_colors_by_shapes),
    step_each!(order_shapes_by_color),
    step_each!(order_shapes_by_size_decreasing),
    step_each!(order_shapes_by_size_increasing),
    step_each!(order_shapes_from_left_to_right),
    step_each!(pick_bottom_right_shape_per_color),
    step_each!(pick_bottom_right_shape),
    step_each!(recolor_saved_shapes_to_current_shape),
    step_each!(remove_grid),
    step_each!(repeat_last_move_and_draw),
    step_each!(repeat_shapes_horizontally),
    step_each!(repeat_shapes_vertically),
    step_each!(shapes_to_number_sequence),
    step_each!(tile_shapes_after_scale_up),
    step_each!(use_image_as_shape),
    step_each!(use_image_without_background_as_shape),
    step_each!(use_next_color),
    step_each!(use_previous_color),
];
pub const SOLVERS: &[&[SolverStep]] = &[
    &[
        // 0
        step_each!(use_image_as_shape),
        step_all!(scale_up_image),
        step_each!(tile_shapes_after_scale_up),
        step_each!(draw_shape_where_non_empty),
    ],
    &[
        // 1
        step_each!(filter_shapes_by_color),
        step_all!(recolor_shapes_per_output),
        step_each!(draw_shapes),
    ],
    &[
        // 2
        step_all!(use_output_size),
        step_all!(use_colorsets_as_shapes),
        step_each!(find_repeating_pattern_in_shape),
        step_all!(recolor_shapes_per_output),
        step_each!(draw_shapes),
    ],
    &[
        // 3
        step_each!(pick_bottom_right_shape_per_color),
        step_all!(load_shapes_except_current_shapes),
        step_each!(delete_background_shapes),
        step_all!(move_shapes_per_output),
    ],
    &[
        // 4
        step_each!(allow_diagonals_in_shapes),
        step_each!(delete_background_shapes),
        step_each!(order_shapes_by_size_decreasing),
        step_all!(save_first_shape_use_the_rest),
        step_all!(substates_for_each_shape),
        step_each!(recolor_saved_shapes_to_current_shape),
        step_each!(move_saved_shape_to_cover_current_shape_max),
        step_each!(repeat_last_move_and_draw),
    ],
    &[
        // 5
        step_all!(split_into_two_images),
        step_each!(boolean_with_saved_image_and),
        step_all!(recolor_image_per_output),
    ],
    &[
        // 6
        step_each!(use_image_without_background_as_shape),
        step_all!(repeat_shapes_on_lattice_per_output),
    ],
    &[
        // 7
        step_each!(use_next_color),
        step_each!(filter_shapes_by_color),
        step_all!(save_shapes_and_load_previous),
        step_each!(use_previous_color),
        step_each!(filter_shapes_by_color),
        step_each!(move_shapes_to_touch_saved_shape),
    ],
    &[
        // 8
        step_each!(remove_grid),
        step_all!(use_colorsets_as_shapes),
        step_each!(connect_aligned_pixels_in_shapes),
        step_all!(restore_grid),
    ],
    &[
        // 9
        step_each!(order_shapes_by_size_increasing),
        step_all!(recolor_shapes_per_output),
        step_each!(draw_shapes),
    ],
    &[
        // 10
        step_all!(select_grid_cell_least_filled_in),
        step_all!(scale_up_image_add_grid),
    ],
    &[
        // 11
        step_all!(use_colorsets_as_shapes),
        step_each!(order_shapes_by_size_increasing),
        step_each!(order_colors_by_shapes),
        step_each!(filter_shapes_by_color),
        step_all!(grow_flowers),
    ],
    &[
        // 12
        step_all!(rotate_to_landscape_ccw),
        step_each!(repeat_shapes_vertically),
        step_each!(draw_shapes),
        step_all!(refresh_from_image),
        step_all!(allow_background_color_shapes),
        step_each!(shapes_to_number_sequence),
        step_all!(solve_number_sequence),
        step_each!(number_sequence_to_shapes),
        step_each!(draw_shapes),
    ],
    &[
        // 13
        step_all!(select_grid_cell_outlier_by_color),
    ],
    &[
        // 14
        step_all!(use_colorsets_as_shapes),
        step_all!(substates_for_each_color),
        step_each!(filter_shapes_by_color),
        step_all!(grow_flowers),
    ],
    &[
        // 15
        step_all!(remap_colors_per_output),
    ],
    &[
        // 16
        step_each!(use_image_without_background_as_shape),
        step_all!(substates_for_each_image),
        step_all!(repeat_shapes_on_lattice_per_image),
    ],
    &[
        // 71
        step_all!(split_into_two_images),
        step_each!(boolean_with_saved_image_xor),
        step_all!(recolor_image_per_output),
    ],
    &[
        // ???
        step_all!(split_into_two_images),
        step_each!(boolean_with_saved_image_or),
        step_all!(recolor_image_per_output),
    ],
    &[
        // unused
        step_each!(delete_shapes_touching_border),
        step_each!(order_shapes_by_color),
        step_each!(order_shapes_from_left_to_right),
        step_each!(pick_bottom_right_shape),
        step_all!(move_shapes_per_output_shapes),
        step_each!(repeat_shapes_horizontally),
        step_all!(select_grid_cell_most_filled_in),
        step_all!(rotate_to_landscape_cw),
    ],
];
