use crate::steps::*;
use crate::tools;
use crate::tools::{Color, Example, Image, Res, Shape, Task, Vec2, COLORS};
use std::rc::Rc;

macro_rules! err {
    ($msg:expr) => {
        concat!($msg, " at ", file!(), ":", line!())
    };
}

pub type Shapes = Vec<Shape>;
pub type ShapesPerExample = Vec<Shapes>;
pub type ImagePerExample = Vec<Image>;
pub type ColorListPerExample = Vec<ColorList>;
pub type LinesPerExample = Vec<Rc<tools::LineSet>>;
pub type NumberSequence = Vec<i32>;
pub type NumberSequencesPerExample = Vec<NumberSequence>;

/// A sub-state that contains only one image and its shapes.
/// It uses the image from the parent state and applies operations on it.
#[derive(Default, Clone)]
pub struct SubState {
    pub state: SolverState,
    pub image_indexes: Vec<usize>,
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
    pub multicolor_shapes: ShapesPerExample,
    pub output_shapes: ShapesPerExample,
    pub output_shapes_including_background: ShapesPerExample,
    pub saved_shapes: Vec<ShapesPerExample>,
    pub colorsets: ShapesPerExample,
    pub scale_up: Vec2,
    pub last_move: Vec<Vec2>,
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
        let mut state = SolverState {
            task: Rc::new(task.clone()),
            output_images,
            last_move: vec![Vec2::ZERO; images.len()],
            ..Default::default()
        };
        let all_colors: ColorList = (0..COLORS.len()).collect();
        state.colors = vec![all_colors; images.len()];
        state.init_from_images(images);
        state.apply(order_colors_by_shapes).unwrap();
        save_image(&mut state).unwrap();
        state
    }

    pub fn init_from_images(&mut self, images: ImagePerExample) {
        self.images = images;
        self.init_from_current_images();
    }
    pub fn init_from_current_images(&mut self) {
        self.colorsets = self
            .images
            .iter()
            .map(|image| tools::find_colorsets_in_image(image))
            .collect();
        self.shapes_including_background = self
            .images
            .iter()
            .map(|image| tools::find_shapes_in_image(image, &Vec2::DIRECTIONS4))
            .collect();
        self.multicolor_shapes = self
            .images
            .iter()
            .map(|image| tools::find_multicolor_shapes_in_image(image, &Vec2::DIRECTIONS4))
            .collect();
        self.output_shapes_including_background = self
            .output_images
            .iter()
            .map(|image| tools::find_shapes_in_image(image, &Vec2::DIRECTIONS4))
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
        for s in &mut self.shapes {
            s.sort_by_key(|shape| shape.color());
        }
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
        if !self.output_images.is_empty() {
            if self.output_images.iter().any(|image| image.is_empty()) {
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

    pub fn apply<F>(&mut self, f: F) -> Res<()>
    where
        F: Fn(&mut SolverState, usize) -> Res<()>,
    {
        let mut any_change = false;
        for i in 0..self.images.len() {
            let res = f(self, i);
            match res {
                Ok(()) => any_change = true,
                Err("no change") => continue,
                err => return err,
            }
        }
        if any_change {
            Ok(())
        } else {
            Err("no change")
        }
    }

    pub fn get_results(&self) -> Vec<Example> {
        fn images_to_examples(state: &SolverState) -> Vec<Example> {
            state
                .images
                .iter()
                .zip(state.task.train.iter().chain(state.task.test.iter()))
                .map(|(image, example)| Example {
                    input: example.input.clone(),
                    output: image.full(),
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
    pub fn width_and_height(&self, i: usize) -> (i32, i32) {
        tools::width_and_height(&self.images[i])
    }
    pub fn width_and_height_all(&self) -> Res<(i32, i32)> {
        let (w, h) = self.width_and_height(0);
        for i in 1..self.images.len() {
            let (w1, h1) = self.width_and_height(i);
            if w1 != w || h1 != h {
                return Err("images have different sizes");
            }
        }
        Ok((w, h))
    }
    pub fn output_width_and_height(&self, i: usize) -> (i32, i32) {
        tools::width_and_height(&self.output_images[i])
    }
    pub fn output_width_and_height_all(&self) -> Res<(i32, i32)> {
        if self.output_images.is_empty() {
            return Err("no output images");
        }
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
    pub fn print_shapes(&self) {
        print_shapes(&self.shapes);
    }
    #[allow(dead_code)]
    pub fn print_saved_shapes(&self) {
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
    pub fn print_colorsets(&self) {
        print_shapes(&self.colorsets);
    }
    #[allow(dead_code)]
    pub fn print_images(&self) {
        for image in &self.images {
            println!(
                "image {} {} {} {}",
                image.left, image.top, image.width, image.height
            );
            image.print();
            println!();
        }
    }

    #[allow(dead_code)]
    pub fn print_colors(&self) {
        for colors in &self.colors {
            tools::print_colors(colors);
            println!();
        }
    }

    #[allow(dead_code)]
    pub fn print_steps(&self) {
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
                    multicolor_shapes: vec![self.multicolor_shapes[i].clone()],
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
            let mut any_change = false;
            let mut res = Ok(());
            for substate in substates {
                let state = &mut substate.state;
                state.images = substate
                    .image_indexes
                    .iter()
                    .enumerate()
                    .map(|(i, &image_index)| self.images[image_index].at(&state.images[i]))
                    .collect();
                res = state.run_step(&step);
                if res.is_ok() {
                    any_change = true;
                    for (i, &image_index) in substate.image_indexes.iter().enumerate() {
                        self.images[image_index] = substate.state.images[i].full();
                    }
                }
            }
            if any_change {
                return Ok(());
            }
            return res;
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
            .all(|(image, output)| image == output)
    }

    pub fn add_finishing_step(&mut self, f: impl Fn(&mut SolverState) -> Res<()> + 'static) {
        self.finishing_steps.push(Rc::new(Box::new(f)));
    }

    pub fn shift_shapes(&mut self, i: usize, offset: Vec2) {
        let fields = [
            &mut self.shapes,
            &mut self.shapes_including_background,
            &mut self.colorsets,
            &mut self.multicolor_shapes,
        ];
        for field in fields {
            for shape in &mut field[i] {
                let new_shape = shape.move_by(offset).into();
                *shape = new_shape;
            }
        }
    }
    pub fn forget_all_lines(&mut self) {
        self.lines = vec![Rc::new(tools::LineSet::default()); self.images.len()];
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
impl std::fmt::Debug for SolverStep {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}
pub const ALL_STEPS: &[SolverStep] = &[
    step_all!(allow_background_color_shapes),
    step_all!(dots_to_lines_per_output),
    step_all!(draw_saved_image),
    step_all!(follow_shape_sequence_per_output),
    step_all!(grow_flowers_square),
    step_all!(load_shapes_except_current_shapes),
    step_all!(make_common_output_image),
    step_all!(move_shapes_per_output_shapes),
    step_all!(move_shapes_per_output),
    step_all!(order_colors_by_frequency_across_images_ascending),
    step_all!(order_colors_by_frequency_across_images_descending),
    step_all!(recolor_image_per_output),
    step_all!(recolor_shapes_per_output),
    step_all!(refresh_from_image),
    step_all!(remap_colors_per_output),
    step_all!(remove_grid),
    step_all!(repeat_shapes_on_lattice_per_image),
    step_all!(repeat_shapes_on_lattice_per_output),
    step_all!(restore_grid),
    step_all!(rotate_to_landscape_ccw),
    step_all!(rotate_to_landscape_cw),
    step_all!(save_first_shape_use_the_rest),
    step_all!(save_image_and_load_previous),
    step_all!(save_shapes_and_load_previous),
    step_all!(scale_up_image_add_grid),
    step_all!(scale_up_image),
    step_all!(select_grid_cell_least_filled_in),
    step_all!(select_grid_cell_most_filled_in),
    step_all!(select_grid_cell_outlier_by_color),
    step_all!(split_into_two_images),
    step_all!(substates_for_each_color),
    step_all!(substates_for_each_image),
    step_all!(substates_for_each_shape),
    step_all!(take_all_shapes_from_output),
    step_all!(tile_image_add_grid),
    step_all!(tile_image),
    step_all!(use_colorsets_as_shapes),
    step_all!(use_first_shape_save_the_rest),
    step_all!(use_multicolor_shapes),
    step_all!(use_output_size),
    step_all!(use_relative_colors),
    step_each!(align_shapes_to_saved_shape_horizontal),
    step_each!(allow_diagonals_in_multicolor_shapes),
    step_each!(allow_diagonals_in_shapes),
    step_each!(atomize_shapes),
    step_each!(boolean_with_saved_image_and),
    step_each!(boolean_with_saved_image_nand),
    step_each!(boolean_with_saved_image_nor),
    step_each!(boolean_with_saved_image_or),
    step_each!(boolean_with_saved_image_xor),
    step_each!(connect_aligned_pixels_in_shapes_4),
    step_each!(connect_aligned_pixels_in_shapes_8),
    step_each!(cover_image_with_shapes),
    step_each!(crop_to_shape),
    step_each!(crop_to_top_left_quadrant),
    step_each!(deduplicate_horizontally),
    step_each!(deduplicate_vertically),
    step_each!(delete_noise),
    step_each!(discard_background_shapes),
    step_each!(discard_shapes_touching_border),
    step_each!(discard_small_shapes),
    step_each!(draw_shape_where_non_empty),
    step_each!(draw_shapes),
    step_each!(drop_all_pixels_down),
    step_each!(erase_shapes),
    step_each!(extend_zoom_up_left_until_square),
    step_each!(filter_shapes_by_color),
    step_each!(find_repeating_pattern_in_shape),
    step_each!(inset_by_one),
    step_each!(keep_only_border_lines),
    step_each!(make_image_rotationally_symmetrical),
    step_each!(make_image_symmetrical),
    step_each!(move_current_shape_to_be_inside_saved_shape_min_4),
    step_each!(move_saved_shape_to_cover_current_shape_max_4),
    step_each!(move_saved_shape_to_cover_current_shape_max_8),
    step_each!(move_saved_shape_to_cover_current_shape_max_diagonally),
    step_each!(move_shapes_to_touch_saved_shape),
    step_each!(order_colors_by_shapes),
    step_each!(order_shapes_by_bb_size_decreasing),
    step_each!(order_shapes_by_bb_size_increasing),
    step_each!(order_shapes_by_color),
    step_each!(order_shapes_by_weight_decreasing),
    step_each!(order_shapes_by_weight_increasing),
    step_each!(order_shapes_from_left_to_right),
    step_each!(order_shapes_from_top_to_bottom),
    step_each!(pick_bottom_right_shape_per_color),
    step_each!(pick_bottom_right_shape),
    step_each!(place_shapes_best_match_with_all_transforms),
    step_each!(place_shapes_best_match_with_just_translation),
    step_each!(recolor_image_to_selected_color),
    step_each!(recolor_saved_shapes_to_current_shape),
    step_each!(recolor_shapes_to_nearest_saved_shape),
    step_each!(recolor_shapes_to_selected_color),
    step_each!(repeat_last_move_and_draw),
    step_each!(repeat_shapes_horizontally),
    step_each!(repeat_shapes_vertically),
    step_each!(reset_zoom),
    step_each!(select_next_color),
    step_each!(select_previous_color),
    step_each!(tile_shapes_after_scale_up),
    step_each!(use_image_as_shape),
    step_each!(use_image_without_background_as_shape),
    step_each!(zoom_to_content),
];
pub const SOLVERS: &[&[SolverStep]] = &[
    &[
        // 0
        step_all!(scale_up_image),
        step_all!(save_image_and_load_previous),
        step_all!(tile_image),
        step_each!(boolean_with_saved_image_and),
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
        step_each!(discard_background_shapes),
        step_all!(move_shapes_per_output),
    ],
    &[
        // 4
        step_each!(allow_diagonals_in_shapes),
        step_each!(discard_background_shapes),
        step_each!(order_shapes_by_weight_decreasing),
        step_all!(save_first_shape_use_the_rest),
        step_all!(substates_for_each_shape),
        step_each!(recolor_saved_shapes_to_current_shape),
        step_each!(move_saved_shape_to_cover_current_shape_max_8),
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
        step_each!(select_next_color),
        step_each!(filter_shapes_by_color),
        step_all!(save_shapes_and_load_previous),
        step_each!(select_previous_color),
        step_each!(filter_shapes_by_color),
        step_each!(move_shapes_to_touch_saved_shape),
    ],
    &[
        // 8
        step_all!(remove_grid),
        step_all!(use_colorsets_as_shapes),
        step_each!(connect_aligned_pixels_in_shapes_4),
        step_all!(restore_grid),
    ],
    &[
        // 9
        step_each!(order_shapes_by_weight_increasing),
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
        step_each!(order_shapes_by_weight_increasing),
        step_each!(order_colors_by_shapes),
        step_all!(use_relative_colors),
        step_each!(filter_shapes_by_color),
        step_all!(grow_flowers_square),
    ],
    &[
        // 12
        step_all!(rotate_to_landscape_ccw),
        step_each!(repeat_shapes_vertically),
        step_each!(draw_shapes),
        step_all!(refresh_from_image),
        step_all!(allow_background_color_shapes),
        step_all!(follow_shape_sequence_per_output),
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
        step_all!(grow_flowers_square),
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
        // 17
        step_all!(use_multicolor_shapes),
        step_each!(discard_small_shapes),
        step_each!(erase_shapes),
        step_each!(place_shapes_best_match_with_all_transforms),
        step_each!(draw_shapes),
    ],
    &[
        // 18
        step_all!(tile_image),
        step_all!(refresh_from_image),
        step_all!(use_colorsets_as_shapes),
        step_all!(grow_flowers_square),
    ],
    &[
        // 19
        step_each!(keep_only_border_lines),
        step_all!(remove_grid),
        step_each!(make_image_rotationally_symmetrical),
        step_all!(restore_grid),
    ],
    &[
        // 20
        step_each!(deduplicate_horizontally),
        step_each!(deduplicate_vertically),
        step_all!(refresh_from_image),
        step_all!(remove_grid),
    ],
    &[
        // 21
        step_each!(allow_diagonals_in_multicolor_shapes),
        step_all!(make_common_output_image),
        step_each!(place_shapes_best_match_with_just_translation),
        step_each!(draw_shapes),
    ],
    &[
        // 22
        step_all!(take_all_shapes_from_output),
        step_each!(cover_image_with_shapes),
        step_each!(draw_shapes),
    ],
    &[
        // 23
        step_all!(dots_to_lines_per_output),
    ],
    &[
        // 24
        step_each!(delete_noise),
        step_all!(refresh_from_image),
        step_all!(substates_for_each_image),
        step_all!(substates_for_each_color),
        step_each!(filter_shapes_by_color),
        step_each!(order_shapes_by_weight_decreasing),
        step_all!(save_first_shape_use_the_rest),
        step_each!(move_shapes_to_touch_saved_shape),
    ],
    &[
        // 25
        step_all!(split_into_two_images),
        step_each!(boolean_with_saved_image_nand),
        step_all!(recolor_image_per_output),
    ],
    &[
        // 26
        step_each!(zoom_to_content),
        step_each!(extend_zoom_up_left_until_square),
        step_each!(make_image_rotationally_symmetrical),
        step_each!(select_previous_color),
        step_each!(recolor_image_to_selected_color),
        step_each!(draw_shapes),
        step_each!(reset_zoom),
        step_all!(remap_colors_per_output),
    ],
    &[
        // 27
        step_each!(order_shapes_from_top_to_bottom),
        step_each!(order_colors_by_shapes),
        step_all!(use_relative_colors),
        step_all!(make_common_output_image),
    ],
    &[
        // 28
        step_all!(use_colorsets_as_shapes),
        step_each!(order_shapes_by_bb_size_increasing),
        step_all!(use_first_shape_save_the_rest),
        step_each!(crop_to_shape),
        step_each!(inset_by_one),
    ],
    &[
        // 29
        step_all!(use_colorsets_as_shapes),
        step_all!(save_first_shape_use_the_rest),
        step_each!(erase_shapes),
        step_each!(align_shapes_to_saved_shape_horizontal),
        step_each!(draw_shapes),
    ],
    &[
        // ???
        step_all!(split_into_two_images),
        step_each!(boolean_with_saved_image_or),
        step_all!(recolor_image_per_output),
    ],
    &[
        // 31
        step_each!(drop_all_pixels_down),
    ],
    &[
        // 32
        step_each!(order_shapes_by_bb_size_decreasing),
        step_each!(order_colors_by_shapes),
        step_all!(select_grid_cell_most_filled_in),
        step_all!(tile_image_add_grid),
        step_each!(recolor_image_to_selected_color),
        step_all!(draw_saved_image),
    ],
    &[
        // 33
        step_all!(order_colors_by_frequency_across_images_ascending),
        step_all!(use_multicolor_shapes),
        step_each!(recolor_shapes_to_selected_color),
        step_all!(save_shapes_and_load_previous),
        step_each!(select_next_color),
        step_each!(filter_shapes_by_color),
        step_each!(atomize_shapes),
        step_all!(substates_for_each_shape),
        step_each!(move_saved_shape_to_cover_current_shape_max_diagonally),
        step_each!(repeat_last_move_and_draw),
    ],
    &[
        // 34
        step_each!(order_shapes_by_weight_decreasing),
        step_all!(save_first_shape_use_the_rest),
        step_each!(move_current_shape_to_be_inside_saved_shape_min_4),
        step_each!(draw_shapes),
    ],
    &[
        // 35
        step_each!(order_shapes_by_bb_size_decreasing),
        step_all!(use_first_shape_save_the_rest),
        step_each!(crop_to_shape),
    ],
    &[
        // 36
        step_all!(use_colorsets_as_shapes),
        step_each!(connect_aligned_pixels_in_shapes_8),
    ],
    &[
        // 37
        // counting
    ],
    &[
        // 38
        step_each!(use_image_as_shape),
        step_each!(crop_to_shape),
        step_each!(crop_to_top_left_quadrant),
    ],
    &[
        // 39
        step_all!(use_colorsets_as_shapes),
        step_all!(order_colors_by_frequency_across_images_descending),
        step_each!(order_shapes_by_color),
        step_all!(use_first_shape_save_the_rest),
        step_each!(atomize_shapes),
        step_each!(recolor_shapes_to_nearest_saved_shape),
        step_each!(draw_shapes),
    ],
    &[
        // Bad solution for 4c4377d9, 6d0aefbc, 963e52fc.
        step_all!(tile_image),
    ],
    &[
        // 71
        step_all!(split_into_two_images),
        step_each!(boolean_with_saved_image_xor),
        step_all!(recolor_image_per_output),
    ],
    &[
        // ded97339
        step_all!(use_colorsets_as_shapes),
        step_each!(connect_aligned_pixels_in_shapes_4),
    ],
    &[
        // 1cf80156
        step_all!(remove_grid),
    ],
    &[
        // unused
        step_all!(move_shapes_per_output_shapes),
        step_all!(rotate_to_landscape_cw),
        step_all!(select_grid_cell_most_filled_in),
        step_each!(boolean_with_saved_image_nor),
        step_each!(discard_shapes_touching_border),
        step_each!(make_image_symmetrical),
        step_each!(order_shapes_from_left_to_right),
        step_each!(order_shapes_by_bb_size_decreasing),
        step_each!(pick_bottom_right_shape),
        step_each!(place_shapes_best_match_with_just_translation),
        step_each!(repeat_shapes_horizontally),
    ],
];
