use crate::tools;
use crate::tools::{COLORS, Task, Example, Image, Shape, Res};

pub fn find_shapes(s: &mut SolverState, i: usize) -> Res<()> {
    if s.shapes.is_none() {
        s.shapes = Some(vec![vec![]; s.images.len()]);
    }
    s.shapes.as_mut().unwrap()[i] = tools::find_shapes_in_image(&s.images[i]);
    Ok(())
}

pub fn find_colorsets(s: &mut SolverState, i: usize) -> Res<()> {
    if s.colorsets.is_none() {
        s.colorsets = Some(vec![vec![]; s.images.len()]);
    }
    s.colorsets.as_mut().unwrap()[i] = tools::find_colorsets_in_image(&s.images[i]);
    Ok(())
}

pub fn use_colorsets_as_shapes(s: &mut SolverState) -> Res<()> {
    s.shapes = s.colorsets.clone();
    Ok(())
}

pub fn sort_shapes_by_size(s: &mut SolverState, i: usize) -> Res<()> {
    let shapes = &mut s.shapes.as_mut().expect("must have shapes")[i];
    *shapes = tools::sort_shapes_by_size(shapes.clone());
    Ok(())
}

fn remap_colors(s: &mut SolverState, i: usize, mapping: &[i32]) {
  tools::remap_colors_in_image(&mut s.images[i], mapping);
    if i < s.output_images.len() {
      tools::remap_colors_in_image(&mut s.output_images[i], mapping);
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
        s.used_colors = tools::get_used_colors(&s.images);
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
    let output_pattern = tools::find_pattern_around(&s.output_images, &dots);
    // TODO: Instead of growing each dot, we should filter by the input_pattern.
    s.apply(|s: &mut SolverState, i: usize| {
        let shapes = &s.shapes.as_ref().expect("must have shapes");
        let dots = &shapes[i][0];
        for dot in dots.cells.iter() {
          tools::draw_pattern_at(&mut s.images[i], dot, &output_pattern);
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
        let used_colors = tools::get_used_colors(&images);
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
    let shape = tools::shape_by_color(&shapes, s.used_colors[0]).ok_or("should have been a shape")?;
    if s.picked_shapes.is_none() {
        s.picked_shapes = Some(vec![vec![]; s.images.len()]);
    }
    s.picked_shapes.as_mut().unwrap()[i] = vec![shape.clone()];
    Ok(())
}

fn move_picked_shape_to_saved_shape(s: &mut SolverState, i: usize) -> Res<()> {
    let picked_shapes = &s.picked_shapes.as_ref().expect("must have picked shapes")[i];
    let saved_shapes = &s.saved_shapes.as_ref().expect("must have saved shapes")[i];
    s.images[i] = tools::move_shape_to_shape_in_image(
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
pub const SOLVERS: &[Solver] = &[solve_example_0, solve_example_7, solve_example_11];